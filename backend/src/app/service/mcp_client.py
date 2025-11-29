"""
MCP (Model Context Protocol) Client Service

This module provides functionality to connect to external MCP servers
and use their tools within the agent.
"""

import asyncio
import json
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


@dataclass
class MCPServer:
    """Represents a connected MCP server"""
    name: str
    url: str
    description: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    error: Optional[str] = None


class MCPToolInput(BaseModel):
    """Generic input model for MCP tools"""
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the MCP tool")


class MCPClientManager:
    """
    Manages connections to MCP servers and provides tools to the agent.
    
    Supports both HTTP/REST-based MCP servers and SSE-based servers.
    """
    
    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}
        self._tools: Dict[str, StructuredTool] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def close(self):
        """Close the HTTP client"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
    
    async def add_server(self, name: str, url: str, description: str = "") -> MCPServer:
        """
        Add and connect to an MCP server.
        
        Args:
            name: Unique name for this server connection
            url: Base URL of the MCP server
            description: Optional description of the server
            
        Returns:
            MCPServer instance with connection status
        """
        # Normalize URL
        url = url.rstrip('/')
        
        server = MCPServer(
            name=name,
            url=url,
            description=description
        )
        
        try:
            # Try to discover tools from the server
            tools = await self._discover_tools(url)
            server.tools = tools
            server.connected = True
            
            # Create LangChain tools for each discovered tool
            for tool_spec in tools:
                lc_tool = self._create_langchain_tool(server, tool_spec)
                tool_key = f"{name}_{tool_spec['name']}"
                self._tools[tool_key] = lc_tool
                
            print(f"[MCP] Connected to server '{name}' at {url}, discovered {len(tools)} tools")
            
        except Exception as e:
            server.connected = False
            server.error = str(e)
            print(f"[MCP] Failed to connect to server '{name}': {e}")
        
        self._servers[name] = server
        return server
    
    async def _discover_tools(self, base_url: str) -> List[Dict[str, Any]]:
        """
        Discover available tools from an MCP server.
        
        Tries multiple discovery methods:
        1. Standard MCP tools/list endpoint
        2. OpenAPI/Swagger spec
        3. Custom tools endpoint
        """
        client = await self._get_client()
        tools = []
        
        # Try standard MCP tools discovery
        discovery_endpoints = [
            f"{base_url}/tools/list",
            f"{base_url}/mcp/tools",
            f"{base_url}/api/tools",
            f"{base_url}/tools",
        ]
        
        for endpoint in discovery_endpoints:
            try:
                response = await client.post(endpoint, json={})
                if response.status_code == 200:
                    data = response.json()
                    # Handle different response formats
                    if isinstance(data, list):
                        tools = data
                    elif isinstance(data, dict):
                        tools = data.get('tools', data.get('result', data.get('data', [])))
                    if tools:
                        print(f"[MCP] Discovered tools from {endpoint}")
                        break
            except Exception:
                pass
            
            # Also try GET method
            try:
                response = await client.get(endpoint)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        tools = data
                    elif isinstance(data, dict):
                        tools = data.get('tools', data.get('result', data.get('data', [])))
                    if tools:
                        print(f"[MCP] Discovered tools from {endpoint} (GET)")
                        break
            except Exception:
                pass
        
        # Try OpenAPI spec if no tools found
        if not tools:
            try:
                for spec_endpoint in [f"{base_url}/openapi.json", f"{base_url}/swagger.json"]:
                    response = await client.get(spec_endpoint)
                    if response.status_code == 200:
                        spec = response.json()
                        tools = self._parse_openapi_spec(spec, base_url)
                        if tools:
                            print(f"[MCP] Parsed tools from OpenAPI spec at {spec_endpoint}")
                            break
            except Exception:
                pass
        
        return tools
    
    def _parse_openapi_spec(self, spec: Dict, base_url: str) -> List[Dict[str, Any]]:
        """Parse OpenAPI spec to extract tool definitions"""
        tools = []
        paths = spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() not in ['get', 'post', 'put', 'delete']:
                    continue
                    
                tool_name = details.get('operationId', f"{method}_{path.replace('/', '_')}")
                tool_description = details.get('summary', details.get('description', f"{method.upper()} {path}"))
                
                # Extract parameters
                parameters = {}
                for param in details.get('parameters', []):
                    param_name = param.get('name')
                    param_schema = param.get('schema', {})
                    parameters[param_name] = {
                        'type': param_schema.get('type', 'string'),
                        'description': param.get('description', ''),
                        'required': param.get('required', False)
                    }
                
                # Handle request body
                request_body = details.get('requestBody', {})
                if request_body:
                    content = request_body.get('content', {})
                    json_content = content.get('application/json', {})
                    schema = json_content.get('schema', {})
                    if 'properties' in schema:
                        for prop_name, prop_schema in schema['properties'].items():
                            parameters[prop_name] = {
                                'type': prop_schema.get('type', 'string'),
                                'description': prop_schema.get('description', ''),
                                'required': prop_name in schema.get('required', [])
                            }
                
                tools.append({
                    'name': tool_name,
                    'description': tool_description,
                    'inputSchema': {
                        'type': 'object',
                        'properties': parameters
                    },
                    '_endpoint': path,
                    '_method': method.upper()
                })
        
        return tools
    
    def _create_langchain_tool(self, server: MCPServer, tool_spec: Dict[str, Any]) -> StructuredTool:
        """Create a LangChain tool from an MCP tool specification"""
        
        tool_name = tool_spec['name']
        tool_description = tool_spec.get('description', f"Tool from {server.name}")
        input_schema = tool_spec.get('inputSchema', {})
        
        # Create the async tool function
        async def call_mcp_tool(**kwargs) -> str:
            return await self._execute_tool(server, tool_spec, kwargs)
        
        # Create sync wrapper for LangChain compatibility
        def call_mcp_tool_sync(**kwargs) -> str:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_tool(server, tool_spec, kwargs)
                    )
                    return future.result()
            else:
                return asyncio.run(self._execute_tool(server, tool_spec, kwargs))
        
        # Build description with parameter info
        full_description = f"[MCP:{server.name}] {tool_description}"
        if input_schema.get('properties'):
            params_desc = ", ".join([
                f"{k}: {v.get('type', 'any')}" 
                for k, v in input_schema.get('properties', {}).items()
            ])
            full_description += f"\nParameters: {params_desc}"
        
        return StructuredTool.from_function(
            func=call_mcp_tool_sync,
            coroutine=call_mcp_tool,
            name=f"mcp_{server.name}_{tool_name}",
            description=full_description
        )
    
    async def _execute_tool(
        self, 
        server: MCPServer, 
        tool_spec: Dict[str, Any], 
        arguments: Dict[str, Any]
    ) -> str:
        """Execute a tool on an MCP server"""
        client = await self._get_client()
        tool_name = tool_spec['name']
        
        print(f"    [MCP Tool] Calling {server.name}/{tool_name} with args: {arguments}")
        
        # Try standard MCP tool call endpoint
        call_endpoints = [
            (f"{server.url}/tools/call", "POST"),
            (f"{server.url}/mcp/call", "POST"),
            (f"{server.url}/api/tools/{tool_name}", "POST"),
        ]
        
        # If tool has explicit endpoint from OpenAPI parsing
        if '_endpoint' in tool_spec:
            endpoint = f"{server.url}{tool_spec['_endpoint']}"
            method = tool_spec.get('_method', 'POST')
            call_endpoints.insert(0, (endpoint, method))
        
        last_error = None
        
        for endpoint, method in call_endpoints:
            try:
                if method == "POST":
                    # Standard MCP format
                    payload = {
                        "name": tool_name,
                        "arguments": arguments
                    }
                    response = await client.post(endpoint, json=payload)
                elif method == "GET":
                    response = await client.get(endpoint, params=arguments)
                else:
                    response = await client.request(method, endpoint, json=arguments)
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle different response formats
                    if isinstance(result, dict):
                        content = result.get('content', result.get('result', result.get('data', result)))
                        if isinstance(content, list) and len(content) > 0:
                            # MCP format: content is array of content blocks
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            return '\n'.join(text_parts) if text_parts else json.dumps(content)
                        return json.dumps(content) if not isinstance(content, str) else content
                    return json.dumps(result) if not isinstance(result, str) else result
                    
            except Exception as e:
                last_error = e
                continue
        
        error_msg = f"Failed to execute tool {tool_name}: {last_error}"
        print(f"    [MCP Tool] {error_msg}")
        return error_msg
    
    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server connection"""
        if name not in self._servers:
            return False
        
        server = self._servers[name]
        
        # Remove associated tools
        tools_to_remove = [
            key for key in self._tools.keys() 
            if key.startswith(f"{name}_")
        ]
        for tool_key in tools_to_remove:
            del self._tools[tool_key]
        
        del self._servers[name]
        print(f"[MCP] Disconnected from server '{name}'")
        return True
    
    def get_servers(self) -> List[Dict[str, Any]]:
        """Get list of all connected servers"""
        return [
            {
                'name': server.name,
                'url': server.url,
                'description': server.description,
                'connected': server.connected,
                'error': server.error,
                'tools': [
                    {'name': t['name'], 'description': t.get('description', '')}
                    for t in server.tools
                ]
            }
            for server in self._servers.values()
        ]
    
    def get_tools(self) -> List[StructuredTool]:
        """Get all tools from connected MCP servers"""
        return list(self._tools.values())
    
    def get_tools_for_agent(self) -> List[StructuredTool]:
        """Get tools formatted for use with the agent"""
        return list(self._tools.values())


# Global MCP client manager instance
mcp_manager = MCPClientManager()
