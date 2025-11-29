"""
MCP Server Management API Controller

Provides endpoints for managing MCP server connections.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from app.service.mcp_client import mcp_manager


mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


class AddServerRequest(BaseModel):
    """Request model for adding an MCP server"""
    name: str = Field(..., description="Unique name for this server connection")
    url: str = Field(..., description="Base URL of the MCP server")
    description: Optional[str] = Field("", description="Optional description of the server")


class RemoveServerRequest(BaseModel):
    """Request model for removing an MCP server"""
    name: str = Field(..., description="Name of the server to remove")


class ServerResponse(BaseModel):
    """Response model for server information"""
    name: str
    url: str
    description: str
    connected: bool
    error: Optional[str] = None
    tools: List[dict] = []


@mcp_router.post("/servers/add")
async def add_mcp_server(request: AddServerRequest):
    """
    Add and connect to an MCP server.
    
    This endpoint attempts to connect to the specified MCP server,
    discover its available tools, and make them available to the agent.
    """
    try:
        server = await mcp_manager.add_server(
            name=request.name,
            url=request.url,
            description=request.description or ""
        )
        
        return JSONResponse(content={
            "success": server.connected,
            "server": {
                "name": server.name,
                "url": server.url,
                "description": server.description,
                "connected": server.connected,
                "error": server.error,
                "tools": [
                    {"name": t["name"], "description": t.get("description", "")}
                    for t in server.tools
                ]
            },
            "message": f"Connected to server '{server.name}' with {len(server.tools)} tools" 
                      if server.connected 
                      else f"Failed to connect: {server.error}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mcp_router.post("/servers/remove")
async def remove_mcp_server(request: RemoveServerRequest):
    """
    Remove an MCP server connection.
    
    This endpoint disconnects from the specified MCP server
    and removes its tools from the agent.
    """
    try:
        success = await mcp_manager.remove_server(request.name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Server '{request.name}' not found")
        
        return JSONResponse(content={
            "success": True,
            "message": f"Server '{request.name}' disconnected successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mcp_router.get("/servers")
async def list_mcp_servers():
    """
    List all connected MCP servers.
    
    Returns information about all MCP servers including their
    connection status and available tools.
    """
    try:
        servers = mcp_manager.get_servers()
        
        return JSONResponse(content={
            "success": True,
            "servers": servers,
            "total_servers": len(servers),
            "total_tools": sum(len(s["tools"]) for s in servers)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mcp_router.get("/tools")
async def list_mcp_tools():
    """
    List all available tools from connected MCP servers.
    
    Returns a list of all tools that have been discovered
    from connected MCP servers.
    """
    try:
        tools = mcp_manager.get_tools()
        
        return JSONResponse(content={
            "success": True,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description
                }
                for tool in tools
            ],
            "total": len(tools)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mcp_router.post("/test-connection")
async def test_mcp_connection(request: AddServerRequest):
    """
    Test connection to an MCP server without adding it.
    
    This endpoint attempts to connect and discover tools
    but does not persist the connection.
    """
    try:
        # Temporarily add the server
        server = await mcp_manager.add_server(
            name=f"_test_{request.name}",
            url=request.url,
            description=request.description or ""
        )
        
        result = {
            "success": server.connected,
            "url": request.url,
            "connected": server.connected,
            "error": server.error,
            "tools_discovered": len(server.tools),
            "tools": [
                {"name": t["name"], "description": t.get("description", "")}
                for t in server.tools
            ]
        }
        
        # Remove the test connection
        await mcp_manager.remove_server(f"_test_{request.name}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        try:
            await mcp_manager.remove_server(f"_test_{request.name}")
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))
