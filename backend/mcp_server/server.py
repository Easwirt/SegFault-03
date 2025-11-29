"""
Simple MCP Server for testing MCP client functionality.
This server provides basic tools that can be used to verify the MCP integration works.
"""

import json
import sys
import asyncio
from typing import Any


class SimpleMCPServer:
    """A simple MCP server that implements the JSON-RPC protocol over stdio."""
    
    def __init__(self):
        self.tools = {
            "echo": {
                "name": "echo",
                "description": "Echoes back the input message. Useful for testing connectivity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo back"
                        }
                    },
                    "required": ["message"]
                }
            },
            "add_numbers": {
                "name": "add_numbers",
                "description": "Adds two numbers together and returns the result.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            },
            "get_current_time": {
                "name": "get_current_time",
                "description": "Returns the current date and time.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            "reverse_string": {
                "name": "reverse_string",
                "description": "Reverses a given string.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The string to reverse"
                        }
                    },
                    "required": ["text"]
                }
            },
            "generate_random_number": {
                "name": "generate_random_number",
                "description": "Generates a random number between min and max (inclusive).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "Minimum value (default: 1)",
                            "default": 1
                        },
                        "max": {
                            "type": "integer",
                            "description": "Maximum value (default: 100)",
                            "default": 100
                        }
                    },
                    "required": []
                }
            }
        }
    
    def handle_initialize(self, params: dict) -> dict:
        """Handle the initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "simple-test-mcp-server",
                "version": "1.0.0"
            }
        }
    
    def handle_list_tools(self) -> dict:
        """Handle tools/list request."""
        return {
            "tools": list(self.tools.values())
        }
    
    def handle_call_tool(self, params: dict) -> dict:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "echo":
            message = arguments.get("message", "")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Echo: {message}"
                    }
                ]
            }
        
        elif tool_name == "add_numbers":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result = a + b
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"The sum of {a} and {b} is {result}"
                    }
                ]
            }
        
        elif tool_name == "get_current_time":
            from datetime import datetime
            now = datetime.now()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        
        elif tool_name == "reverse_string":
            text = arguments.get("text", "")
            reversed_text = text[::-1]
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Reversed: {reversed_text}"
                    }
                ]
            }
        
        elif tool_name == "generate_random_number":
            import random
            min_val = arguments.get("min", 1)
            max_val = arguments.get("max", 100)
            number = random.randint(min_val, max_val)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Random number between {min_val} and {max_val}: {number}"
                    }
                ]
            }
        
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}"
                    }
                ],
                "isError": True
            }
    
    def handle_request(self, request: dict) -> dict:
        """Handle an incoming JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        result = None
        error = None
        
        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "notifications/initialized":
                # This is a notification, no response needed
                return None
            elif method == "tools/list":
                result = self.handle_list_tools()
            elif method == "tools/call":
                result = self.handle_call_tool(params)
            else:
                error = {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
        except Exception as e:
            error = {
                "code": -32603,
                "message": str(e)
            }
        
        if request_id is None:
            # It's a notification, don't send response
            return None
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        
        if error:
            response["error"] = error
        else:
            response["result"] = result
        
        return response
    
    async def run(self):
        """Run the MCP server, reading from stdin and writing to stdout."""
        while True:
            try:
                # Read a line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse the JSON-RPC request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    continue
                
                # Handle the request
                response = self.handle_request(request)
                
                # Send the response (if not a notification)
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except Exception as e:
                # Log error to stderr (won't interfere with JSON-RPC on stdout)
                print(f"Server error: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    server = SimpleMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    print("start")
    main()
