"""
HTTP-based MCP Server for testing MCP client functionality.
This server provides basic tools via HTTP endpoints that can be used to verify the MCP integration works.

Run with: python http_server.py
Then connect using URL: http://localhost:8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime
import random
import uvicorn

app = FastAPI(
    title="Simple Test MCP Server",
    description="A simple MCP server for testing MCP client functionality",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tool definitions
TOOLS = [
    {
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
    {
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
    {
        "name": "get_current_time",
        "description": "Returns the current date and time.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
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
    {
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
]


class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    content: List[Dict[str, Any]]
    isError: bool = False


# Endpoints for tool discovery
@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "name": "simple-test-mcp-server",
        "version": "1.0.0",
        "description": "A simple MCP server for testing",
        "endpoints": {
            "tools": "/tools",
            "call": "/tools/call"
        }
    }


@app.get("/tools")
async def list_tools_get():
    """List available tools (GET method)"""
    return {"tools": TOOLS}


@app.post("/tools")
async def list_tools_post():
    """List available tools (POST method)"""
    return {"tools": TOOLS}


@app.get("/tools/list")
async def list_tools_mcp_get():
    """List available tools - MCP style endpoint (GET)"""
    return {"tools": TOOLS}


@app.post("/tools/list")
async def list_tools_mcp_post():
    """List available tools - MCP style endpoint (POST)"""
    return {"tools": TOOLS}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a tool with the given arguments"""
    tool_name = request.name
    arguments = request.arguments
    
    if tool_name == "echo":
        message = arguments.get("message", "")
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Echo: {message}"}]
        )
    
    elif tool_name == "add_numbers":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return ToolCallResponse(
            content=[{"type": "text", "text": f"The sum of {a} and {b} is {result}"}]
        )
    
    elif tool_name == "get_current_time":
        now = datetime.now()
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"}]
        )
    
    elif tool_name == "reverse_string":
        text = arguments.get("text", "")
        reversed_text = text[::-1]
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Reversed: {reversed_text}"}]
        )
    
    elif tool_name == "generate_random_number":
        min_val = arguments.get("min", 1)
        max_val = arguments.get("max", 100)
        number = random.randint(min_val, max_val)
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Random number between {min_val} and {max_val}: {number}"}]
        )
    
    else:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    print("=" * 50)
    print("Simple Test MCP Server")
    print("=" * 50)
    print("\nStarting HTTP MCP Server on http://localhost:8001")
    print("\nTo connect this server to your agent, use URL:")
    print("  http://localhost:8001")
    print("\nAvailable tools:")
    for tool in TOOLS:
        print(f"  - {tool['name']}: {tool['description']}")
    print("\n" + "=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
