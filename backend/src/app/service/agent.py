import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated, List, Dict, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from dotenv import load_dotenv
from app.service.mcp_client import mcp_manager


load_dotenv()

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment. Please check .env file.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Thread pool for running sync tools
executor = ThreadPoolExecutor(max_workers=4)


# Helper function to run tool synchronously
def run_tool_sync(tool_func, tool_input):
    """Run a tool synchronously - to be called in thread pool"""
    return tool_func.invoke(tool_input)

# --- 2. Define Tools ---
@tool
def search_web(query: str) -> str:
    """Useful for searching information on the web. Returns a summary of results."""
    print(f"    [Tool] Searching web for: {query}")
    try:
        # Use duckduckgo_search directly
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        # Format results
        formatted_results = ""
        for r in results:
            formatted_results += f"- {r['title']}: {r['body']}\n"
            
        return formatted_results
    except Exception as e:
        return f"Error searching web: {e}"


@tool
def read_file(filename: str) -> str:
    """Useful for reading a file from disk. Arguments: filename."""
    print(f"    [Tool] Reading file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# Base tools (always available)
base_tools = [search_web, read_file]

def get_all_tools():
    """Get all tools including MCP tools"""
    all_tools = list(base_tools)
    mcp_tools = mcp_manager.get_tools()
    all_tools.extend(mcp_tools)
    return all_tools

def get_tools_map():
    """Get tools map including MCP tools"""
    all_tools = get_all_tools()
    return {t.name: t for t in all_tools}

# Initial tools (will be updated when MCP servers are added)
tools = base_tools
tools_map = {t.name: t for t in tools}

# --- 3. Define State ---
class AgentState(TypedDict):
    request: str
    plan: List[str]
    current_task_index: int
    results: Dict[str, str]
    final_answer: str

# --- 4. Define Nodes ---

async def planner_node(state: AgentState):
    print("\n--- 1. PLANNING ---")
    request = state["request"]
    
    # Ask LLM to generate a plan
    system_prompt = (
        "You are a planning agent. Break down the user request into a list of concise, actionable tasks. "
        "The tasks should be steps that an AI agent can perform using web search or file writing. "
        "Return ONLY the list of tasks, one per line. Do not number them."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request)
    ]
    response = await llm.ainvoke(messages)
    
    # Parse the plan
    plan = [line.strip("- ").strip() for line in response.content.split("\n") if line.strip()]
    
    print(f"Generated Plan ({len(plan)} steps):")
    for i, task in enumerate(plan):
        print(f"  {i+1}. {task}")
        
    return {"plan": plan, "current_task_index": 0, "results": {}}

async def executor_node(state: AgentState):
    print("\n--- 2. EXECUTING ---")
    plan = state["plan"]
    index = state["current_task_index"]
    current_task = plan[index]
    results = state["results"]
    
    print(f"Current Task [{index+1}/{len(plan)}]: {current_task}")
    
    # Context for the agent to make decisions
    context = "\n".join([f"Task: {k}\nResult: {v}" for k, v in results.items()])
    
    # Get current tools including any MCP tools
    current_tools = get_all_tools()
    current_tools_map = get_tools_map()
    
    # Ask LLM which tool to use
    tools_description = "\n".join([f"{t.name}: {t.description}" for t in current_tools])
    system_prompt = f"""You are an execution agent. You have the following tools:
{tools_description}

Your goal is to execute the current task: "{current_task}"
You have access to the results of previous tasks:
{context}

Based on the task, decide which tool to use.
- If you need to search, use 'search_web'.
- If you need to read a file, use 'read_file'.
- If you can answer directly without a tool (e.g. summarizing), return 'ANSWER: <your content>'.

Return your response in the format: TOOL_NAME: ARGUMENT
Example: search_web: python 3.13 features
Example: read_file: data.txt
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Execute this task: {current_task}")
    ]
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    task_result = ""
    
    # Parse tool call
    if ":" in content:
        # Split only on the first colon
        action, arg = content.split(":", 1)
        action = action.strip()
        arg = arg.strip()
        
        if action in current_tools_map:
            tool_func = current_tools_map[action]
            try:
                # Check if it's an MCP tool (has coroutine)
                if hasattr(tool_func, 'coroutine') and tool_func.coroutine:
                    # MCP tools are async, call directly
                    task_result = await tool_func.coroutine(**{'arguments': arg} if action.startswith('mcp_') else {})
                    if not task_result or task_result == '{}':
                        # Fallback to sync invocation
                        loop = asyncio.get_running_loop()
                        task_result = await loop.run_in_executor(
                            executor,
                            partial(run_tool_sync, tool_func, arg)
                        )
                else:
                    # Run tool in thread pool to avoid blocking
                    loop = asyncio.get_running_loop()
                    task_result = await loop.run_in_executor(
                        executor,
                        partial(run_tool_sync, tool_func, arg)
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                task_result = f"Error executing tool: {e}"
        elif action == "ANSWER":
            task_result = arg
        else:
            task_result = f"Unknown action: {content}"
    else:
        task_result = content

    print(f"Result: {task_result[:100]}...") # Truncate log
    
    # Update results
    new_results = results.copy()
    new_results[current_task] = task_result
    
    return {"results": new_results, "current_task_index": index + 1}


async def responder_node(state: AgentState):
    print("\n--- 3. FINAL ANSWER ---")
    request = state["request"]
    results = state["results"]
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Synthesize the results of the executed tasks into a final answer for the user."),
        HumanMessage(content=f"Original Request: {request}\n\nTask Results: {results}")
    ]
    response = await llm.ainvoke(messages)
    
    return {"final_answer": response.content}

# --- 5. Define Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("responder", responder_node)

workflow.set_entry_point("planner")

def should_continue(state: AgentState):
    if state["current_task_index"] < len(state["plan"]):
        return "executor"
    return "responder"

workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue)
workflow.add_edge("responder", END)

app = workflow.compile()
