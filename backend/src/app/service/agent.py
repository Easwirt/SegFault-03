import os
import operator
from typing import Annotated, List, Dict, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from fpdf import FPDF
from ddgs import DDGS
from dotenv import load_dotenv

# --- 1. Setup & Configuration ---
# Load environment variables from .env file
load_dotenv()

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment. Please check .env file.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
def write_file(filename: str, content: str) -> str:
    """Useful for writing a file to disk. Arguments: filename, content."""
    print(f"    [Tool] Writing file: {filename}")
    try:
        if filename.lower().endswith('.pdf'):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Handle text content line by line
            # Note: Standard FPDF fonts only support Latin-1. 
            # We replace unsupported characters to prevent errors.
            for line in content.split('\n'):
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, txt=safe_line)
                
            pdf.output(filename)
            return f"Successfully wrote PDF to {filename}"
        else:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def read_file(filename: str) -> str:
    """Useful for reading a file from disk. Arguments: filename."""
    print(f"    [Tool] Reading file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def convert_to_pdf(source_file: str, dest_file: str) -> str:
    """Useful for converting a text file to a PDF file. Arguments: source_file, dest_file."""
    print(f"    [Tool] Converting {source_file} to {dest_file}")
    try:
        if not os.path.exists(source_file):
            return f"Error: Source file {source_file} does not exist."
            
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Handle text content line by line
        for line in content.split('\n'):
            # Replace unsupported characters to prevent errors (basic FPDF limitation)
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=safe_line)
            
        pdf.output(dest_file)
        return f"Successfully converted {source_file} to {dest_file}"
    except Exception as e:
        return f"Error converting file: {e}"

tools = [search_web, write_file, read_file, convert_to_pdf]
tools_map = {t.name: t for t in tools}

# --- 3. Define State ---
class AgentState(TypedDict):
    request: str
    plan: List[str]
    current_task_index: int
    results: Dict[str, str]
    final_answer: str

# --- 4. Define Nodes ---

def planner_node(state: AgentState):
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
    response = llm.invoke(messages)
    
    # Parse the plan
    plan = [line.strip("- ").strip() for line in response.content.split("\n") if line.strip()]
    
    print(f"Generated Plan ({len(plan)} steps):")
    for i, task in enumerate(plan):
        print(f"  {i+1}. {task}")
        
    return {"plan": plan, "current_task_index": 0, "results": {}}

def executor_node(state: AgentState):
    print("\n--- 2. EXECUTING ---")
    plan = state["plan"]
    index = state["current_task_index"]
    current_task = plan[index]
    results = state["results"]
    
    print(f"Current Task [{index+1}/{len(plan)}]: {current_task}")
    
    # Context for the agent to make decisions
    context = "\n".join([f"Task: {k}\nResult: {v}" for k, v in results.items()])
    
    # Ask LLM which tool to use
    tools_description = "\n".join([f"{t.name}: {t.description}" for t in tools])
    system_prompt = f"""You are an execution agent. You have the following tools:
{tools_description}

Your goal is to execute the current task: "{current_task}"
You have access to the results of previous tasks:
{context}

Based on the task, decide which tool to use.
- If you need to search, use 'search_web'.
- If you need to read a file, use 'read_file'.
- If you need to write a file (like a report or code), use 'write_file'.
- If you need to convert a text file to PDF, use 'convert_to_pdf'.
- If you can answer directly without a tool (e.g. summarizing), return 'ANSWER: <your content>'.

Return your response in the format: TOOL_NAME: ARGUMENT
Example: search_web: python 3.13 features
Example: read_file: data.txt
Example: write_file: report.txt, This is the content...
Example: convert_to_pdf: report.txt, report.pdf
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Execute this task: {current_task}")
    ]
    response = llm.invoke(messages)
    content = response.content.strip()
    
    task_result = ""
    
    # Parse tool call
    if ":" in content:
        # Split only on the first colon
        action, arg = content.split(":", 1)
        action = action.strip()
        arg = arg.strip()
        
        if action in tools_map:
            tool_func = tools_map[action]
            try:
                # Handle multi-argument tools
                if action == "write_file" and "," in arg:
                    fname, fcontent = arg.split(",", 1)
                    task_result = tool_func.invoke({"filename": fname.strip(), "content": fcontent.strip()})
                elif action == "convert_to_pdf" and "," in arg:
                    src, dst = arg.split(",", 1)
                    task_result = tool_func.invoke({"source_file": src.strip(), "dest_file": dst.strip()})
                else:
                    task_result = tool_func.invoke(arg)
            except Exception as e:
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


def responder_node(state: AgentState):
    print("\n--- 3. FINAL ANSWER ---")
    request = state["request"]
    results = state["results"]
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Synthesize the results of the executed tasks into a final answer for the user."),
        HumanMessage(content=f"Original Request: {request}\n\nTask Results: {results}")
    ]
    response = llm.invoke(messages)
    
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

# --- 6. Run ---
if __name__ == "__main__":
    print("=== AI Agent Started (Type 'exit' to quit) ===")
    
    while True:
        try:
            user_input = input("\nEnter your request: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            print(f"Processing Request: {user_input}")
            
            inputs = {"request": user_input}
            final_state = app.invoke(inputs)
            
            print("\n=== FINAL ANSWER ===")
            print(final_state["final_answer"])
            print("====================")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
