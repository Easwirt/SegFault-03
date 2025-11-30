import os
import asyncio
import base64
import httpx
import re
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated, List, Dict, TypedDict, Union, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from app.service.mcp_client import mcp_manager
from app.service.semantic_cache import semantic_cache


load_dotenv()

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment. Please check .env file.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Keywords that indicate complex tasks requiring multi-step agent
COMPLEX_TASK_INDICATORS = [
    "and then", "after that", "followed by", "next", "finally",
    "multiple", "several", "batch", "all of", "each of",
    "compare", "analyze and", "research and", "find and",
    "create report", "summarize", "combine", "merge",
    "step by step", "workflow", "pipeline", "process",
]

def is_complex_task(prompt: str) -> bool:
    """Determine if a task requires multi-step agent processing."""
    prompt_lower = prompt.lower()
    
    # Check for explicit complexity indicators
    for indicator in COMPLEX_TASK_INDICATORS:
        if indicator in prompt_lower:
            return True
    
    # Check for multiple action verbs (indicates multiple steps)
    action_verbs = ["search", "find", "generate", "create", "analyze", "read", "write", "process", "resize", "crop", "blur"]
    verb_count = sum(1 for verb in action_verbs if verb in prompt_lower)
    if verb_count >= 2:
        return True
    
    # Check sentence count (multiple sentences often = multiple tasks)
    sentences = [s.strip() for s in re.split(r'[.!?]', prompt) if s.strip()]
    if len(sentences) >= 3:
        return True
    
    return False

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


@tool
def generate_image(prompt: str) -> str:
    """Generate an image using AI (DALL-E). Arguments: prompt."""
    try:
        client = OpenAI()
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        
        import re, os, httpx
        
        safe_name = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:30].strip().replace(' ', '_')
        filename = f"{safe_name or 'generated'}_image.png"
        
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(script_dir, "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with httpx.Client(timeout=60.0) as http_client:
            img_response = http_client.get(image_url)
            img_response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
        
        return f"Successfully generated and saved image to {filepath}"

    except Exception as e:
        return f"Error generating image: {e}"

@tool
def process_image(args: str) -> str:
    """Process an image with various operations. 
    Arguments: a string with format 'input_file|operation|output_file|params'
    - input_file: path to input image
    - operation: one of 'resize', 'crop', 'rotate', 'grayscale', 'blur', 'thumbnail'
    - output_file: path for output image (optional, defaults to input_file with _processed suffix)
    - params: operation-specific parameters (optional, e.g., '800x600' for resize, '90' for rotate)
    Example: 'photo.png|grayscale' or 'photo.png|resize|resized.png|800x600'
    """
    print(f"    [Tool] Processing image: {args}")
    try:
        parts = [p.strip() for p in args.split('|')]
        if len(parts) < 2:
            return "Error: Need at least input_file and operation separated by |"
        
        input_file = parts[0]
        operation = parts[1].lower()
        output_file = parts[2] if len(parts) > 2 else f"{input_file.rsplit('.', 1)[0]}_processed.{input_file.rsplit('.', 1)[1]}"
        params = parts[3] if len(parts) > 3 else ""
        
        img = Image.open(input_file)
        
        if operation == "resize":
            # params should be 'widthxheight' like '800x600'
            if params:
                width, height = map(int, params.lower().split('x'))
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            else:
                return "Error: resize requires params in format 'widthxheight' (e.g., '800x600')"
                
        elif operation == "crop":
            # params should be 'left,top,right,bottom'
            if params:
                left, top, right, bottom = map(int, params.split(','))
                img = img.crop((left, top, right, bottom))
            else:
                return "Error: crop requires params in format 'left,top,right,bottom'"
                
        elif operation == "rotate":
            # params should be degrees (e.g., '90', '180', '-45')
            if params:
                angle = float(params)
                img = img.rotate(angle, expand=True)
            else:
                return "Error: rotate requires params as degrees (e.g., '90')"
                
        elif operation == "grayscale":
            img = img.convert('L')
            
        elif operation == "blur":
            from PIL import ImageFilter
            # params is optional blur radius (default 2)
            radius = int(params) if params else 2
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        elif operation == "thumbnail":
            # params should be 'widthxheight' max size
            if params:
                width, height = map(int, params.lower().split('x'))
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
            else:
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
        elif operation == "convert":
            # params should be format like 'PNG', 'JPEG', 'WEBP'
            if not params:
                params = output_file.split('.')[-1].upper()
            # Convert mode if needed for JPEG
            if params.upper() in ['JPEG', 'JPG'] and img.mode in ['RGBA', 'P']:
                img = img.convert('RGB')
                
        else:
            return f"Error: Unknown operation '{operation}'. Valid operations: resize, crop, rotate, grayscale, blur, thumbnail, convert"
        
        # Determine format from output file extension
        output_format = output_file.split('.')[-1].upper()
        if output_format == 'JPG':
            output_format = 'JPEG'
            
        img.save(output_file, format=output_format if output_format in ['PNG', 'JPEG', 'WEBP', 'GIF', 'BMP'] else None)
        return f"Successfully processed image and saved to {output_file}"
        
    except FileNotFoundError:
        return f"Error: Input file '{input_file}' not found"
    except Exception as e:
        return f"Error processing image: {e}"


@tool
def analyze_image(image_file: str) -> str:
    """Analyze an image and describe its contents using AI vision. Arguments: image_file (path to the image)."""
    print(f"    [Tool] Analyzing image: {image_file}")
    try:
        # Read and encode the image
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine mime type
        ext = image_file.lower().split('.')[-1]
        mime_types = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. What do you see? Include colors, objects, people, text, and any other notable elements."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except FileNotFoundError:
        return f"Error: Image file '{image_file}' not found"
    except Exception as e:
        return f"Error analyzing image: {e}"


# Base tools (always available)
base_tools = [search_web, read_file, generate_image, process_image, analyze_image]

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
    
    # Get current tools for context
    current_tools = get_all_tools()
    tools_list = ", ".join([t.name for t in current_tools])
    
    # Ask LLM to generate a plan
    system_prompt = f"""You are a planning agent. Break down the user request into a list of concise, actionable tasks.
You have these tools available: {tools_list}

CRITICAL GUIDELINES:
- Keep plans SHORT and DIRECT (1-3 steps for simple tasks)
- NEVER ask for clarification - just execute with reasonable defaults
- For image generation: Use 'generate_image' directly with a creative, detailed prompt based on what the user asked
  - If user says "generate a dog", create a prompt like "a friendly golden retriever sitting in a sunny park"
  - If user says "generate a cat", create a prompt like "a fluffy orange tabby cat lounging on a windowsill"
- For image analysis: Use 'analyze_image' directly
- For web searches: Use 'search_web'
- Don't create unnecessary steps like "access tool", "adjust settings", or "ask for details"
- Each task should map to ONE tool call
- BE PROACTIVE - make reasonable creative decisions rather than asking the user

Return ONLY the list of tasks, one per line. Do not number them."""
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

CRITICAL RULES:
- NEVER ask for clarification or more details - just execute!
- If generating an image, create a detailed, creative prompt from the task description
- If the task mentions "dog" -> use generate_image with a detailed dog description
- If the task mentions "cat" -> use generate_image with a detailed cat description
- Be creative and fill in reasonable details yourself

Tool usage:
- search_web: for finding information online
- read_file: for reading files
- generate_image: for creating images (provide detailed visual description)
- analyze_image: for analyzing existing images
- process_image: for modifying images (format: input|operation|output|params)

Return your response in the format: TOOL_NAME: ARGUMENT
Example: search_web: python 3.13 features
Example: generate_image: a happy golden retriever puppy playing in a grassy meadow with flowers, sunny day, photorealistic
Example: ANSWER: Task completed successfully."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Execute this task: {current_task}")
    ]
    
    try:
        # Add timeout to prevent hanging
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30.0)
        content = response.content.strip()
    except asyncio.TimeoutError:
        content = "ANSWER: Task timed out, skipping."
    except Exception as e:
        content = f"ANSWER: Error during task execution: {e}"
    
    task_result = ""
    
    # Handle empty response
    if not content:
        content = "ANSWER: No action needed for this step."
    
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
    
    # Check if any files were generated
    file_results = []
    for task, result in results.items():
        if "Successfully generated" in result or "Successfully wrote" in result or "Successfully processed" in result:
            file_results.append(result)
    
    # If files were created, include them in the response
    if file_results:
        system_prompt = """You are a helpful assistant. Summarize what was accomplished.
If files were created, mention them clearly with their filenames.
Keep the response concise and informative."""
    else:
        system_prompt = "You are a helpful assistant. Synthesize the results of the executed tasks into a final answer for the user."
    
    messages = [
        SystemMessage(content=system_prompt),
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


# --- Simple Direct Generation (for simple tasks) ---

async def simple_generate(prompt: str) -> AsyncGenerator[dict, None]:
    """
    Simple direct generation for non-complex tasks.
    Uses a single LLM call with tool use instead of multi-step agent pipeline.
    """
    print(f"\n=== SIMPLE GENERATION MODE ===")
    print(f"Prompt: {prompt}")
    
    yield {"type": "mode", "mode": "simple"}
    
    # Get current tools
    current_tools = get_all_tools()
    current_tools_map = get_tools_map()
    tools_description = "\n".join([f"- {t.name}: {t.description}" for t in current_tools])
    
    # Determine if this needs a tool or is just a question
    system_prompt = f"""You are a helpful assistant. Answer the user's question or perform the requested task.

You have these tools available (use ONLY if needed):
{tools_description}

RULES:
1. For simple questions or conversations, just respond directly
2. For tasks requiring tools, respond with: TOOL: tool_name: argument
3. For image generation: TOOL: generate_image: detailed visual description
4. For web searches: TOOL: search_web: search query
5. For image analysis: TOOL: analyze_image: filepath
6. For image processing: TOOL: process_image: input|operation|output|params
7. Be creative and fill in reasonable details yourself
8. NEVER ask for clarification - make reasonable assumptions

Examples:
- "What is the capital of France?" -> Just answer: "Paris is the capital of France."
- "Generate a cat" -> TOOL: generate_image: a fluffy orange tabby cat lounging on a cozy windowsill with warm sunlight
- "Search for Python tutorials" -> TOOL: search_web: Python programming tutorials for beginners"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30.0)
        content = response.content.strip()
        print(f"LLM Response: {content[:200]}...")
    except asyncio.TimeoutError:
        yield {"type": "final_answer", "content": "Request timed out. Please try again.", "files": []}
        yield {"type": "done"}
        return
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return
    
    # Check if tool use is requested
    if content.startswith("TOOL:"):
        # Parse: TOOL: tool_name: argument
        parts = content[5:].strip().split(":", 1)
        if len(parts) >= 2:
            tool_name = parts[0].strip()
            tool_arg = parts[1].strip()
            
            yield {"type": "task_start", "task": f"Using {tool_name}", "task_index": 0}
            
            if tool_name in current_tools_map:
                tool_func = current_tools_map[tool_name]
                try:
                    # Execute the tool
                    if hasattr(tool_func, 'coroutine') and tool_func.coroutine:
                        result = await tool_func.coroutine(**{'arguments': tool_arg} if tool_name.startswith('mcp_') else {})
                        if not result or result == '{}':
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                executor,
                                partial(run_tool_sync, tool_func, tool_arg)
                            )
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            executor,
                            partial(run_tool_sync, tool_func, tool_arg)
                        )
                    
                    print(f"Tool result: {result[:200] if result else 'None'}...")
                    
                    yield {
                        "type": "task_complete",
                        "task": f"Using {tool_name}",
                        "task_index": 0,
                        "result": result
                    }
                    
                    # Generate final answer based on result
                    final_messages = [
                        SystemMessage(content="Summarize the result concisely for the user. If a file was created, mention the filename."),
                        HumanMessage(content=f"Task: {prompt}\nResult: {result}")
                    ]
                    final_response = await llm.ainvoke(final_messages)
                    
                    yield {
                        "type": "final_answer",
                        "content": final_response.content,
                        "files": [],
                        "tool_result": result
                    }
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield {"type": "error", "message": f"Tool error: {str(e)}"}
                    return
            else:
                yield {"type": "final_answer", "content": f"Unknown tool: {tool_name}", "files": []}
        else:
            yield {"type": "final_answer", "content": content, "files": []}
    else:
        # Direct response without tool use
        yield {"type": "final_answer", "content": content, "files": []}
    
    yield {"type": "done"}


async def agent_generate(prompt: str) -> AsyncGenerator[dict, None]:
    """
    Full agent pipeline for complex multi-step tasks.
    Uses planner -> executor -> responder workflow.
    """
    print(f"\n=== AGENT GENERATION MODE ===")
    print(f"Prompt: {prompt}")
    
    yield {"type": "mode", "mode": "agent"}
    
    try:
        async for chunk in app.astream({"request": prompt}, stream_mode="updates"):
            for node_name, output in chunk.items():
                if node_name == "planner":
                    yield {"type": "node_start", "node": "planner"}
                    if "plan" in output:
                        yield {"type": "plan", "plan": output["plan"]}
                
                elif node_name == "executor":
                    yield {"type": "node_start", "node": "executor"}
                    if "results" in output:
                        results = output["results"]
                        if results:
                            latest_task = list(results.keys())[-1]
                            latest_result = results[latest_task]
                            task_index = output.get("current_task_index", 1) - 1
                            
                            yield {
                                "type": "task_complete",
                                "task": latest_task,
                                "task_index": task_index,
                                "result": latest_result[:500]
                            }
                
                elif node_name == "responder":
                    yield {"type": "node_start", "node": "responder"}
                    if "final_answer" in output:
                        yield {
                            "type": "final_answer",
                            "content": output["final_answer"],
                            "files": []
                        }
        
        yield {"type": "done"}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}


async def generate_response(prompt: str, mode: str = "auto", use_cache: bool = True) -> AsyncGenerator[dict, None]:
    """
    Main entry point for generation.
    
    Args:
        prompt: User's request
        mode: "auto" (detect complexity), "simple" (direct), or "agent" (full pipeline)
        use_cache: Whether to use semantic cache for responses
    
    Yields:
        Streaming response events
    """
    # Check semantic cache first (only for non-tool-requiring requests in auto/simple mode)
    if use_cache and mode != "agent":
        # Skip cache for requests that clearly need fresh data or tool execution
        skip_cache_keywords = ["generate image", "create image", "search for", "search the", 
                               "look up", "find latest", "current", "today", "now"]
        should_skip_cache = any(kw in prompt.lower() for kw in skip_cache_keywords)
        
        if not should_skip_cache:
            cached = semantic_cache.get(prompt, mode)
            if cached:
                print(f"\n[Generate] Cache HIT - similarity: {cached['similarity']:.4f}")
                yield {"type": "mode", "mode": "cached"}
                yield {"type": "cache_hit", "similarity": cached['similarity'], "cached_request": cached['cached_request']}
                yield {"type": "final_answer", "content": cached['response'], "files": [], "cached": True}
                yield {"type": "done"}
                return
    
    # Determine which mode to use
    if mode == "agent":
        use_agent = True
    elif mode == "simple":
        use_agent = False
    else:  # auto mode
        use_agent = is_complex_task(prompt)
    
    print(f"\n[Generate] Mode requested: {mode}, Using agent: {use_agent}")
    
    # Collect response for caching
    final_response = None
    
    if use_agent:
        async for event in agent_generate(prompt):
            if event.get("type") == "final_answer":
                final_response = event.get("content", "")
            yield event
    else:
        async for event in simple_generate(prompt):
            if event.get("type") == "final_answer":
                final_response = event.get("content", "")
            yield event
    
    # Cache the response (only for text responses, not file generations)
    if use_cache and final_response and not any(kw in final_response.lower() for kw in ["successfully generated", "successfully saved", "successfully wrote"]):
        try:
            semantic_cache.set(prompt, final_response, mode)
        except Exception as e:
            print(f"[Generate] Failed to cache response: {e}")
