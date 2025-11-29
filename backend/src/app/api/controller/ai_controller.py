import json
import os
import re
import asyncio
import base64
import mimetypes
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from app.service.agent import app
from app.service.agent import get_all_tools

ai_router = APIRouter()

# Base URL for file endpoints
BASE_URL = "http://localhost:8000"


def extract_filename_from_result(result: str) -> str | None:
    """Extract filename from tool result like 'Successfully wrote to report.txt'"""
    patterns = [
        r"Successfully wrote to (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, result)
        if match:
            return match.group(1).strip()
    return None


def get_file_type(filename: str) -> str:
    """Determine file type from filename"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.txt', '.md', '.json', '.csv', '.xml', '.html', '.css', '.js', '.py']:
        return 'text'
    elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']:
        return 'image'
    else:
        return 'binary'


def get_file_preview(filepath: str, file_type: str) -> dict | None:
    """Generate preview data for a file"""
    try:
        if file_type == 'text':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # First 1000 chars for preview
                truncated = len(content) == 1000
            return {
                "type": "text",
                "content": content,
                "truncated": truncated
            }
        elif file_type == 'image':
            mime_type = mimetypes.guess_type(filepath)[0] or 'image/png'
            with open(filepath, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return {
                "type": "image",
                "dataUrl": f"data:{mime_type};base64,{data}"
            }
        return None
    except Exception as e:
        return {"type": "error", "message": str(e)}


def build_file_status(filename: str, filepath: str, result: str) -> dict:
    """Build a comprehensive file status response"""
    file_exists = os.path.exists(filepath)
    file_type = get_file_type(filename)
    
    status = {
        "created": file_exists,
        "filename": filename,
        "filepath": filepath,
        "fileType": file_type,
        "message": result
    }
    
    if file_exists:
        # Get file size
        status["fileSize"] = os.path.getsize(filepath)
        # Add download URL
        status["downloadUrl"] = f"{BASE_URL}/download?filepath={filepath}"
        # Add preview URL
        status["previewUrl"] = f"{BASE_URL}/preview?filepath={filepath}"
        # Generate inline preview if possible
        preview = get_file_preview(filepath, file_type)
        if preview:
            status["preview"] = preview
    
    return status


async def run_agent_and_collect(prompt: str, queue: asyncio.Queue):
    """Run agent and put results in queue"""
    generated_files = []
    
    try:
        async for chunk in app.astream({"request": prompt}, stream_mode="updates"):
            for node_name, output in chunk.items():
                if node_name == "planner":
                    await queue.put({"type": "node_start", "node": "planner"})
                    if "plan" in output:
                        await queue.put({"type": "plan", "plan": output["plan"]})
                
                elif node_name == "executor":
                    await queue.put({"type": "node_start", "node": "executor"})
                    if "results" in output:
                        results = output["results"]
                        if results:
                            latest_task = list(results.keys())[-1]
                            latest_result = results[latest_task]
                            
                            # Calculate task index (current_task_index is the NEXT task, so subtract 1)
                            task_index = output.get("current_task_index", 1) - 1
                            
                            filename = extract_filename_from_result(latest_result)
                            if filename:
                                # Build comprehensive file status
                                file_status = build_file_status(
                                    os.path.basename(filename),
                                    filename,
                                    latest_result
                                )
                                
                                if file_status["created"]:
                                    generated_files.append(filename)
                                
                                await queue.put({
                                    "type": "file_status",
                                    "task": latest_task,
                                    "task_index": task_index,
                                    "status": file_status
                                })
                            else:
                                await queue.put({
                                    "type": "task_complete",
                                    "task": latest_task,
                                    "task_index": task_index,
                                    "result": latest_result[:500]
                                })
                
                elif node_name == "responder":
                    await queue.put({"type": "node_start", "node": "responder"})
                    if "final_answer" in output:
                        # Build file status for all generated files
                        files_with_status = []
                        for f in generated_files:
                            file_status = build_file_status(
                                os.path.basename(f),
                                f,
                                "File generated successfully"
                            )
                            files_with_status.append(file_status)
                        
                        await queue.put({
                            "type": "final_answer",
                            "content": output["final_answer"],
                            "files": files_with_status
                        })
        
        await queue.put({"type": "done"})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        await queue.put({"type": "error", "message": str(e)})
    
    await queue.put(None)  # Signal end


async def stream_generator(prompt: str):
    """Generator that streams from queue as SSE."""
    queue = asyncio.Queue()
    
    # Start agent task
    task = asyncio.create_task(run_agent_and_collect(prompt, queue))
    
    try:
        while True:
            # Wait for data with timeout to send heartbeats
            try:
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Send heartbeat more frequently
                yield ": heartbeat\n\n"
                continue
            
            if data is None:
                break
            
            yield f"data: {json.dumps(data)}\n\n"
            # Small delay to ensure data is flushed
            await asyncio.sleep(0.01)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        if not task.done():
            task.cancel()


@ai_router.post("/generate")
async def generate(request: Request, prompt: str):
    print(get_all_tools())
    return StreamingResponse(
        stream_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
        }
    )


@ai_router.get("/download")
async def download_file(filepath: str):
    """Download a generated file."""
    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found", "created": False}
        )
    
    filename = os.path.basename(filepath)
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


@ai_router.get("/preview")
async def preview_file(filepath: str):
    """Preview a generated file (for images and text)."""
    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found", "created": False}
        )
    
    filename = os.path.basename(filepath)
    file_type = get_file_type(filename)
    
    if file_type == 'image':
        mime_type = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=mime_type
        )
    elif file_type == 'text':
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="text/plain; charset=utf-8"
        )
    else:
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream"
        )


@ai_router.get("/file-status")
async def get_file_status(filepath: str):
    """Check file creation status and get file info."""
    filename = os.path.basename(filepath)
    file_status = build_file_status(filename, filepath, "Status check")
    return JSONResponse(content=file_status)