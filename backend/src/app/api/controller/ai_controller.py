import json
import os
import re
import asyncio
import base64
import mimetypes
import time
from urllib.parse import quote
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from app.service.agent import app, generate_response, is_complex_task
from app.service.agent import get_all_tools
from app.service.semantic_cache import semantic_cache

ai_router = APIRouter()

# Base URL for file endpoints
BASE_URL = "http://localhost:8000"


def extract_filename_from_result(result: str) -> str | None:
    """Extract filename from tool result like 'Successfully wrote to report.txt'"""
    patterns = [
        r"Successfully wrote to (.+)",
        r"Successfully generated and saved image to (.+)",
        r"Successfully processed image and saved to (.+)",
        r"saved (?:to|as) (.+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))",
    ]
    for pattern in patterns:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            filepath = match.group(1).strip()
            print(f"    [Extract] Found file path: {filepath}")
            return filepath
    print(f"    [Extract] No file path found in result: {result[:100]}...")
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


# Maximum inline image size (100KB) - larger images use URL preview only
MAX_INLINE_IMAGE_SIZE = 100 * 1024

def get_file_preview(filepath: str, file_type: str, inline: bool = True) -> dict | None:
    """Generate preview data for a file
    
    Args:
        filepath: Path to the file
        file_type: Type of file ('text', 'image', 'binary')
        inline: If True, embed content/base64. If False, only return metadata.
    """
    try:
        print(f"    [Preview] Generating preview for: {filepath} (type: {file_type}, inline: {inline})")
        
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
            file_size = os.path.getsize(filepath)
            mime_type = mimetypes.guess_type(filepath)[0] or 'image/png'
            
            print(f"    [Preview] Image size: {file_size} bytes, MIME: {mime_type}")
            
            # For SSE streaming, skip inline preview for large images
            if not inline or file_size > MAX_INLINE_IMAGE_SIZE:
                print(f"    [Preview] Image too large for inline, using URL preview")
                return {
                    "type": "image",
                    "useUrl": True,
                    "fileSize": file_size,
                    "mimeType": mime_type
                }
            
            with open(filepath, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            print(f"    [Preview] Base64 length: {len(data)} chars")
            return {
                "type": "image",
                "dataUrl": f"data:{mime_type};base64,{data}"
            }
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    [Preview] Error: {e}")
        return {"type": "error", "message": str(e)}


def build_file_status(filename: str, filepath: str, result: str, inline_preview: bool = True, base_url: str | None = None) -> dict:
    """Build a comprehensive file status response
    
    Args:
        filename: Name of the file
        filepath: Full path to the file
        result: Result message from the tool
        inline_preview: Whether to include inline base64 preview (for SSE, set to True but will be limited)
    """
    file_exists = os.path.exists(filepath)
    file_type = get_file_type(filename)
    
    print(f"    [FileStatus] Building status for: {filepath}")
    print(f"    [FileStatus] File exists: {file_exists}, Type: {file_type}")
    
    status = {
        "created": file_exists,
        "filename": filename,
        "filepath": filepath,
        "fileType": file_type,
        "message": result
    }
    
    if file_exists:
        try:
            encoded_path = quote(filepath)
            status["fileSize"] = os.path.getsize(filepath)
            if base_url:
                status["downloadUrl"] = f"{base_url}/download?filepath={encoded_path}"
                status["previewUrl"] = f"{base_url}/preview?filepath={encoded_path}"
            else:
                status["downloadUrl"] = f"/download?filepath={encoded_path}"
                status["previewUrl"] = f"/preview?filepath={encoded_path}"
            preview = get_file_preview(filepath, file_type, inline=inline_preview)
            if preview:
                status["preview"] = preview
                print(f"    [FileStatus] Preview generated for {file_type}: {preview.get('type')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [FileStatus] Error getting file details: {e}")
    
    return status


async def run_agent_and_collect(prompt: str, queue: asyncio.Queue, base_url: str | None, mode: str = "auto"):
    """Run agent/simple generation and put results in queue"""
    generated_files = []
    
    try:
        print(f"[Agent] Starting generation (mode={mode})...")
        
        async for event in generate_response(prompt, mode=mode):
            event_type = event.get("type")
            print(f"[Agent] Event: {event_type}")
            
            if event_type == "mode":
                await queue.put(event)
            
            elif event_type == "cache_hit":
                await queue.put(event)
            
            elif event_type == "plan":
                await queue.put({"type": "node_start", "node": "planner"})
                await queue.put(event)
            
            elif event_type == "node_start":
                await queue.put(event)
            
            elif event_type == "task_start":
                await queue.put(event)
            
            elif event_type == "task_complete":
                task = event.get("task", "")
                result = event.get("result", "")
                task_index = event.get("task_index", 0)
                
                # Check for file generation
                filename = extract_filename_from_result(result)
                if filename:
                    file_status = build_file_status(
                        os.path.basename(filename),
                        filename,
                        result,
                        base_url=base_url
                    )
                    if file_status["created"]:
                        generated_files.append(filename)
                    
                    await queue.put({
                        "type": "file_status",
                        "task": task,
                        "task_index": task_index,
                        "status": file_status
                    })
                else:
                    await queue.put(event)
            
            elif event_type == "final_answer":
                # Check tool_result for files if present
                tool_result = event.get("tool_result", "")
                if tool_result:
                    filename = extract_filename_from_result(tool_result)
                    if filename and filename not in generated_files:
                        file_status = build_file_status(
                            os.path.basename(filename),
                            filename,
                            tool_result,
                            base_url=base_url
                        )
                        if file_status["created"]:
                            generated_files.append(filename)
                
                # Build file statuses for all generated files
                files_with_status = []
                for f in generated_files:
                    file_status = build_file_status(
                        os.path.basename(f),
                        f,
                        "File generated successfully",
                        base_url=base_url
                    )
                    files_with_status.append(file_status)
                
                await queue.put({
                    "type": "final_answer",
                    "content": event.get("content", ""),
                    "files": files_with_status
                })
            
            elif event_type == "done":
                await queue.put(event)
            
            elif event_type == "error":
                await queue.put(event)
        
        print(f"[Agent] Generation finished. generated_files={len(generated_files)}")
        
    except asyncio.CancelledError:
        print("[Agent] Task was cancelled")
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        await queue.put({"type": "error", "message": str(e)})
    finally:
        print("[Agent] Sending None to signal stream end")
        await queue.put(None)


async def stream_generator(prompt: str, base_url: str | None, mode: str = "auto"):
    """Generator that streams from queue as SSE."""
    queue = asyncio.Queue()
    
    # Start agent task
    task = asyncio.create_task(run_agent_and_collect(prompt, queue, base_url, mode))
    
    try:
        while True:
            # Wait for data with timeout to send heartbeats
            try:
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Check if the task has failed
                if task.done():
                    exc = task.exception()
                    if exc:
                        print(f"[Stream] Task failed with exception: {exc}")
                        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                        break
                # Send heartbeat
                yield ": heartbeat\n\n"
                continue
            
            if data is None:
                print("[Stream] Received None, ending stream")
                # Send a final empty comment to signal clean end
                yield ": end\n\n"
                # Small delay to ensure everything is flushed
                await asyncio.sleep(0.1)
                break
            
            print(f"[Stream] Sending event: {data.get('type', 'unknown')}")
            yield f"data: {json.dumps(data)}\n\n"
            # Small delay to ensure data is flushed
            await asyncio.sleep(0.01)
            
    except asyncio.CancelledError:
        # Generator was cancelled (likely because the client disconnected).
        # Don't re-raise here so we can run cleanup below and avoid killing the agent task immediately.
        print("[Stream] Generator was cancelled (client disconnected)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception:
            # If the client disconnected we can't yield; just log
            print('[Stream] Failed to yield error to client (likely disconnected)')
    finally:
        print("[Stream] Cleaning up")
        # If the agent task is still running, do not cancel it; let it finish in background.
        # This avoids killing long-running work when a client disconnects.
        if not task.done():
            print("[Stream] Agent task still running; leaving it to finish in background")
        else:
            # If task is done, await to propagate exceptions if any
            try:
                await task
            except asyncio.CancelledError:
                print('[Stream] Agent task was cancelled')
            except Exception as e:
                print(f'[Stream] Agent task finished with exception: {e}')


@ai_router.post("/generate")
async def generate(request: Request, prompt: str, mode: str = "auto"):
    """
    Generate a response to the user's prompt.
    
    Args:
        prompt: User's request
        mode: Generation mode - "auto" (detect complexity), "simple" (direct), or "agent" (full pipeline)
    """
    base_url = str(request.base_url).rstrip("/")
    return StreamingResponse(
        stream_generator(prompt, base_url, mode),
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
async def get_file_status(request: Request, filepath: str):
    filename = os.path.basename(filepath)
    base_url = str(request.base_url).rstrip("/")
    file_status = build_file_status(filename, filepath, "Status check", base_url=base_url)
    return JSONResponse(content=file_status)


@ai_router.post("/analyze-complexity")
async def analyze_complexity(prompt: str):
    """
    Analyze if a prompt requires complex multi-step processing.
    Returns the recommended mode.
    """
    is_complex = is_complex_task(prompt)
    return JSONResponse(content={
        "prompt": prompt,
        "is_complex": is_complex,
        "recommended_mode": "agent" if is_complex else "simple"
    })


# --- Cache Management Endpoints ---

@ai_router.get("/cache/stats")
async def get_cache_stats():
    """Get semantic cache statistics."""
    stats = semantic_cache.get_stats()
    return JSONResponse(content=stats)


@ai_router.delete("/cache/clear")
async def clear_cache():
    """Clear all cached entries."""
    semantic_cache.clear()
    return JSONResponse(content={"message": "Cache cleared successfully"})


@ai_router.get("/cache/entries")
async def get_cache_entries(limit: int = 20):
    """Get recent cache entries (for debugging/monitoring)."""
    entries = semantic_cache.entries[-limit:]
    return JSONResponse(content={
        "total": len(semantic_cache.entries),
        "showing": len(entries),
        "entries": [
            {
                "request": e.request[:100] + "..." if len(e.request) > 100 else e.request,
                "response": e.response[:200] + "..." if len(e.response) > 200 else e.response,
                "mode": e.mode,
                "hit_count": e.hit_count,
                "age_hours": round((time.time() - e.timestamp) / 3600, 1)
            }
            for e in reversed(entries)
        ]
    })


@ai_router.post("/cache/search")
async def search_cache(query: str):
    """Search the cache for similar queries."""
    result = semantic_cache.get(query)
    if result:
        return JSONResponse(content={
            "found": True,
            "similarity": result["similarity"],
            "cached_request": result["cached_request"],
            "response_preview": result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"]
        })
    return JSONResponse(content={"found": False})