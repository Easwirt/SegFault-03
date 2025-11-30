from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.controller.ai_controller import ai_router
from app.api.controller.mcp_controller import mcp_router

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)
app.include_router(mcp_router)

# Get frontend path (relative to this file)
FRONTEND_PATH = Path(__file__).parent.parent.parent.parent / "frontend"

@app.get("/")
def entry():
    """Serve the frontend index.html"""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return f"Hello - Frontend not found {index_path}"

# Mount static files for frontend assets (CSS, JS)
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

