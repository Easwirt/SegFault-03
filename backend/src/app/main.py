from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def entry():
    return "Hello"

