# AI Agent Frontend

A simple web interface for interacting with the AI Agent backend.

## Quick Start

### Option 1: Using Python's built-in server

```bash
cd frontend
python -m http.server 3000
```

Then open http://localhost:3000 in your browser.

### Option 2: Using Node.js (if installed)

```bash
cd frontend
npx serve -p 3000
```

### Option 3: Open directly

Simply open `index.html` in your browser. Note: Some features may not work due to CORS when using `file://` protocol.

## Configuration

The frontend connects to the backend at `http://localhost:8000` by default. To change this, edit the `API_BASE_URL` variable in `app.js`.

## Running the Backend

Make sure the FastAPI backend is running:

```bash
cd backend/src
uvicorn app.main:app --reload
```

## Features

- **Real-time streaming**: Watch the AI agent plan and execute tasks in real-time
- **Plan visualization**: See the generated plan with task progress indicators
- **Task execution tracking**: Monitor each task as it completes
- **Live token streaming**: See LLM output as it's generated
- **Final answer display**: View the synthesized response

## Event Types

The frontend handles the following SSE events from the backend:

| Event | Description |
|-------|-------------|
| `node_start` | When a workflow node (planner/executor/responder) starts |
| `plan` | The generated task list |
| `task_complete` | A task has finished executing |
| `token` | LLM token streaming |
| `final_answer` | The complete synthesized response |
| `done` | Stream complete |
| `error` | An error occurred |
