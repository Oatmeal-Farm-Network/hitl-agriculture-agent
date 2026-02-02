# --- api.py --- (Enhanced API for farm advisory system)
from fastapi import FastAPI
from main import graph  # Import the enhanced graph
from langgraph.types import Command
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Farm Advisory API", version="2.0.0")

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [frontend_url]

if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[API] CORS enabled for origins: {allowed_origins}")


class ChatRequest(BaseModel):
    user_input: str
    thread_id: str


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "User-driven assessment",
            "Hybrid routing (keyword + LLM)",
            "Livestock RAG integration",
            "Separate advisory nodes"
        ]
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint for farm advisory system.
    
    Features:
    - User-driven assessment (starts with open question)
    - Hybrid routing (keyword matching + LLM fallback)
    - Specialized advisory nodes (livestock/crops/mixed)
    - Livestock RAG integration
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Check if the thread is already waiting for an answer
    state = graph.get_state(config)
    
    if state.next:
        # Resume the existing conversation
        print(f"[API] Resuming conversation for thread {request.thread_id}")
        events = graph.stream(Command(resume=request.user_input), config, stream_mode="values")
    else:
        # Start a brand new conversation with user's first message
        print(f"[API] Starting new conversation for thread {request.thread_id}")
        print(f"[API] First message: {request.user_input[:100]}...")
        # Include the user's first message so assessment can detect complete questions
        initial_history = [f"User: {request.user_input}"]
        events = graph.stream({"history": initial_history}, config, stream_mode="values")

    # Collect all events to ensure state is fully updated
    events_list = []
    for event in events:
        events_list.append(event)
        print(f"[API] Event keys: {list(event.keys())}")

    # Get the latest state after streaming
    final_state = graph.get_state(config)
    print(f"[API] Final state - Next nodes: {final_state.next}")
    print(f"[API] Final state - Has tasks: {len(final_state.tasks) if final_state.tasks else 0}")
    
    if final_state.next:
        # We hit an interrupt! Send the UI schema
        print(f"[API] Interrupt detected - need user input")
        if final_state.tasks and final_state.tasks[0].interrupts:
            return {
                "status": "requires_input",
                "ui": final_state.tasks[0].interrupts[0].value
            }
        else:
            print(f"[API] Warning: Next nodes exist but no interrupt found")
            return {
                "status": "error",
                "message": "Graph in unexpected state - next nodes but no interrupt"
            }
    
    # Get diagnosis/advice from final state
    final_values = final_state.values
    diagnosis = final_values.get("diagnosis", "")
    recommendations = final_values.get("recommendations", [])
    assessment_summary = final_values.get("assessment_summary", "")
    advisory_type = final_values.get("advisory_type", "unknown")
    
    print(f"[API] Final values:")
    print(f"  - assessment_summary: {assessment_summary[:100] if assessment_summary else 'None'}...")
    print(f"  - advisory_type: {advisory_type}")
    print(f"  - diagnosis: {diagnosis[:100] if diagnosis else 'None'}...")
    print(f"  - recommendations count: {len(recommendations)}")
    
    # Format response
    print(f"[API] Conversation complete for thread {request.thread_id}")
    return {
        "status": "complete",
        "diagnosis": diagnosis if diagnosis else "No diagnosis generated",
        "recommendations": recommendations if recommendations else ["Please try again"],
        "advisory_type": advisory_type,
        "assessment_summary": assessment_summary
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "farm-advisory-api"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"[API] Starting Farm Advisory API on port {port}")
    print(f"[API] Features: User-driven assessment + Hybrid routing + Livestock RAG")
    uvicorn.run(app, host="0.0.0.0", port=port)
