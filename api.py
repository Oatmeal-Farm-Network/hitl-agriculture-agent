# --- api.py --- (API DESIGN/BACKEND)
from fastapi import FastAPI, Body
from main import graph # Your existing graph
from langgraph.types import Command
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    thread_id: str


@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Check if the thread is already waiting for an answer
    state = graph.get_state(config)
    
    if state.next:
        # Resume the existing quiz - resume with user input
        # The Command will update history and loop back to advisor_node
        events = graph.stream(Command(resume=request.user_input), config, stream_mode="values")
    else:
        # Start a brand new diagnosis with initial user message
        initial_history = [f"User: {request.user_input}"]
        events = graph.stream({"history": initial_history}, config, stream_mode="values")

    # Collect all events to ensure state is fully updated
    events_list = []
    for event in events:
        events_list.append(event)

    # Get the latest state after streaming (this ensures checkpoint is saved)
    final_state = graph.get_state(config)
    
    if final_state.next:
        # We hit an interrupt! Send the UI schema
        return {
            "status": "requires_input",
            "ui": final_state.tasks[0].interrupts[0].value
        }
    
    # Get advice from final event or state
    final_event = events_list[-1] if events_list else {}
    advice = final_event.get("advice") or final_state.values.get("advice")
    
    return {
        "status": "complete",
        "advice": advice
    }