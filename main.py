# --- main.py --- (Langgraph workflow/bussiness logic)
import os
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Determine authentication method and initialize LLM
def initialize_llm():
    """
    Initialize ChatGoogleGenerativeAI with either:
    1. Vertex AI (Google Cloud) - if project/credentials are available
    2. Gemini Developer API - if API key is available
    """
    # Check for Vertex AI configuration
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Check if we should use Vertex AI
    if use_vertexai or project:
        # Vertex AI mode - use gemini-1.5-pro which is more widely available
        # You can also try: gemini-pro, gemini-1.5-flash-002, gemini-1.5-pro-002
        vertex_model = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash-lite")
        llm_kwargs = {
            "model": vertex_model,
            "temperature": 0,
        }
        
        # Add project if available
        if project:
            llm_kwargs["project"] = project
        
        # Add location if specified
        if location:
            llm_kwargs["location"] = location
        
        # Add credentials if service account file is provided
        if service_account_path:
            try:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                llm_kwargs["credentials"] = credentials
            except ImportError:
                raise ImportError(
                    "google-auth library is required for service account authentication. "
                    "Install it with: pip install google-auth"
                )
            except Exception as e:
                raise ValueError(f"Failed to load service account credentials from {service_account_path}: {e}")
        
        # Explicitly set vertexai=True if project is set
        if project:
            llm_kwargs["vertexai"] = True
        
        print(f"Initializing with Vertex AI (project: {project}, location: {location})")
        return ChatGoogleGenerativeAI(**llm_kwargs)
    
    # Fall back to Gemini Developer API with API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No authentication method found. Please set one of:\n"
            "  - GOOGLE_API_KEY or GEMINI_API_KEY for Gemini Developer API\n"
            "  - GOOGLE_CLOUD_PROJECT for Vertex AI (with ADC or GOOGLE_APPLICATION_CREDENTIALS)"
        )
    
    print("Initializing with Gemini Developer API (API key)")
    # Use gemini-2.5-flash-lite for cost effectiveness
    dev_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model=dev_model,
        temperature=0,
    )

class AdvisorState(TypedDict, total=False):
    history: List[str]
    advice: Optional[str]
    # Add pending question/options to state to ensure consistency across interrupts
    pending_question: Optional[str]
    pending_options: Optional[List[str]]

class UI_Decision(BaseModel):
    is_ready: bool = Field(description="True if you have enough information to provide a useful diagnosis. Assess based on information quality, not question count.")
    question: str = Field(description="The question for the user. Required if is_ready=False.")
    options: Optional[List[str]] = Field(default=None, description="REQUIRED if is_ready=False. Provide 3-4 specific, descriptive options relevant to the question. Examples: For 'Where are the spots?', use ['On the upper leaves', 'On the lower leaves', 'Throughout the plant', 'Only on the stems']. DO NOT use generic Yes/No options.")
    advice: Optional[str] = Field(default=None, description="The final diagnosis and recommendations. REQUIRED if is_ready=True.")


# Initialize LLM with automatic authentication detection
llm = initialize_llm()

def model_node(state: AdvisorState):
    """
    Generates the next question or final advice.
    Returns update to state with either advice (completing flow) or pending_question (requiring input).
    """
    structured_llm = llm.with_structured_output(UI_Decision)
    
    # Ensure history exists and is a list - read from state values
    current_history = state.get("history") or []
    if not isinstance(current_history, list):
        current_history = list(current_history) if current_history else []
    
    # Format history concisely
    history_text = "\n".join(current_history) if current_history else "No previous conversation."
    
    # Count questions asked to help guide decision (soft limit, not hard requirement)
    question_count = len([h for h in current_history if h.startswith("AI:")])
    
    # Hard maximum limit to prevent infinite loops (safety mechanism)
    MAX_QUESTIONS = 10
    if question_count >= MAX_QUESTIONS:
        # Force diagnosis after maximum questions
        prompt = f"""History:\n{history_text}\n\nYou have asked {question_count} questions. Provide your best diagnosis and recommendations based on the information collected, even if not perfect. Set is_ready=True and provide advice."""
        res = structured_llm.invoke(prompt)
        # Ensure advice is provided
        if not res.advice or res.advice.strip() == "":
            res.advice = f"Based on our conversation ({question_count} questions), here's my assessment: [Please provide diagnosis based on the symptoms and information discussed]"
        return {"advice": res.advice}
    
    prompt = f"""History:\n{history_text}\n\nTask: Diagnose crop issue through Q&A.

Rules:
- Don't repeat questions already asked
- If crop type mentioned, don't ask again
- Ask questions only if you truly need more information for accurate diagnosis

CRITICAL: When asking a question (is_ready=False), you MUST provide 3-4 SPECIFIC options relevant to that question.
- DO NOT use generic "Yes/No/Not sure" options
- Provide descriptive, specific choices that help diagnose the issue
- Examples:
  * Question: "Where are the spots located?" → Options: ["On upper leaves", "On lower leaves", "Throughout plant", "Only on stems"]
  * Question: "What color are the spots?" → Options: ["Yellow", "Brown/black", "White/powdery", "Red/orange"]
  * Question: "How would you describe the texture?" → Options: ["Fuzzy/cotton-like", "Powdery", "Smooth/raised bumps", "Sticky"]

Decision Logic - Assess if you have ENOUGH information:
You have enough info if you know:
✓ Crop/plant type
✓ Main symptoms (what's wrong)
✓ Symptom patterns (where, how they appear)
✓ Severity or progression

Set is_ready=True when:
- You have the above information AND can provide useful diagnosis/advice
- Even if not 100% certain, provide your best diagnosis with recommendations

Set is_ready=False when:
- Missing critical information (e.g., don't know crop type, unclear symptoms)
- Need one more specific detail to distinguish between similar diseases
- MUST provide 3-4 specific options (not Yes/No)

Questions asked so far: {question_count} (max: {MAX_QUESTIONS})
Note: After 7+ questions, strongly consider providing diagnosis even if not perfect - you likely have enough info."""
    
    res = structured_llm.invoke(prompt)

    if not res.is_ready:
        # Validate options - if missing or generic, try to generate better ones
        options = res.options
        if not options or len(options) < 3 or all(opt.lower() in ["yes", "no", "not sure", "maybe"] for opt in options):
            # Fallback: generate context-aware options based on question
            question_lower = res.question.lower()
            if "where" in question_lower or "location" in question_lower:
                options = ["On the upper leaves", "On the lower leaves", "Throughout the plant", "On stems or fruits"]
            elif "color" in question_lower or "appearance" in question_lower:
                options = ["Yellow", "Brown or black", "White or powdery", "Other color"]
            elif "texture" in question_lower or "feel" in question_lower:
                options = ["Fuzzy or cotton-like", "Powdery", "Smooth or raised", "Sticky or wet"]
            elif "size" in question_lower:
                options = ["Very small (pinpoint)", "Small (1-2mm)", "Medium (3-5mm)", "Large (5mm+)"]
            else:
                # Last resort: at least make them more descriptive
                options = ["Option 1 (describe)", "Option 2 (describe)", "Option 3 (describe)", "Other/Not sure"]
        
        # Return pending question/options to state instead of interrupting directly
        return {
            "pending_question": res.question,
            "pending_options": options,
            "advice": None # Clear any previous advice if exists
        }
    
    # Ensure advice is provided
    advice = res.advice
    if not advice or advice.strip() == "":
        # Fallback: generate advice from history if model didn't provide it
        advice = f"Based on our conversation, here's my diagnosis and recommendations. Please consult with a local agricultural expert for confirmation."
    
    return {"advice": advice}

def human_node(state: AdvisorState):
    """
    Handles user interaction.
    Reads pending question from state, interrupts for user input, and updates history.
    """
    question = state.get("pending_question")
    options = state.get("pending_options")

    if not question:
        # Should not happen if logic is correct, but safe fallback
        return Command(goto="model_node")

    ui_schema = {
        "type": "quiz",
        "question": question,
        "options": options
    }

    # The interrupt pauses the thread here
    user_response = interrupt(ui_schema)

    current_history = state.get("history") or []
    updated_history = current_history + [f"AI: {question}", f"User: {user_response}"]

    return Command(
        update={
            "history": updated_history,
            "pending_question": None, # Clear pending
            "pending_options": None
        },
        goto="model_node"
    )

def should_continue(state: AdvisorState):
    if state.get("advice"):
        return END
    return "human_node"

# Compile outside of any functions so it persists when imported
builder = StateGraph(AdvisorState)
builder.add_node("model_node", model_node)
builder.add_node("human_node", human_node)

builder.add_edge(START, "model_node")
builder.add_conditional_edges("model_node", should_continue)
# human_node always goes back to model_node (handled by Command logic, but good to have edge definition if not using Command goto)
# But since we use Command(goto="model_node"), we don't strictly need a static edge,
# however, for visualization and correctness it's good practice.
builder.add_edge("human_node", "model_node")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)