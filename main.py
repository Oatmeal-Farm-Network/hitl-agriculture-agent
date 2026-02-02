# --- main.py --- (Enhanced LangGraph farm advisory system with livestock RAG)
import os
import re
from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_google_genai import ChatGoogleGenerativeAI

# Livestock RAG dependencies  
try:
    import pymssql
    from google.cloud import firestore
    from google.cloud.firestore_v1.vector import Vector
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
    from langchain_google_vertexai import VertexAIEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    print("[Warning] RAG dependencies not installed. Livestock RAG will be disabled.")
    RAG_AVAILABLE = False

load_dotenv()

print("[Main] Loading enhanced farm advisory system...")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Livestock Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "").strip(),
    "port": int(os.getenv("DB_PORT", "1433").strip()) if os.getenv("DB_PORT") else 1433,
    "user": os.getenv("DB_USER", "").strip(),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "").strip(),
}

# Allowed tables for security
ALLOWED_TABLES = [
    "Speciesavailable", "Speciesbreedlookuptable", "Speciescategory",
    "Speciescolorlookuptable", "Speciespatternlookuptable", "Speciesregistrationtypelookuptable",
]

# GCP Configuration
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# RAG Configuration
EMBEDDING_MODEL = "text-embedding-004"
TOP_K_RESULTS = 10
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "charlie").strip()
FIRESTORE_COLLECTION = "livestock_knowledge"

# Assessment Configuration
MAX_QUESTIONS = 8

print(f"[Config] GCP Project: {GCP_PROJECT or 'Not set'}")
print(f"[Config] RAG Available: {RAG_AVAILABLE}")

# ============================================================================
# DATABASE CLASS
# ============================================================================

class Database:
    """Manages database connections and queries for livestock data."""

    def __init__(self):
        self._connection = None
        self._allowed_tables = [t.lower() for t in ALLOWED_TABLES]

    @property
    def connection(self):
        """Lazy connection to database."""
        if self._connection is None and RAG_AVAILABLE:
            try:
                if all([DB_CONFIG["host"], DB_CONFIG["user"], DB_CONFIG["database"]]):
                    self._connection = pymssql.connect(
                        server=DB_CONFIG["host"],
                        port=DB_CONFIG["port"],
                        user=DB_CONFIG["user"],
                        password=DB_CONFIG["password"],
                        database=DB_CONFIG["database"],
                        as_dict=True
                    )
                    print(f"[DB] Connected to {DB_CONFIG['database']}")
            except Exception as e:
                print(f"[DB] Connection failed: {e}")
        return self._connection

    def _validate_query(self, query: str) -> None:
        """Validate query only accesses allowed tables."""
        query_lower = query.lower()
        tables = re.findall(r'from\s+\[?(\w+)\]?', query_lower)
        tables += re.findall(r'join\s+\[?(\w+)\]?', query_lower)
        for table in tables:
            if table not in self._allowed_tables:
                raise PermissionError(f"Access denied to table: {table}")

    def execute(self, query: str) -> List[Dict]:
        """Execute a SELECT query and return results."""
        if not self.connection:
            return []
        try:
            self._validate_query(query)
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results if results else []
        except Exception as e:
            print(f"[DB] Query error: {e}")
            return []

db = Database()

# ============================================================================
# RAG SYSTEM CLASS
# ============================================================================

class RAGSystem:
    """RAG system using Firestore Vector Search for livestock knowledge."""

    def __init__(self):
        self._db = None
        self._initialized = False
        self._embeddings = None

    def _init_embeddings(self):
        """Initialize embeddings model."""
        if self._embeddings is None and GCP_PROJECT and RAG_AVAILABLE:
            try:
                self._embeddings = VertexAIEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    project=GCP_PROJECT,
                    location=GCP_LOCATION
                )
                print(f"[RAG] Embeddings initialized ({EMBEDDING_MODEL})")
            except Exception as e:
                print(f"[RAG] Embeddings init failed: {e}")

    @property
    def firestore_db(self):
        """Lazy initialization of Firestore client."""
        if self._db is None and GCP_PROJECT and RAG_AVAILABLE:
            credentials = None
            if GCP_CREDENTIALS:
                try:
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(
                        GCP_CREDENTIALS,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                except Exception as e:
                    print(f"[RAG] Credentials load failed: {e}")
            try:
                if credentials:
                    self._db = firestore.Client(
                        project=GCP_PROJECT, database=FIRESTORE_DATABASE, credentials=credentials
                    )
                else:
                    self._db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)
                print(f"[RAG] Connected to Firestore ({FIRESTORE_DATABASE})")
            except Exception as e:
                print(f"[RAG] Firestore connection failed: {e}")
        return self._db

    @property
    def collection(self):
        """Get the Firestore collection."""
        if self.firestore_db:
            return self.firestore_db.collection(FIRESTORE_COLLECTION)
        return None

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        self._init_embeddings()
        if self._embeddings:
            return self._embeddings.embed_query(text)
        return []

    def initialize(self):
        """Initialize the RAG system."""
        if not self._initialized and self.collection:
            try:
                docs = list(self.collection.limit(1).get())
                self._initialized = len(docs) > 0
                if self._initialized:
                    print(f"[RAG] Index ready")
            except Exception as e:
                print(f"[RAG] Init error: {e}")
        return self._initialized

    def search(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for relevant livestock documents."""
        if not self._initialized:
            self.initialize()
        if not self.collection or not query:
            return []
        try:
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
            vector_query = self.collection.find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_embedding),
                distance_measure=DistanceMeasure.COSINE,
                limit=n_results
            )
            results = vector_query.get()
            return [{"content": doc.to_dict().get("content", ""), 
                    "metadata": doc.to_dict().get("metadata", {})} 
                   for doc in results]
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return []

    def get_context_for_query(self, query: str) -> str:
        """Get formatted context string for LLM."""
        results = self.search(query)
        if not results:
            return ""
        context_parts = ["Relevant livestock information from database:\n"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['content']}")
        return "\n".join(context_parts)

rag = RAGSystem()

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def initialize_llm():
    """Initialize ChatGoogleGenerativeAI with Vertex AI or Developer API."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if use_vertexai or project:
        vertex_model = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash-lite")
        llm_kwargs = {"model": vertex_model, "temperature": 0}
        if project:
            llm_kwargs["project"] = project
        if location:
            llm_kwargs["location"] = location
        if service_account_path:
            try:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                llm_kwargs["credentials"] = credentials
            except Exception as e:
                print(f"[LLM] Credentials error: {e}")
        if project:
            llm_kwargs["vertexai"] = True
        print(f"[LLM] Using Vertex AI ({vertex_model})")
        return ChatGoogleGenerativeAI(**llm_kwargs)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No authentication found. Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT")
    
    dev_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    print(f"[LLM] Using Developer API ({dev_model})")
    return ChatGoogleGenerativeAI(model=dev_model, temperature=0)

llm = initialize_llm()

# ============================================================================
# STATE DEFINITION
# ============================================================================

class FarmState(TypedDict, total=False):
    """State for managing farm information and diagnostics"""
    farm_name: Optional[str]
    location: Optional[str]
    farm_size: Optional[str]
    crops: Optional[List[str]]
    current_issues: Optional[List[str]]
    history: Optional[List[str]]
    diagnosis: Optional[str]
    soil_info: Optional[Dict[str, Any]]
    weather_conditions: Optional[Dict[str, Any]]
    management_practices: Optional[List[str]]
    recommendations: Optional[List[str]]
    assessment_summary: Optional[str]  # CRITICAL for routing
    advisory_type: Optional[str]  # 'livestock', 'crops', or 'mixed'

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AssessmentDecision(BaseModel):
    is_complete: bool = Field(description="True if enough information collected")
    question: str = Field(description="Question to ask. Required if is_complete=False")
    options: Optional[List[str]] = Field(default=None, description="3-4 options if is_complete=False")
    assessment_summary: Optional[str] = Field(default=None, description="Summary if is_complete=True")

class QueryClassification(BaseModel):
    category: str = Field(description="'livestock', 'crops', or 'mixed'")
    confidence: str = Field(description="'high' or 'low'")
    reasoning: str = Field(description="Brief explanation")

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def assessment_node(state: FarmState):
    """User-driven assessment: starts with open question, then contextual follow-ups."""
    structured_llm = llm.with_structured_output(AssessmentDecision)
    
    history = state.get("history") or []
    location = state.get("location")
    farm_size = state.get("farm_size")
    crops = state.get("crops") or []
    current_issues = state.get("current_issues") or []
    
    if state.get("assessment_summary"):
        return {}
    
    questions_asked = [h for h in history if h.startswith("AI:")]
    question_count = len(questions_asked)
    is_first_interaction = question_count == 0 and not current_issues
    
    # Check if user provided a complete question in first message
    print(f"[Assessment] Checking for fast-track - is_first_interaction: {is_first_interaction}, history length: {len(history)}")
    if is_first_interaction and history:
        first_user_message = None
        for msg in history:
            if msg.startswith("User:"):
                first_user_message = msg.replace("User:", "").strip()
                break
        
        print(f"[Assessment] First user message: {first_user_message[:100] if first_user_message else 'None'}...")
        
        if first_user_message and len(first_user_message) > 15:
            msg_lower = first_user_message.lower()
            
            # Check for SPECIFIC crops (not just generic "field" or "crop")
            specific_crops = ["paddy", "rice", "wheat", "maize", "corn", "cotton", "soybean", 
                            "tomato", "potato", "sugarcane", "barley", "millet"]
            
            # Check for SPECIFIC livestock (not just generic "animal" or "livestock")  
            specific_livestock = ["cattle", "cow", "buffalo", "sheep", "goat", "pig", "chicken", 
                                "duck", "turkey", "horse"]
            
            has_specific_crop = any(crop in msg_lower for crop in specific_crops)
            has_specific_livestock = any(animal in msg_lower for animal in specific_livestock)
            
            # Only fast-track if question mentions SPECIFIC crop or livestock
            if has_specific_crop or has_specific_livestock:
                print(f"[Assessment] ✓ Specific crop/livestock detected - fast-tracking")
                current_issues = [first_user_message]
                
                # Extract the specific items
                detected_items = []
                for crop in specific_crops:
                    if crop in msg_lower:
                        detected_items.append(crop)
                for animal in specific_livestock:
                    if animal in msg_lower:
                        detected_items.append(animal)
                
                if detected_items:
                    crops = detected_items
                
                # Build summary and complete assessment immediately
                summary_parts = [f"Farmer seeks assistance with: {first_user_message}"]
                if detected_items:
                    summary_parts.append(f"Related to: {', '.join(set(detected_items))}")
                
                assessment_summary = " | ".join(summary_parts)
                print(f"[Assessment] ✓✓✓ Fast-track COMPLETE ✓✓✓")
                print(f"[Assessment] Summary: {assessment_summary}")
                
                return {
                    "assessment_summary": assessment_summary,
                    "current_issues": current_issues,
                    "crops": crops if crops else None
                }
            else:
                print(f"[Assessment] Generic question detected (no specific crop/livestock) - will ask clarifying questions")
                # Store the issue and continue with contextual follow-up questions
                current_issues = [first_user_message]
                # Don't return here - let the prompt generation handle the contextual question
    
    print(f"[Assessment] Not fast-tracking - will ask questions")
    
    # Update current_issues in state if we captured it from first message
    if is_first_interaction and current_issues and not state.get("current_issues"):
        # We need to update the state with the captured issue before asking questions
        print(f"[Assessment] Storing user's initial concern: {current_issues}")
        # Continue to prompt generation with current_issues populated
    
    # Determine completion
    should_complete = False
    has_issue = bool(current_issues)
    has_crops_or_livestock = bool(crops)
    has_location = bool(location)
    
    if question_count >= MAX_QUESTIONS:
        should_complete = True
    elif has_issue and has_crops_or_livestock and has_location and question_count >= 2:
        should_complete = True
    elif has_issue and has_crops_or_livestock and question_count >= 3:
        should_complete = True
    
    if should_complete:
        summary_parts = [f"Farmer seeks assistance with: {', '.join(current_issues) if current_issues else 'general farm advice'}"]
        if crops:
            summary_parts.append(f"Growing/Raising: {', '.join(crops)}")
        if location:
            summary_parts.append(f"Location: {location}")
        assessment_summary = " | ".join(summary_parts)
        print(f"[Assessment] ✓ Complete: {assessment_summary}")
        return {"assessment_summary": assessment_summary}
    
    # Build prompt
    # Check if user already provided their concern in the first message
    user_has_concern = bool(current_issues)
    
    if is_first_interaction and not user_has_concern:
        # Truly first interaction - no user message yet
        prompt = """You are a friendly farm advisor. This is your first interaction.

Ask ONE open-ended question to understand what brings them here today.
Be warm and welcoming. Provide 3-4 option suggestions but allow free-text response.
Set is_complete=False."""
    elif is_first_interaction and user_has_concern:
        # User already stated their concern - ask contextual follow-up
        user_concern = ', '.join(current_issues)
        prompt = f"""You are a friendly farm advisor. The farmer just asked: "{user_concern}"

This is your FIRST follow-up question. Based on their concern, ask ONE specific clarifying question.

For example:
- If they mention "animal/breed for field" → Ask what type of field/crop
- If they mention a crop → Ask about their specific issue or goal
- If they mention livestock → Ask about their farm setup or goal

Provide 3-4 specific, relevant options based on their question.
Set is_complete=False.

DO NOT repeat what they said. Just ask your clarifying question."""
    else:
        history_text = "\n".join(history[-10:])
        prompt = f"""Farm Info: Location {'✓' if location else '✗'}, Crops/Livestock {'✓' if crops else '✗'}

History:
{history_text}

Ask ONE relevant follow-up question. Provide 3-4 specific options (not Yes/No).
Questions asked: {question_count}/{MAX_QUESTIONS}

Set is_complete=True when you have:
✓ User's issue/concern
✓ What they're growing/raising  
✓ Location (if needed)"""
    
    res = structured_llm.invoke(prompt)
    
    if not res.is_complete:
        options = res.options or ["Option 1", "Option 2", "Option 3", "Other"]
        if len(options) < 3:
            q_lower = res.question.lower()
            if "location" in q_lower or "where" in q_lower:
                options = ["North region", "South region", "Central region", "Other"]
            elif "size" in q_lower:
                options = ["Small (1-5 acres)", "Medium (5-20 acres)", "Large (20+ acres)", "Other"]
            else:
                options = ["Option 1", "Option 2", "Option 3", "Other"]
        
        ui_schema = {"type": "quiz", "question": res.question, "options": options}
        user_response = interrupt(ui_schema)
        
        updates = {"history": history + [f"AI: {res.question}", f"User: {user_response}"]}
        
        q_lower = res.question.lower()
        if is_first_interaction:
            updated_issues = list(current_issues)
            updated_issues.append(user_response)
            updates["current_issues"] = updated_issues
        elif "location" in q_lower or "where" in q_lower or "region" in q_lower:
            updates["location"] = user_response
        elif "size" in q_lower:
            updates["farm_size"] = user_response
        elif "crop" in q_lower or "growing" in q_lower or "plant" in q_lower or "livestock" in q_lower or "animal" in q_lower:
            updated_crops = list(crops)
            updated_crops.append(user_response)
            updates["crops"] = updated_crops
        
        return updates
    
    return {"assessment_summary": res.assessment_summary or "Assessment complete"}


def routing_node(state: FarmState) -> Dict[str, str]:
    """Hybrid routing: keyword matching first, LLM fallback for uncertain cases."""
    crops = state.get("crops", [])
    issues = state.get("current_issues", [])
    assessment = state.get("assessment_summary", "")
    
    query_text = f"{' '.join(crops)} {' '.join(issues)} {assessment}".lower()
    
    livestock_keywords = [
        "cattle", "cow", "sheep", "goat", "pig", "chicken", "duck", "turkey", 
        "horse", "rabbit", "livestock", "animal", "breed", "dairy", "beef", 
        "poultry", "lamb", "calf", "piglet", "chick"
    ]
    crop_keywords = [
        "corn", "maize", "wheat", "rice", "barley", "soybean", "cotton",
        "tomato", "potato", "vegetable", "fruit", "grain", "crop", "plant",
        "paddy", "field", "harvest"
    ]
    
    livestock_matches = sum(1 for k in livestock_keywords if k in query_text)
    crop_matches = sum(1 for k in crop_keywords if k in query_text)
    
    print(f"[Routing] Keywords - Livestock: {livestock_matches}, Crops: {crop_matches}")
    
    if livestock_matches > 0 and crop_matches == 0:
        print(f"[Routing] → livestock (keyword)")
        return {"advisory_type": "livestock"}
    if crop_matches > 0 and livestock_matches == 0:
        print(f"[Routing] → crops (keyword)")
        return {"advisory_type": "crops"}
    if livestock_matches > 0 and crop_matches > 0:
        print(f"[Routing] → mixed (keywords)")
        return {"advisory_type": "mixed"}
    
    # LLM fallback
    print(f"[Routing] Using LLM...")
    classifier = llm.with_structured_output(QueryClassification)
    prompt = f"""Classify as 'livestock', 'crops', or 'mixed':
Crops/Animals: {', '.join(crops) if crops else 'Not specified'}
Issues: {', '.join(issues) if issues else 'None'}"""
    
    try:
        result = classifier.invoke(prompt)
        print(f"[Routing] LLM: {result.category}")
        if "livestock" in result.category.lower():
            return {"advisory_type": "livestock"}
        elif "mixed" in result.category.lower():
            return {"advisory_type": "mixed"}
    except Exception as e:
        print(f"[Routing] LLM error: {e}")
    
    return {"advisory_type": "crops"}


def livestock_advisory_node(state: FarmState):
    """Livestock advisory with RAG integration."""
    print("\n[Livestock Advisory] Processing...")
    print(f"[Livestock Advisory] 🐄 Using RAG for breed recommendations")
    
    location = state.get("location", "Unknown")
    crops = state.get("crops") or []
    issues = state.get("current_issues") or []
    assessment = state.get("assessment_summary", "")
    
    livestock_context = ""
    if RAG_AVAILABLE:
        query_text = f"{', '.join(crops)} {', '.join(issues)} {assessment}"
        try:
            rag.initialize()
            livestock_context = rag.get_context_for_query(query_text)
            if livestock_context:
                print(f"[Livestock Advisory] ✓ RAG context retrieved")
        except Exception as e:
            print(f"[Livestock Advisory] RAG error: {e}")
    
    prompt = f"""You are a friendly livestock advisor. Provide brief, practical advice.

Farmer's question: {', '.join(issues) if issues else 'General livestock inquiry'}
Location: {location}
Context: {', '.join(crops) if crops else 'General farming'}

{livestock_context}

Provide a SHORT response (3-4 sentences max) covering:
- Direct answer to their question
- Top 2-3 breed recommendations with brief reasons
- 1-2 key management tips

Use simple, conversational language. NO markdown formatting, NO asterisks, NO headers.
Write like you're talking to a friend."""
    
    try:
        response = llm.invoke(prompt)
        advice = response.content if hasattr(response, 'content') else str(response)
        
        # Extract key recommendations as bullet points
        lines = advice.split('\n')
        recommendations = []
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['recommend', 'consider', 'try', 'use', 'breed']):
                # Clean up any markdown
                line = line.replace('**', '').replace('*', '').replace('#', '').strip('- •')
                if line and len(line) > 20:
                    recommendations.append(line)
        
        return {
            "diagnosis": advice, 
            "recommendations": recommendations[:5] if recommendations else ["Consider consulting a local livestock expert for specific breed recommendations"]
        }
    except Exception as e:
        print(f"[Livestock Advisory] Error: {e}")
        return {"diagnosis": "I'm having trouble generating advice right now. Please try again.", "recommendations": ["Consult a local livestock expert"]}


def crop_advisory_node(state: FarmState):
    """Crop advisory."""
    print("\n[Crop Advisory] Processing...")
    print(f"[Crop Advisory] 🌾 NO RAG - pure crop expertise")
    
    location = state.get("location", "Unknown")
    crops = state.get("crops") or []
    issues = state.get("current_issues") or []
    assessment = state.get("assessment_summary", "")
    
    prompt = f"""You are a friendly agronomist. Provide brief, practical crop advice.

Farmer's question: {', '.join(issues) if issues else 'General crop inquiry'}
Crops: {', '.join(crops) if crops else 'Not specified'}
Location: {location}

Provide a SHORT response (3-4 sentences max) covering:
- Direct answer to their question
- Top 2-3 actionable steps
- Key tips for their specific situation

Use simple, conversational language. NO markdown formatting, NO asterisks, NO headers.
Write like you're talking to a friend."""
    
    try:
        response = llm.invoke(prompt)
        advice = response.content if hasattr(response, 'content') else str(response)
        
        # Extract key recommendations
        lines = advice.split('\n')
        recommendations = []
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['recommend', 'consider', 'try', 'apply', 'use']):
                line = line.replace('**', '').replace('*', '').replace('#', '').strip('- •')
                if line and len(line) > 20:
                    recommendations.append(line)
        
        return {
            "diagnosis": advice,
            "recommendations": recommendations[:5] if recommendations else ["Consider consulting a local agricultural extension office"]
        }
    except Exception as e:
        print(f"[Crop Advisory] Error: {e}")
        return {"diagnosis": "I'm having trouble generating advice right now. Please try again.", "recommendations": ["Consult a local agricultural expert"]}


def mixed_advisory_node(state: FarmState):
    """Integrated crop+livestock advisory with RAG."""
    print("\n[Mixed Advisory] Processing...")
    print(f"[Mixed Advisory] 🌾🐄 Using RAG for integrated farm advice")
    
    location = state.get("location", "Unknown")
    crops = state.get("crops") or []
    issues = state.get("current_issues") or []
    assessment = state.get("assessment_summary", "")
    
    livestock_context = ""
    if RAG_AVAILABLE:
        query_text = f"{', '.join(crops)} {', '.join(issues)} {assessment}"
        try:
            rag.initialize()
            livestock_context = rag.get_context_for_query(query_text)
            if livestock_context:
                print(f"[Mixed Advisory] ✓ RAG context retrieved")
        except Exception as e:
            print(f"[Mixed Advisory] RAG error: {e}")
    
    prompt = f"""You are a friendly integrated farming advisor. Provide brief, practical advice.

Farmer's question: {', '.join(issues) if issues else 'General farming inquiry'}
Context: {', '.join(crops) if crops else 'Mixed farming'}
Location: {location}

{livestock_context}

Provide a SHORT response (3-4 sentences max) covering:
- Direct answer to their question
- Top 2-3 specific breed/variety recommendations
- 1-2 integration tips (how crops and livestock work together)

Use simple, conversational language. NO markdown formatting, NO asterisks, NO headers.
Write like you're talking to a friend."""
    
    try:
        response = llm.invoke(prompt)
        advice = response.content if hasattr(response, 'content') else str(response)
        
        # Extract key recommendations
        lines = advice.split('\n')
        recommendations = []
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['recommend', 'consider', 'try', 'use', 'breed', 'variety']):
                line = line.replace('**', '').replace('*', '').replace('#', '').strip('- •')
                if line and len(line) > 20:
                    recommendations.append(line)
        
        return {
            "diagnosis": advice,
            "recommendations": recommendations[:5] if recommendations else ["Consider consulting a local integrated farming expert"]
        }
    except Exception as e:
        print(f"[Mixed Advisory] Error: {e}")
        return {"diagnosis": "I'm having trouble generating advice right now. Please try again.", "recommendations": ["Consult a local farming expert"]}


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_assessment(state: FarmState) -> str:
    """Route from assessment to routing node when complete."""
    summary = state.get("assessment_summary")
    advisory_type = state.get("advisory_type")
    
    print(f"\n[Route] route_after_assessment called:")
    print(f"  - assessment_summary exists: {bool(summary and summary.strip())}")
    print(f"  - advisory_type exists: {bool(advisory_type and advisory_type.strip())}")
    
    if summary and summary.strip() and not (advisory_type and advisory_type.strip()):
        print(f"  → routing_node (assessment complete, need advisory type)")
        return "routing_node"
    if summary and summary.strip():
        print(f"  → routing_node (assessment complete)")
        return "routing_node"
    print(f"  → assessment_node (continue assessment)")
    return "assessment_node"


def route_to_advisory(state: FarmState) -> str:
    """Route to appropriate advisory node."""
    advisory_type = state.get("advisory_type", "crops")
    if advisory_type == "livestock":
        return "livestock_advisory_node"
    elif advisory_type == "mixed":
        return "mixed_advisory_node"
    return "crop_advisory_node"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

print("[Graph] Building farm advisory graph...")

builder = StateGraph(FarmState)

# Add nodes
builder.add_node("assessment_node", assessment_node)
builder.add_node("routing_node", routing_node)
builder.add_node("livestock_advisory_node", livestock_advisory_node)
builder.add_node("crop_advisory_node", crop_advisory_node)
builder.add_node("mixed_advisory_node", mixed_advisory_node)

# Add edges
builder.add_edge(START, "assessment_node")

builder.add_conditional_edges(
    "assessment_node",
    route_after_assessment,
    {"assessment_node": "assessment_node", "routing_node": "routing_node"}
)

builder.add_conditional_edges(
    "routing_node",
    route_to_advisory,
    {
        "livestock_advisory_node": "livestock_advisory_node",
        "crop_advisory_node": "crop_advisory_node",
        "mixed_advisory_node": "mixed_advisory_node"
    }
)

builder.add_edge("livestock_advisory_node", END)
builder.add_edge("crop_advisory_node", END)
builder.add_edge("mixed_advisory_node", END)

# Compile
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

print("\n" + "="*60)
print("✓ Farm Advisory Graph Compiled Successfully!")
print("="*60)
print("Features:")
print("  ✓ User-driven assessment (open question → contextual follow-ups)")
print("  ✓ Hybrid routing (keyword matching + LLM fallback)")
print("  ✓ Separate advisory nodes (livestock/crops/mixed)")
print("  ✓ Livestock RAG integration (breed recommendations)")
print("  ✓ Enhanced state management (assessment_summary + advisory_type)")
print("="*60)
