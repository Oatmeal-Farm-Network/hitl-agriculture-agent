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
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

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

# Weather API dependencies
try:
    import requests
    WEATHER_AVAILABLE = True
except ImportError:
    print("[Warning] requests not installed. Weather service will be disabled.")
    WEATHER_AVAILABLE = False
    requests = None  # Set to None to avoid errors

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
# WEATHER SERVICE CLASS
# ============================================================================

class WeatherService:
    """Weather service for fetching current and forecast data from weather APIs."""

    def __init__(self):
        self._api_key = os.getenv("WEATHER_API_KEY", "").strip()
        self._provider = os.getenv("WEATHER_API_PROVIDER", "openweathermap").strip().lower()
        self._cache = {}  # Simple in-memory cache: {location: (data, timestamp)}
        self._cache_ttl = 300  # 5 minutes
        self._available = WEATHER_AVAILABLE and bool(self._api_key)

    def _is_cache_valid(self, location: str) -> bool:
        """Check if cached data is still valid."""
        if location not in self._cache:
            return False
        data, timestamp = self._cache[location]
        import time
        return (time.time() - timestamp) < self._cache_ttl

    def _get_from_cache(self, location: str) -> Optional[Dict[str, Any]]:
        """Get weather data from cache if valid."""
        if self._is_cache_valid(location):
            return self._cache[location][0]
        return None

    def _save_to_cache(self, location: str, data: Dict[str, Any]):
        """Save weather data to cache."""
        import time
        self._cache[location] = (data, time.time())

    def _fetch_openweathermap(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather from OpenWeatherMap API."""
        if not self._api_key:
            return None
        try:
            # First, try to get coordinates from location name
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {"q": location, "limit": 1, "appid": self._api_key}
            geo_response = requests.get(geo_url, params=geo_params, timeout=5)
            
            if geo_response.status_code != 200:
                print(f"[Weather] Geo API error: {geo_response.status_code}")
                return None
            
            geo_data = geo_response.json()
            if not geo_data:
                print(f"[Weather] Location not found: {location}")
                return None
            
            lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
            
            # Fetch current weather
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": self._api_key,
                "units": "metric"  # Use Celsius
            }
            weather_response = requests.get(weather_url, params=weather_params, timeout=5)
            
            if weather_response.status_code != 200:
                print(f"[Weather] Weather API error: {weather_response.status_code}")
                return None
            
            data = weather_response.json()
            
            # Format response
            return {
                "location": f"{geo_data[0].get('name', location)}, {geo_data[0].get('country', '')}",
                "temperature": round(data["main"]["temp"]),
                "feels_like": round(data["main"]["feels_like"]),
                "condition": data["weather"][0]["description"].title(),
                "humidity": data["main"]["humidity"],
                "wind_speed": round(data["wind"].get("speed", 0) * 3.6, 1),  # Convert m/s to km/h
                "pressure": data["main"]["pressure"],
                "clouds": data["clouds"]["all"],
                "visibility": data.get("visibility", 0) / 1000 if data.get("visibility") else None,  # Convert to km
            }
        except Exception as e:
            print(f"[Weather] OpenWeatherMap error: {e}")
            return None

    def _fetch_weatherapi(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather from WeatherAPI.com."""
        if not self._api_key:
            return None
        try:
            url = "https://api.weatherapi.com/v1/current.json"
            params = {"key": self._api_key, "q": location, "aqi": "no"}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200:
                print(f"[Weather] WeatherAPI error: {response.status_code}")
                return None
            
            data = response.json()
            
            return {
                "location": f"{data['location']['name']}, {data['location']['country']}",
                "temperature": round(data["current"]["temp_c"]),
                "feels_like": round(data["current"]["feelslike_c"]),
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "wind_speed": round(data["current"]["wind_kph"], 1),
                "pressure": data["current"]["pressure_mb"],
                "clouds": data["current"]["cloud"],
                "visibility": round(data["current"]["vis_km"], 1) if data["current"].get("vis_km") else None,
            }
        except Exception as e:
            print(f"[Weather] WeatherAPI error: {e}")
            return None

    def _fetch_weatherapi_forecast(self, location: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """Fetch weather forecast from WeatherAPI.com."""
        if not self._api_key:
            return None
        try:
            # WeatherAPI.com free tier supports up to 3 days, paid up to 14
            days = min(days, 10)  # Cap at 10 days
            url = "https://api.weatherapi.com/v1/forecast.json"
            params = {"key": self._api_key, "q": location, "days": days, "aqi": "no"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[Weather] WeatherAPI forecast error: {response.status_code}")
                return None
            
            data = response.json()
            
            # Build forecast data
            forecast_days = []
            for day in data.get("forecast", {}).get("forecastday", []):
                forecast_days.append({
                    "date": day["date"],
                    "max_temp": round(day["day"]["maxtemp_c"]),
                    "min_temp": round(day["day"]["mintemp_c"]),
                    "avg_temp": round(day["day"]["avgtemp_c"]),
                    "condition": day["day"]["condition"]["text"],
                    "rain_chance": day["day"].get("daily_chance_of_rain", 0),
                    "humidity": day["day"].get("avghumidity", 0),
                    "max_wind": round(day["day"].get("maxwind_kph", 0), 1),
                })
            
            return {
                "location": f"{data['location']['name']}, {data['location']['country']}",
                "current": {
                    "temperature": round(data["current"]["temp_c"]),
                    "condition": data["current"]["condition"]["text"],
                },
                "forecast": forecast_days,
                "forecast_days": len(forecast_days),
            }
        except Exception as e:
            print(f"[Weather] WeatherAPI forecast error: {e}")
            return None

    def get_forecast(self, location: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """Fetch weather forecast for location."""
        if not self._available or not location or location == "Unknown":
            return None
        
        # Only WeatherAPI.com supports forecasts in our implementation
        if self._provider != "weatherapi":
            print(f"[Weather] Forecast only available with WeatherAPI.com provider")
            return None
        
        print(f"[Weather] Fetching {days}-day forecast for {location}...")
        data = self._fetch_weatherapi_forecast(location, days)
        
        if data:
            print(f"[Weather] ✓ Forecast data retrieved ({data['forecast_days']} days)")
        
        return data

    def format_forecast_for_llm(self, forecast_data: Optional[Dict[str, Any]]) -> str:
        """Format forecast data as context string for LLM."""
        if not forecast_data or not forecast_data.get("forecast"):
            return ""
        
        parts = [f"Weather forecast for {forecast_data['location']}:\n"]
        
        # Current conditions brief
        if forecast_data.get("current"):
            parts.append(f"Current: {forecast_data['current']['temperature']}°C, {forecast_data['current']['condition']}\n")
        
        parts.append("Forecast:")
        for day in forecast_data["forecast"]:
            rain_str = f", {day['rain_chance']}% rain" if day.get('rain_chance', 0) > 0 else ""
            parts.append(f"  {day['date']}: {day['min_temp']}°C - {day['max_temp']}°C, {day['condition']}{rain_str}")
        
        return "\n".join(parts)

    def get_weather(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch current weather for location."""
        if not self._available or not location or location == "Unknown":
            return None
        
        # Check cache first
        cached = self._get_from_cache(location)
        if cached:
            print(f"[Weather] Using cached data for {location}")
            return cached
        
        # Fetch from API
        print(f"[Weather] Fetching weather for {location}...")
        if self._provider == "weatherapi":
            data = self._fetch_weatherapi(location)
        else:  # Default to OpenWeatherMap
            data = self._fetch_openweathermap(location)
        
        if data:
            self._save_to_cache(location, data)
            print(f"[Weather] ✓ Weather data retrieved")
        
        return data

    def format_for_llm(self, weather_data: Optional[Dict[str, Any]]) -> str:
        """Format weather data as context string for LLM."""
        if not weather_data:
            return ""
        
        parts = ["Current weather conditions:\n"]
        parts.append(f"Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)")
        parts.append(f"Condition: {weather_data['condition']}")
        parts.append(f"Humidity: {weather_data['humidity']}%")
        parts.append(f"Wind Speed: {weather_data['wind_speed']} km/h")
        parts.append(f"Pressure: {weather_data['pressure']} hPa")
        if weather_data.get('visibility'):
            parts.append(f"Visibility: {weather_data['visibility']} km")
        
        return "\n".join(parts)

weather_service = WeatherService()

# ============================================================================
# WEATHER TOOL
# ============================================================================

@tool
def get_weather_tool(location: str) -> str:
    """Get current weather conditions for a specific location.
    
    Use this tool when weather information is needed to provide accurate farm advice.
    The location can be a city name, region, or any geographic location.
    
    Args:
        location: The location to get weather for (e.g., "Boston", "New York", "North region")
    
    Returns:
        A formatted string with current weather conditions including temperature, 
        condition, humidity, wind speed, and pressure.
    """
    weather_data = weather_service.get_weather(location)
    if not weather_data:
        return f"Unable to fetch weather data for {location}. Please check the location name or try again later."
    
    return weather_service.format_for_llm(weather_data)

# List of available tools
weather_tools = [get_weather_tool]

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

class WeatherQueryParsed(BaseModel):
    """Structured extraction of weather query details."""
    is_weather_query: bool = Field(description="True if this is primarily a weather-related question")
    location: Optional[str] = Field(default=None, description="City, region, or location mentioned (e.g., 'Hayward, California', 'New York')")
    is_forecast: bool = Field(default=False, description="True if asking about future weather (forecast, next days, tomorrow, etc.)")
    forecast_days: Optional[int] = Field(default=None, description="Number of days for forecast (1 for tomorrow, 7 for week, etc.). Convert months to days (1 month = 30 days).")
    has_farm_context: bool = Field(default=False, description="True if query also mentions crops, livestock, or farming")

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
        
        if first_user_message and len(first_user_message) > 5:
            msg_lower = first_user_message.lower()
            
            # Use LLM to intelligently classify the query and determine next steps
            print(f"[Assessment] Using LLM for smart query classification...")
            
            try:
                classification_response = llm.invoke(f"""Analyze this farmer's query and respond in EXACTLY this format:

Query: "{first_user_message}"

Respond with these 4 lines ONLY (no extra text):
TYPE: [weather/livestock/crops/mixed]
SPECIFIC: [yes/no] (yes if specific crop or animal named, no if generic like "animal" or "crop")
NEEDS_CLARIFICATION: [yes/no] (yes if query is vague or needs more info like location, animal type, etc.)
ITEMS: [comma-separated list of specific crops/animals mentioned, or "none"]

Examples:
- "weather in California" → TYPE: weather, SPECIFIC: no, NEEDS_CLARIFICATION: no, ITEMS: none
- "cattle breeds for my farm" → TYPE: livestock, SPECIFIC: yes, NEEDS_CLARIFICATION: yes, ITEMS: cattle
- "animal recommendation for maize field" → TYPE: mixed, SPECIFIC: no, NEEDS_CLARIFICATION: yes, ITEMS: maize
- "my tomato plants have yellow leaves" → TYPE: crops, SPECIFIC: yes, NEEDS_CLARIFICATION: no, ITEMS: tomato""")
                
                response_text = classification_response.content if hasattr(classification_response, 'content') else str(classification_response)
                print(f"[Assessment] LLM Classification:\n{response_text[:200]}")
                
                # Parse the response
                query_type = "crops"  # default
                is_specific = False
                needs_clarification = True
                detected_items = []
                
                for line in response_text.strip().split('\n'):
                    line = line.strip().upper()
                    if line.startswith('TYPE:'):
                        type_val = line.replace('TYPE:', '').strip().lower()
                        if type_val in ['weather', 'livestock', 'crops', 'mixed']:
                            query_type = type_val
                    elif line.startswith('SPECIFIC:'):
                        is_specific = 'yes' in line.lower()
                    elif line.startswith('NEEDS_CLARIFICATION:'):
                        needs_clarification = 'yes' in line.lower()
                    elif line.startswith('ITEMS:'):
                        items_str = line.replace('ITEMS:', '').strip().lower()
                        if items_str and items_str != 'none':
                            detected_items = [item.strip() for item in items_str.split(',') if item.strip()]
                
                print(f"[Assessment] Parsed: type={query_type}, specific={is_specific}, needs_clarification={needs_clarification}, items={detected_items}")
                
                # Decision logic based on LLM classification
                if query_type == "weather" and not needs_clarification:
                    # Pure weather query - fast-track
                    print(f"[Assessment] ✓ Weather query - fast-tracking")
                    return {
                        "assessment_summary": f"Weather query: {first_user_message}",
                        "current_issues": [first_user_message],
                        "advisory_type": "weather"
                    }
                
                elif is_specific and not needs_clarification:
                    # Specific query with enough info - fast-track
                    print(f"[Assessment] ✓ Specific query detected - fast-tracking to {query_type}")
                    return {
                        "assessment_summary": f"Farmer seeks assistance with: {first_user_message}",
                        "current_issues": [first_user_message],
                        "crops": detected_items if detected_items else None,
                        "advisory_type": query_type
                    }
                
                else:
                    # Needs clarification - don't fast-track, ask questions
                    print(f"[Assessment] Query needs clarification - will ask questions")
                    current_issues = [first_user_message]
                    if detected_items:
                        crops = detected_items
                    # Fall through to ask questions
                    
            except Exception as e:
                print(f"[Assessment] LLM classification error: {e} - falling back to keyword matching")
                # Fallback to simple keyword matching
                msg_lower = first_user_message.lower()
                
                weather_keywords = ["weather", "temperature", "forecast", "rain", "climate"]
                specific_crops = ["paddy", "rice", "wheat", "maize", "corn", "cotton", "soybean", "tomato", "potato"]
                specific_livestock = ["cattle", "cow", "buffalo", "sheep", "goat", "pig", "chicken", "duck", "turkey", "horse"]
                
                if any(kw in msg_lower for kw in weather_keywords):
                    return {
                        "assessment_summary": f"Weather query: {first_user_message}",
                        "current_issues": [first_user_message],
                        "advisory_type": "weather"
                    }
                
                has_specific_crop = any(crop in msg_lower for crop in specific_crops)
                has_specific_livestock = any(animal in msg_lower for animal in specific_livestock)
                
                if has_specific_crop or has_specific_livestock:
                    print(f"[Assessment] ✓ Specific crop/livestock detected (fallback) - fast-tracking")
                    current_issues = [first_user_message]
                    
                    # Determine advisory_type
                    if has_specific_livestock and not has_specific_crop:
                        advisory_type = "livestock"
                    elif has_specific_crop and not has_specific_livestock:
                        advisory_type = "crops"
                    else:
                        advisory_type = "mixed"
                    
                    return {
                        "assessment_summary": f"Farmer seeks assistance with: {first_user_message}",
                        "current_issues": current_issues,
                        "advisory_type": advisory_type
                    }
                else:
                    print(f"[Assessment] Generic question (fallback) - will ask questions")
                    current_issues = [first_user_message]
    
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
    """Hybrid routing: fast-path check, keyword matching, then LLM fallback."""
    
    # FAST PATH: Did assessment_node already determine advisory_type?
    if state.get("advisory_type"):
        print(f"[Routing] ⚡ Using pre-determined type: {state['advisory_type']} (skipping analysis)")
        return {"advisory_type": state["advisory_type"]}
    
    crops = state.get("crops", [])
    issues = state.get("current_issues", [])
    assessment = state.get("assessment_summary", "")
    
    query_text = f"{' '.join(crops)} {' '.join(issues)} {assessment}".lower()
    
    # Weather keywords for pure weather queries
    weather_keywords = [
        "weather", "temperature", "rain", "forecast", "climate", "humidity", 
        "wind", "sunny", "cloudy", "precipitation", "storm", "snow", "fog",
        "temp", "how hot", "how cold", "what's the weather"
    ]
    
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
    
    weather_matches = sum(1 for k in weather_keywords if k in query_text)
    livestock_matches = sum(1 for k in livestock_keywords if k in query_text)
    crop_matches = sum(1 for k in crop_keywords if k in query_text)
    
    print(f"[Routing] Keywords - Weather: {weather_matches}, Livestock: {livestock_matches}, Crops: {crop_matches}")
    
    # Check for pure weather queries FIRST (no farm context)
    if weather_matches > 0 and livestock_matches == 0 and crop_matches == 0:
        print(f"[Routing] → weather (pure weather query)")
        return {"advisory_type": "weather"}
    
    # Farm-related routing
    if livestock_matches > 0 and crop_matches == 0:
        print(f"[Routing] → livestock (keyword)")
        return {"advisory_type": "livestock"}
    if crop_matches > 0 and livestock_matches == 0:
        print(f"[Routing] → crops (keyword)")
        return {"advisory_type": "crops"}
    if livestock_matches > 0 and crop_matches > 0:
        print(f"[Routing] → mixed (keywords)")
        return {"advisory_type": "mixed"}
    
    # LLM fallback (only for ambiguous cases)
    print(f"[Routing] Using LLM fallback...")
    classifier = llm.with_structured_output(QueryClassification)
    prompt = f"""Classify as 'livestock', 'crops', 'mixed', or 'weather':
Crops/Animals: {', '.join(crops) if crops else 'Not specified'}
Issues: {', '.join(issues) if issues else 'None'}
Assessment: {assessment}"""
    
    try:
        result = classifier.invoke(prompt)
        print(f"[Routing] LLM: {result.category}")
        if "weather" in result.category.lower():
            return {"advisory_type": "weather"}
        elif "livestock" in result.category.lower():
            return {"advisory_type": "livestock"}
        elif "mixed" in result.category.lower():
            return {"advisory_type": "mixed"}
    except Exception as e:
        print(f"[Routing] LLM error: {e}")
    
    return {"advisory_type": "crops"}


# ============================================================================
# UNIFIED ADVISORY ENGINE (DRY Principle)
# ============================================================================

def run_advisory_agent(state: FarmState, role_prompt: str, use_rag: bool = False) -> Dict[str, Any]:
    """
    Unified engine for all advisory nodes (Crop, Livestock, Mixed).
    Handles context gathering, RAG retrieval, and the Tool-Calling Loop.
    
    Args:
        state: Current FarmState
        role_prompt: The persona/role for the advisor
        use_rag: Whether to retrieve context from RAG (livestock knowledge base)
    
    Returns:
        Dict with diagnosis, recommendations, and optionally weather_conditions
    """
    print(f"\n[Advisory Agent] Processing with role: {role_prompt[:50]}...")
    
    # 1. Gather Context from State
    location = state.get("location", "Unknown")
    crops = state.get("crops") or []
    issues = state.get("current_issues") or []
    assessment = state.get("assessment_summary", "")
    
    # 2. RAG Retrieval (Centralized)
    rag_context = ""
    if use_rag and RAG_AVAILABLE:
        query_text = f"{', '.join(crops)} {', '.join(issues)} {assessment}"
        try:
            rag.initialize()
            rag_context = rag.get_context_for_query(query_text)
            if rag_context:
                print(f"[Advisory Agent] ✓ RAG context retrieved")
        except Exception as e:
            print(f"[Advisory Agent] RAG error: {e}")

    # 3. Construct Full Prompt
    rag_section = f"RELEVANT KNOWLEDGE BASE:\n{rag_context}" if rag_context else ""
    
    full_prompt = f"""{role_prompt}

Farmer's question: {', '.join(issues) if issues else 'General inquiry'}
Current Context:
- Crops/Livestock: {', '.join(crops) if crops else 'Not specified'}
- Location: {location}

{rag_section}

You have access to a weather tool. Use it if weather conditions are critical for the advice 
(e.g., sowing time, heat stress in animals, pest humidity thresholds).

Provide a concise, friendly response (3-4 sentences) with:
1. Direct answer to their question
2. 2-3 specific, actionable recommendations

Use simple, conversational language. NO markdown formatting, NO asterisks, NO headers.
Write like you're talking to a friend."""

    # 4. Bind Tools
    llm_with_tools = llm.bind_tools(weather_tools) if WEATHER_AVAILABLE else llm

    # 5. Tool Execution Loop (ReAct Pattern)
    weather_data = None
    weather_context = ""
    max_iterations = 3
    final_response = ""

    try:
        for iteration in range(max_iterations):
            current_input = full_prompt + (f"\n\n[Weather Update]: {weather_context}" if weather_context else "")
            response = llm_with_tools.invoke(current_input)
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations - 1:
                print(f"[Advisory Agent] Tool call detected: {len(response.tool_calls)}")
                for tool_call in response.tool_calls:
                    if tool_call.get('name') == 'get_weather_tool':
                        loc = tool_call.get('args', {}).get('location', location)
                        print(f"[Advisory Agent] Executing Weather Tool for: {loc}")
                        
                        # Execute tool
                        tool_result = get_weather_tool.invoke({"location": loc})
                        weather_context = f"Weather Information:\n{tool_result}"
                        
                        # Capture structured data for State
                        try:
                            weather_data = weather_service.get_weather(loc)
                        except:
                            pass
                continue  # Loop back to LLM with new context
            
            # No tool calls - we have our answer
            final_response = response.content if hasattr(response, 'content') else str(response)
            break
        else:
            # Fallback if max iterations reached
            final_response = response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"[Advisory Agent] Error: {e}")
        return {
            "diagnosis": "I'm having trouble generating advice right now. Please try again.",
            "recommendations": ["Consult a local expert"]
        }

    # 6. Parse Recommendations (Simple Heuristic)
    recommendations = []
    for line in final_response.split('\n'):
        line = line.strip()
        if line and any(kw in line.lower() for kw in ['recommend', 'consider', 'try', 'ensure', 'avoid', 'use', 'apply']):
            clean_line = line.replace('**', '').replace('*', '').replace('#', '').strip('- •')
            if clean_line and len(clean_line) > 15:
                recommendations.append(clean_line)

    result = {
        "diagnosis": final_response,
        "recommendations": recommendations[:5] if recommendations else ["Consider consulting a local expert"]
    }
    
    if weather_data:
        result["weather_conditions"] = weather_data
        
    return result


# ============================================================================
# ADVISORY NODES (Declarative - using unified engine)
# ============================================================================

def livestock_advisory_node(state: FarmState):
    """Livestock advisory with RAG and weather tool integration."""
    return run_advisory_agent(
        state,
        role_prompt="You are an expert livestock veterinarian and breed specialist. Provide practical advice on animal health, breed selection, and management.",
        use_rag=True
    )


def crop_advisory_node(state: FarmState):
    """Crop advisory with weather tool integration."""
    return run_advisory_agent(
        state,
        role_prompt="You are an expert agronomist specializing in crop pathology, soil health, and sustainable farming practices.",
        use_rag=False  # RAG currently contains livestock data only
    )


def mixed_advisory_node(state: FarmState):
    """Integrated crop+livestock advisory with RAG and weather tool."""
    return run_advisory_agent(
        state,
        role_prompt="You are an integrated farming systems expert specializing in permaculture, mixed farming, and sustainable agricultural practices.",
        use_rag=True
    )


def weather_advisory_node(state: FarmState):
    """Dedicated weather advisory for pure weather queries."""
    print("\n[Weather Advisory] Processing...")
    print(f"[Weather Advisory] 🌤️ Providing weather information")
    
    location = state.get("location")
    issues = state.get("current_issues") or []
    assessment = state.get("assessment_summary", "")
    user_query = ' '.join(issues) if issues else assessment
    
    # Use LLM to parse weather query if location or forecast info is missing
    forecast_days = None
    
    # Quick check for existing forecast tag in assessment
    forecast_match = re.search(r'\[forecast:(\d+)days\]', assessment)
    if forecast_match:
        forecast_days = int(forecast_match.group(1))
        print(f"[Weather Advisory] Forecast from assessment: {forecast_days} days")
    
    # If location missing or no forecast info, use simple LLM call (fast)
    if not location or location == "Unknown" or not forecast_days:
        print(f"[Weather Advisory] Extracting location from query: {user_query[:50]}...")
        
        try:
            # Simple text-based extraction (much faster than structured output)
            response = llm.invoke(f"""From this query: "{user_query}"
Extract and respond in EXACTLY this format (nothing else):
LOCATION: [city, state/country]
DAYS: [number or 0 for current]

Example response:
LOCATION: Hayward, California
DAYS: 150""")
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"[Weather Advisory] LLM response: {response_text[:100]}")
            
            # Parse the simple response
            for line in response_text.strip().split('\n'):
                if line.startswith('LOCATION:') and (not location or location == "Unknown"):
                    location = line.replace('LOCATION:', '').strip()
                    print(f"[Weather Advisory] Extracted location: {location}")
                elif line.startswith('DAYS:') and not forecast_days:
                    try:
                        days_str = line.replace('DAYS:', '').strip()
                        forecast_days = int(days_str) if days_str.isdigit() else None
                        if forecast_days:
                            print(f"[Weather Advisory] Extracted forecast days: {forecast_days}")
                    except:
                        pass
                
        except Exception as e:
            print(f"[Weather Advisory] LLM extraction error: {e}")
    
    # Fetch weather data
    if location and location != "Unknown":
        try:
            # Use forecast if requested, otherwise current weather
            if forecast_days and forecast_days > 1:
                print(f"[Weather Advisory] Fetching {forecast_days}-day forecast for {location}")
                weather_data = weather_service.get_forecast(location, forecast_days)
                
                if weather_data:
                    formatted_weather = weather_service.format_forecast_for_llm(weather_data)
                    response = f"Here's the {forecast_days}-day weather forecast for {weather_data.get('location', location)}:\n\n{formatted_weather}"
                    
                    return {
                        "diagnosis": response,
                        "recommendations": [],
                        "weather_conditions": weather_data
                    }
                else:
                    # Fallback to current weather if forecast fails
                    print(f"[Weather Advisory] Forecast failed, falling back to current weather")
                    weather_data = weather_service.get_weather(location)
            else:
                weather_data = weather_service.get_weather(location)
            
            if weather_data:
                formatted_weather = weather_service.format_for_llm(weather_data)
                
                # Create a friendly response
                response = f"Here's the current weather for {weather_data.get('location', location)}:\n\n{formatted_weather}"
                
                return {
                    "diagnosis": response,
                    "recommendations": [],
                    "weather_conditions": weather_data
                }
            else:
                return {
                    "diagnosis": f"I couldn't fetch weather data for '{location}'. Please check the location name and try again.",
                    "recommendations": ["Make sure the location name is spelled correctly", "Try using a city name or region"]
                }
        except Exception as e:
            print(f"[Weather Advisory] Error: {e}")
            return {
                "diagnosis": f"Sorry, I encountered an error while fetching weather data. Please try again later.",
                "recommendations": []
            }
    else:
        return {
            "diagnosis": "I need a location to provide weather information. Please tell me which city or region you'd like to know about.",
            "recommendations": ["Provide a city name (e.g., 'Boston', 'New York')", "Or provide a region (e.g., 'North region', 'Central region')"]
        }


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
    if advisory_type == "weather":
        return "weather_advisory_node"
    elif advisory_type == "livestock":
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
builder.add_node("weather_advisory_node", weather_advisory_node)
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
        "weather_advisory_node": "weather_advisory_node",
        "livestock_advisory_node": "livestock_advisory_node",
        "crop_advisory_node": "crop_advisory_node",
        "mixed_advisory_node": "mixed_advisory_node"
    }
)

builder.add_edge("weather_advisory_node", END)
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
print("  ✓ Separate advisory nodes (weather/livestock/crops/mixed)")
print("  ✓ Pure weather queries support (dedicated weather node)")
print("  ✓ Livestock RAG integration (breed recommendations)")
print("  ✓ Weather tool integration (LLM decides when to fetch weather)")
print("  ✓ Enhanced state management (assessment_summary + advisory_type)")
print("="*60)


try:
    from IPython.display import Image, display
    
    graph_repr = graph.get_graph()
    png_bytes = graph_repr.draw_mermaid_png(output_file_path="farm_advisory_graph.png")
    display(Image(png_bytes))
    print("✓ Graph visualization displayed and saved as 'farm_advisory_graph.png'")
except ImportError:
    print("⚠ IPython not available. Install with: pip install ipython")
except Exception as e:
    print(f"⚠ Could not display graph: {e}")
