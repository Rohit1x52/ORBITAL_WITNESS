import os
from pydantic_settings import BaseSettings
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Orbital Witness API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    WORKFLOW_MODE: str = os.getenv("WORKFLOW_MODE", "graph")
    CHECKPOINT_BACKEND: str = os.getenv("WORKFLOW_CHECKPOINT_BACKEND", "memory")
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000"
    ]
    
    NASA_API_KEY: str = os.getenv("NASA_API_KEY", "DEMO_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    LANGSMITH_TRACING_ENABLED: bool = os.getenv("LANGSMITH_TRACING_ENABLED", "false").lower() == "true"
    LANGSMITH_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "orbital-witness")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    ENABLE_GRAPH_METRICS: bool = os.getenv("ENABLE_GRAPH_METRICS", "true").lower() == "true"

    DATABASE_ENABLED: bool = os.getenv("DATABASE_ENABLED", "true").lower() == "true"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/orbital_witness",
    )
    
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300
    
    class Config:
        case_sensitive = True

settings = Settings()