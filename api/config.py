import os
from pydantic_settings import BaseSettings
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Orbital Witness API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
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
    
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300
    
    class Config:
        case_sensitive = True

settings = Settings()