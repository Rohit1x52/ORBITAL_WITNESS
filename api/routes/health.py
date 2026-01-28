from fastapi import APIRouter, status
from datetime import datetime
import psutil
import os

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Orbital Witness API"
    }

@router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Orbital Witness API",
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        },
        "environment": {
            "nasa_api_configured": bool(os.getenv("NASA_API_KEY")),
            "groq_api_configured": bool(os.getenv("GROQ_API_KEY"))
        }
    }