from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .routes import analysis, health
from .config import settings
from app.observability import configure_langsmith
from .db import init_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application started")
    tracing_enabled = configure_langsmith(settings)
    logger.info("LangSmith tracing active: %s", tracing_enabled)
    if settings.DATABASE_ENABLED:
        try:
            init_db()
        except Exception as exc:
            logger.warning("Database initialization failed, continuing without DB persistence: %s", exc)
    logger.info("Agent will be initialized on first request")
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="Orbital Witness API",
    description="Satellite Intelligence & Change Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/")
async def root():
    return {
        "message": "Orbital Witness API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "observability": {
            "langsmith_tracing_enabled": settings.LANGSMITH_TRACING_ENABLED,
            "graph_metrics_enabled": settings.ENABLE_GRAPH_METRICS,
        },
        "database": {
            "enabled": settings.DATABASE_ENABLED,
        },
    }
