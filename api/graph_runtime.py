from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.agent import SatelliteAgentConfig, create_satellite_agent
from app.graph_workflow import GraphOrchestrator
from .config import settings


@lru_cache(maxsize=1)
def get_satellite_agent():
    return create_satellite_agent()


@lru_cache(maxsize=1)
def get_graph_orchestrator() -> GraphOrchestrator:
    config = SatelliteAgentConfig(
        confidence_threshold=float(getattr(settings, "CONFIDENCE_THRESHOLD", 0.6)),
        workflow_mode=settings.WORKFLOW_MODE,
        checkpoint_backend=settings.CHECKPOINT_BACKEND,
    )
    return GraphOrchestrator(config)


def get_workflow_runner():
    if settings.WORKFLOW_MODE.lower() == "linear":
        return get_satellite_agent()
    return get_graph_orchestrator()
