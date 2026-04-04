import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


def configure_langsmith(settings: Any) -> bool:
    enabled = bool(getattr(settings, "LANGSMITH_TRACING_ENABLED", False))
    api_key = getattr(settings, "LANGSMITH_API_KEY", "")

    if not enabled:
        logger.info("LangSmith tracing disabled by config")
        return False

    if not api_key:
        logger.warning("LangSmith tracing enabled but LANGCHAIN_API_KEY is missing; tracing will remain disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = getattr(settings, "LANGSMITH_PROJECT", "orbital-witness")
    os.environ["LANGCHAIN_ENDPOINT"] = getattr(
        settings,
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    logger.info(
        "LangSmith tracing enabled for project '%s'",
        os.environ["LANGCHAIN_PROJECT"],
    )
    return True
