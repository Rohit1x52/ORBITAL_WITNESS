import logging

from .base import Base
from .session import engine
from . import models  # noqa: F401

logger = logging.getLogger(__name__)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
