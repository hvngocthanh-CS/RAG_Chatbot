"""Centralized logging configuration for the backend application."""
import logging

from backend.config import settings


def configure_logging() -> None:
    """Configure application logging once during startup."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
