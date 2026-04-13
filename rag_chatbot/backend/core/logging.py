"""
Structured logging with per-request correlation IDs.

- `request_id_var` is a contextvar set by the API middleware on every request.
  Any log emitted while handling that request will automatically carry the ID,
  which makes traces easy to follow across services in production (Docker/K8s).
- `setup_logging()` is called once at app startup and picks the format based
  on `settings.LOG_FORMAT` ("json" for production, "console" for local dev).
"""
import logging
import sys
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# Set by CorrelationIdMiddleware for the lifetime of one HTTP request.
# Background tasks (ingestion scripts, etc.) leave this as "-".
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — friendly for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """
    Configure the root logger.

    Args:
        level: standard logging level name ("DEBUG", "INFO", ...).
        fmt: "json" for structured output (recommended in Docker), or
             "console" for human-readable output (recommended locally).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove handlers that may have been added by basicConfig or libraries,
    # so we don't end up with duplicate log lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s - %(message)s"
        ))

    root.addHandler(handler)


def get_request_id() -> Optional[str]:
    """Return the current request_id, or None if not in a request context."""
    rid = request_id_var.get()
    return rid if rid != "-" else None
