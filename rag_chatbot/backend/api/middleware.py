"""
HTTP middlewares for the FastAPI app.
"""
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Attach a correlation ID to every request.

    - If the client sent `X-Request-ID`, we reuse it (useful when an upstream
      gateway already generates IDs).
    - Otherwise we generate a fresh UUID.
    - The ID is stored in a contextvar so all log lines emitted during the
      request automatically include it, and is echoed back in the response
      header so clients can quote it when reporting issues.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
