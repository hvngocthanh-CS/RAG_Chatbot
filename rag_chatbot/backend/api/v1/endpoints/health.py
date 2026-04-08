"""
Health check endpoints.
"""
from fastapi import APIRouter
from datetime import datetime, timezone
from typing import Dict, Any

from backend.config import settings
from backend.services import get_service, get_service_status

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
    }


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe — checks critical services."""
    status = await get_service_status()

    all_healthy = all(
        s.get("status") == "healthy" for s in status.values()
    )

    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with all service statuses."""
    service_status = await get_service_status()

    result = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENVIRONMENT,
        "services": service_status,
    }

    unhealthy = [
        name for name, s in service_status.items()
        if s.get("status") != "healthy"
    ]

    if unhealthy:
        result["status"] = "degraded" if len(unhealthy) < len(service_status) else "unhealthy"

    return result
