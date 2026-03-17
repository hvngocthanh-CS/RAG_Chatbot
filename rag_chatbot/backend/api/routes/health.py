"""
Health check and metrics endpoints.

Production-ready endpoints:
- Basic health check
- Detailed health check with service status
- Prometheus metrics export
- Readiness and liveness probes
"""
from fastapi import APIRouter, Response
from datetime import datetime
from typing import Dict, Any

from backend.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Used by load balancers and orchestrators for simple health verification.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENVIRONMENT
    }


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if the application is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe() -> Dict[str, Any]:
    """
    Kubernetes readiness probe.
    
    Returns 200 if the application is ready to accept traffic.
    Checks critical dependencies.
    """
    from backend.services.llm import LLMService
    
    ready = True
    checks = {}
    
    # Check LLM service
    try:
        llm_service = LLMService()
        health = await llm_service.health_check()
        checks["llm"] = health.get("status", "unknown")
        if checks["llm"] != "healthy":
            ready = False
    except Exception as e:
        checks["llm"] = "error"
        ready = False
    
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with all service statuses.
    
    Returns comprehensive health information including:
    - LLM service status (vLLM or Ollama)
    - Circuit breaker states
    - Vector database connectivity
    - Redis cache status
    """
    from backend.services.llm import LLMService
    from backend.services.observability import health_checker
    
    result = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENVIRONMENT,
        "services": {}
    }
    
    # Check LLM service
    try:
        llm_service = LLMService()
        llm_health = await llm_service.health_check()
        result["services"]["llm"] = llm_health
        
        if llm_health.get("status") != "healthy":
            result["status"] = "degraded"
    except Exception as e:
        result["services"]["llm"] = {"status": "error", "error": str(e)}
        result["status"] = "degraded"
    
    # Check vector database
    try:
        from backend.services import get_service
        vector_service = get_service("vector_store")
        if vector_service:
            is_healthy = await vector_service.health_check()
            result["services"]["vector_db"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "type": settings.VECTOR_DB_TYPE
            }
        else:
            result["services"]["vector_db"] = {"status": "not_initialized"}
            result["status"] = "degraded"
    except Exception as e:
        result["services"]["vector_db"] = {"status": "error", "error": str(e)}
        result["status"] = "degraded"
    
    # Check Redis cache
    if settings.USE_CACHE:
        try:
            from backend.services.cache import CacheService
            cache_service = CacheService()
            await cache_service.initialize()
            result["services"]["cache"] = {
                "status": "healthy",
                "type": "redis"
            }
        except Exception as e:
            result["services"]["cache"] = {"status": "error", "error": str(e)}
            result["status"] = "degraded"
    
    # If all services are in error state, mark as unhealthy
    all_error = all(
        s.get("status") in ["error", "unhealthy"]
        for s in result["services"].values()
    )
    if all_error and result["services"]:
        result["status"] = "unhealthy"
    
    return result


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Exports metrics in Prometheus text format.
    """
    from backend.services.observability import metrics
    from backend.services.llm import LLMService
    
    # Add LLM metrics
    try:
        llm_service = LLMService()
        llm_metrics = llm_service.get_metrics()
        
        # Update global metrics with LLM data
        metrics._metrics["llm_requests_total"]["success"] = \
            llm_metrics["llm_metrics"]["successful_requests"]
        metrics._metrics["llm_requests_total"]["failure"] = \
            llm_metrics["llm_metrics"]["failed_requests"]
        metrics._metrics["llm_tokens_total"]["input"] = \
            llm_metrics["llm_metrics"]["total_tokens_input"]
        metrics._metrics["llm_tokens_total"]["output"] = \
            llm_metrics["llm_metrics"]["total_tokens_output"]
        metrics._metrics["llm_fallback_total"] = \
            llm_metrics["llm_metrics"]["fallback_count"]
    except:
        pass
    
    prometheus_output = metrics.to_prometheus_format()
    
    return Response(
        content=prometheus_output,
        media_type="text/plain; charset=utf-8"
    )


@router.get("/metrics/json")
async def json_metrics() -> Dict[str, Any]:
    """
    JSON metrics endpoint.
    
    Returns metrics in JSON format for easier debugging.
    """
    from backend.services.observability import metrics
    from backend.services.llm import LLMService
    
    result = metrics.get_metrics()
    
    # Add LLM-specific metrics
    try:
        llm_service = LLMService()
        result["llm_detailed"] = llm_service.get_metrics()
    except:
        result["llm_detailed"] = {"error": "Unable to get LLM metrics"}
    
    return result
