"""
Services module — global service registry.

All services are initialized once at startup and shared across requests.
Use get_service(name) to retrieve a service instance.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global service instances
_services: Dict[str, Any] = {}


async def initialize_services():
    """Initialize all services on application startup."""
    from backend.config import settings

    logger.info("Initializing services...")

    # Vector store (Qdrant)
    from backend.services.vectorstore import VectorStoreService
    _services["vector_store"] = VectorStoreService()
    await _services["vector_store"].initialize()

    # Embedding service
    from backend.services.embedding import EmbeddingService
    _services["embedding"] = EmbeddingService()
    await _services["embedding"].initialize()

    # LLM service
    from backend.services.llm import LLMService
    _services["llm"] = LLMService()
    await _services["llm"].initialize()

    # Retrieval service (depends on embedding + vector_store)
    from backend.services.retrieval import RetrievalService
    _services["retrieval"] = RetrievalService()

    # Conversation manager (singleton)
    from backend.services.conversation import ConversationManager
    _services["conversation"] = ConversationManager()

    # Cache (optional)
    if settings.USE_CACHE:
        from backend.services.cache import CacheService
        _services["cache"] = CacheService()
        await _services["cache"].initialize()

    logger.info("All services initialized successfully")


async def cleanup_services():
    """Cleanup services on application shutdown."""
    logger.info("Cleaning up services...")

    for name, service in _services.items():
        if hasattr(service, "shutdown"):
            await service.shutdown()
            logger.info("Service %s shutdown", name)
        elif hasattr(service, "close"):
            await service.close()
            logger.info("Service %s closed", name)


async def get_service_status() -> Dict[str, Dict[str, str]]:
    """Get status of all services."""
    status = {}

    for name, service in _services.items():
        try:
            if hasattr(service, "health_check"):
                is_healthy = await service.health_check()
                status[name] = {
                    "status": "healthy" if is_healthy else "unhealthy"
                }
            else:
                status[name] = {"status": "healthy"}
        except Exception as e:
            status[name] = {"status": "unhealthy", "error": str(e)}

    return status


def get_service(name: str):
    """Get a service instance by name."""
    return _services.get(name)
