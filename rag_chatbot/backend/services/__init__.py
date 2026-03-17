"""
Services module initialization.
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
    
    # Initialize vector store
    if settings.VECTOR_DB_TYPE == "chroma":
        from backend.services.vector_store import ChromaVectorStore
        _services["vector_store"] = ChromaVectorStore()
    else:
        from backend.services.vector_store import QdrantVectorStore
        _services["vector_store"] = QdrantVectorStore()
    
    await _services["vector_store"].initialize()
    
    # Initialize embedding service
    from backend.services.embeddings import EmbeddingService
    _services["embedding"] = EmbeddingService()
    await _services["embedding"].initialize()
    
    # Initialize LLM service
    from backend.services.llm import LLMService
    _services["llm"] = LLMService()
    
    # Initialize cache if enabled
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
            logger.info(f"Service {name} shutdown")
        elif hasattr(service, "close"):
            await service.close()
            logger.info(f"Service {name} closed")


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
