"""
Cache Service.
Redis-based caching for query responses.
"""
import json
import hashlib
import logging
from typing import Optional, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Redis cache service for caching query responses.
    
    Features:
    - Query response caching
    - Configurable TTL
    - Hash-based cache keys
    """
    
    def __init__(self):
        self.client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis client."""
        if not settings.USE_CACHE:
            logger.info("Cache disabled by configuration")
            return
        
        if self._initialized:
            return
        
        try:
            import redis.asyncio as redis
            
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            
            # Test connection
            await self.client.ping()
            
            self._initialized = True
            logger.info("Redis cache initialized")
            
        except ImportError:
            logger.warning("redis package not installed. Caching disabled.")
            self.client = None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.client = None
    
    def _generate_cache_key(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a unique cache key for a query.
        
        Uses MD5 hash of question and filters.
        """
        key_data = {
            "question": question.lower().strip(),
            "filters": filters or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"rag:query:{key_hash}"
    
    async def get_cached_response(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query.
        
        Returns None if not found or cache is disabled.
        """
        if not self._initialized or not self.client:
            return None
        
        try:
            key = self._generate_cache_key(question, filters)
            cached = await self.client.get(key)
            
            if cached:
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(cached)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def cache_response(
        self,
        question: str,
        filters: Optional[Dict[str, Any]],
        response: Dict[str, Any]
    ):
        """
        Cache a query response.
        """
        if not self._initialized or not self.client:
            return
        
        try:
            key = self._generate_cache_key(question, filters)
            value = json.dumps(response)
            
            await self.client.setex(
                key,
                settings.CACHE_TTL,
                value
            )
            
            logger.debug(f"Cached response for key: {key}")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate_document_cache(self, document_id: str):
        """
        Invalidate all cache entries related to a document.
        
        Note: This is a simplified implementation. In production,
        you might want to track which queries touched which documents.
        """
        if not self._initialized or not self.client:
            return
        
        try:
            # For simplicity, clear all query cache
            # In production, implement more granular invalidation
            pattern = "rag:query:*"
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            
            if deleted > 0:
                logger.info(f"Invalidated {deleted} cache entries")
                
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    async def clear_all(self):
        """Clear all cached data."""
        if not self._initialized or not self.client:
            return
        
        try:
            await self.client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        if not settings.USE_CACHE:
            return True
        
        if not self._initialized or not self.client:
            return False
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
