"""
API v1 router — aggregates all v1 endpoints.
"""
from fastapi import APIRouter

from backend.api.v1.endpoints import chat, documents, health

router = APIRouter()

# Health routes at root level (for K8s/load balancers)
router.include_router(health.router, tags=["Health"])

# API routes
router.include_router(documents.router, tags=["Documents"])
router.include_router(chat.router, tags=["Chat"])
