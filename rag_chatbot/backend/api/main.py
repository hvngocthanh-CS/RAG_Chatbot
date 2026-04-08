"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1.endpoints import chat, documents, health
from backend.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    from backend.services import initialize_services
    await initialize_services()

    yield

    logger.info("Shutting down application...")
    from backend.services import cleanup_services
    await cleanup_services()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-ready RAG Chatbot for Enterprise Document Q&A",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health routes at root level (for K8s/load balancers)
    app.include_router(health.router, tags=["Health"])
    # API routes with prefix
    app.include_router(documents.router, prefix=settings.API_PREFIX, tags=["Documents"])
    app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["Chat"])

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": f"{settings.API_PREFIX}/docs",
        }

    return app


app = create_app()
