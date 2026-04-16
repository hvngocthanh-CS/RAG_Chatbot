"""
Configuration for the RAG Chatbot.
All settings are loaded from environment variables via Pydantic BaseSettings.
"""
from typing import Optional, Literal, List
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
import json as _json


class Settings(BaseSettings):
    """Application settings. Override any value via environment variables or .env file."""

    # --- Application ---
    APP_NAME: str = "RAG Chatbot"
    APP_VERSION: str = "2.0.0"
    APP_ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # --- LLM (Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "llama3.1:8b"
    # Judge model for RAGAS — can reuse OLLAMA_MODEL (same 8B is strong
    # enough for structured JSON). Leave empty to reuse OLLAMA_MODEL.
    RAGAS_JUDGE_MODEL: str = ""
    RAGAS_TIMEOUT_SECONDS: int = 600
    RAGAS_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024
    LLM_TOP_P: float = 0.95
    LLM_PRESENCE_PENALTY: float = 0.2
    LLM_FREQUENCY_PENALTY: float = 0.3

    # --- Resilience ---
    REQUEST_TIMEOUT: int = 300
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 30
    RETRY_ENABLED: bool = True
    RETRY_MAX_ATTEMPTS: int = 2
    RETRY_INITIAL_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 10.0
    RETRY_EXPONENTIAL_BASE: float = 2.0
    MAX_CONCURRENT_REQUESTS: int = 50
    HEALTH_CHECK_TIMEOUT: int = 10

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = "cpu"

    # --- Vector Database (Qdrant) ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "documents"

    # --- Document Processing ---
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md"]

    # Section-Aware Chunker
    SECTION_MAX_CHUNK_TOKENS: int = 600
    SECTION_MIN_CHUNK_TOKENS: int = 80
    SECTION_OVERLAP_SENTENCES: int = 2
    SECTION_SEMANTIC_LOOK_BACK: int = 3
    SECTION_SEMANTIC_MIN_SCORE: float = 0.15

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = 40
    TOP_K_RERANK: int = 5
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: Literal["cpu", "cuda"] = "cpu"
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.7
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3

    # --- Query Rewriting ---
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_REWRITE_MIN_TURNS: int = 2

    # --- Cache (Redis) ---
    USE_CACHE: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600

    # --- Storage ---
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"

    # --- HuggingFace Model Cache ---
    HF_HOME: str = "./models"
    SENTENCE_TRANSFORMERS_HOME: str = "./models"
    TRANSFORMERS_CACHE: str = "./models"
    HF_HUB_DOWNLOAD_TIMEOUT: int = 180
    HF_HUB_ETAG_TIMEOUT: int = 180

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    # "json" for production (Docker/K8s log aggregators); "console" for local dev.
    LOG_FORMAT: str = "console"

    # --- Security ---
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True

    # --- Validators ---
    @field_validator("SUPPORTED_EXTENSIONS", "CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_list(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return _json.loads(v)
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
