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
    TOP_K_RETRIEVAL: int = 70
    # Final chunk count passed to the LLM.  Complex multi-part questions use
    # TOP_K_RERANK_COMPLEX to ensure context for all sub-questions survives
    # after reranking (roughly 2-3 chunks per sub-question).
    TOP_K_RERANK: int = 7
    TOP_K_RERANK_COMPLEX: int = 20
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: Literal["cpu", "cuda"] = "cpu"
    USE_HYBRID_SEARCH: bool = True
    # Default hybrid alpha — 70% dense, 30% keyword.
    HYBRID_ALPHA: float = 0.7
    # Alpha for legal/compliance queries: more keyword weight because statute
    # numbers, deadlines, and article references are exact-match signals that
    # dense embeddings routinely miss.
    HYBRID_ALPHA_LEGAL: float = 0.55
    # Relative floor for hybrid (RRF) results: drop any chunk whose RRF score
    # is below this fraction of the top-ranked chunk's score.  Keeps the
    # candidate pool clean before the cross-encoder without requiring a
    # hand-tuned absolute threshold (RRF scores are not on a 0-1 scale).
    HYBRID_RRF_MIN_RATIO: float = 0.2
    # Keyword fallback: when the top-1 dense score is below this threshold the
    # query likely contains exact-match signals (codes, numbers, rare terms)
    # that BM25 handles better than dense embeddings.  Alpha is capped at
    # HYBRID_ALPHA_FALLBACK to give BM25 equal or higher weight.
    HYBRID_DENSE_FALLBACK_THRESHOLD: float = 0.50
    HYBRID_ALPHA_FALLBACK: float = 0.50
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3
    # Reranker relevance gate — simple queries use the default threshold.
    # Complex decomposed queries use a lower threshold because a chunk
    # covering only one of four sub-questions will legitimately score lower
    # against the combined query but is still required context.
    RERANKER_SCORE_THRESHOLD: float = 0.35
    RERANKER_SCORE_THRESHOLD_COMPLEX: float = 0.20

    # --- Query Rewriting ---
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_REWRITE_MIN_TURNS: int = 2

    # --- Multi-Query Retrieval ---
    # When MULTI_QUERY_ENABLED is True, the query expander runs in one of two
    # modes depending on the question:
    #   paraphrase mode — generates MULTI_QUERY_COUNT synonymous phrasings
    #                     (for simple, single-intent queries)
    #   decompose mode  — generates atomic sub-questions, one per aspect
    #                     (for complex multi-part questions)
    # Set MULTI_QUERY_ENABLED=False to disable all expansion (single-query).
    MULTI_QUERY_ENABLED: bool = True
    MULTI_QUERY_COUNT: int = 2  # paraphrase variants for simple queries

    # --- Query Decomposition ---
    # Decomposition is activated when a query has multi-part signals (numbered
    # lists, "including:", multiple "and"s, etc.) AND is at least
    # DECOMPOSE_MIN_WORDS long.  Requires MULTI_QUERY_ENABLED=True.
    DECOMPOSE_ENABLED: bool = True
    DECOMPOSE_MIN_WORDS: int = 8

    # --- Cache (Redis) ---
    USE_CACHE: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600

    # --- Conversation Persistence (PostgreSQL) ---
    # Set to enable persistent conversations across restarts.
    # Format: postgresql+asyncpg://user:password@host:5432/dbname
    # Leave empty to use in-memory storage (default).
    DATABASE_URL: Optional[str] = None

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
