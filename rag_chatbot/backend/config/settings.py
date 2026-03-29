"""
Configuration settings for RAG Enterprise Chatbot.
Uses Pydantic Settings for environment variable management.

Production-ready configuration with:
- vLLM integration
- Observability settings
- Rate limiting
- Circuit breaker configs
"""
from typing import Optional, Literal, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    def __init__(self, **values):
        # Support PROVIDER and VLLM_SERVER_URL aliases from .env
        provider = values.get('PROVIDER')
        vllm_server_url = values.get('VLLM_SERVER_URL')
        if provider:
            values['LLM_PROVIDER'] = provider
        if vllm_server_url:
            values['VLLM_BASE_URL'] = vllm_server_url
        super().__init__(**values)
    
    # ===========================================
    # Custom Provider Settings (for .env compatibility)
    # ===========================================
    PROVIDER: Optional[str] = None  # 'vllm' or 'openai' (for .env compatibility)
    VLLM_SERVER_URL: Optional[str] = None  # for .env compatibility
    # ===========================================
    # Application Settings
    # ===========================================
    APP_NAME: str = "RAG Enterprise Chatbot"
    APP_VERSION: str = "1.0.0"
    APP_ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    
    # ===========================================
    # API Settings
    # ===========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    API_WORKERS: int = 4  # Number of uvicorn workers
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests per window
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Request Settings
    REQUEST_TIMEOUT: int = 120  # seconds
    MAX_CONCURRENT_REQUESTS: int = 50
    
    # ===========================================
    # LLM Settings
    # ===========================================
    LLM_PROVIDER: Literal["vllm", "ollama"] = "ollama"
    
    # Ollama Settings (local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "phi3"  # or llama3.2, qwen2.5, mistral, etc.
    
    # vLLM Settings (for Docker deployment)
    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
    VLLM_API_KEY: str = "not-needed"
    
    # Generation Settings
    LLM_TEMPERATURE: float = 0.0  # Deterministic - same question = same answer
    LLM_MAX_TOKENS: int = 2048  # Sufficient for complete multi-part answers
    LLM_TOP_P: float = 0.95
    LLM_PRESENCE_PENALTY: float = 0.2  # Reduce repetition
    LLM_FREQUENCY_PENALTY: float = 0.3  # Reduce repetition
    
    # ===========================================
    # Resilience Settings (Circuit Breaker)
    # ===========================================
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 30  # seconds
    
    # Retry Settings
    RETRY_ENABLED: bool = True
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_INITIAL_DELAY: float = 1.0  # seconds
    RETRY_MAX_DELAY: float = 10.0  # seconds
    RETRY_EXPONENTIAL_BASE: float = 2.0
    
    # ===========================================
    # Embedding Settings (HuggingFace)
    # ===========================================
    EMBEDDING_PROVIDER: Literal["huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = "cpu"
    
    # ==========================================
    # Vector Database Settings
    # ===========================================
    VECTOR_DB_TYPE: Literal["qdrant", "chroma"] = "chroma"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_PREFER_GRPC: bool = True
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_NAME: str = "documents"
    
    # ===========================================
    # Document Processing
    # ===========================================
    CHUNK_SIZE: int = 500  # tokens per chunk (balanced size for context)
    CHUNK_OVERLAP: int = 100  # tokens (more overlap = better continuity)
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md"]
    
    # Chunking Strategy Selection
    CHUNKING_STRATEGY: Literal["token", "semantic"] = "semantic"  # "token" or "semantic"
    
    # Semantic Chunking Settings (when CHUNKING_STRATEGY = "semantic")
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.4  # Lower = more breaks, better precision (0.3-0.7 recommended)
    SEMANTIC_MAX_CHUNK_SIZE: int = 600  # Maximum tokens per semantic chunk (smaller = more precise)
    SEMANTIC_MIN_CHUNK_SIZE: int = 80  # Minimum tokens per semantic chunk
    SEMANTIC_OVERLAP_SENTENCES: int = 3  # Number of sentences to overlap between chunks (more context)
    
    # ===========================================
    # Retrieval Settings
    # ===========================================
    TOP_K_RETRIEVAL: int = 40  # More candidates for stable retrieval
    TOP_K_RERANK: int = 6  # Final results returned after reranking (more context)
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # CPU OPTIMIZED: Force reranker to run on CPU (lightweight cross-encoder)
    RERANKER_DEVICE: Literal["cpu", "cuda"] = "cpu"
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.7  # Favor semantic search (0.7) over keyword (0.3)
    # Minimum cosine similarity score (0-1) to include a chunk in results.
    # Set to 0.3 to filter out irrelevant chunks early
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3  # Filter low-quality matches

    # ===========================================
    # Conversational Query Rewriting
    # ===========================================
    # Enable LLM-based rewriting of follow-up questions into standalone
    # queries before vector retrieval.  Costs one extra LLM call per
    # multi-turn request; disable if latency is critical.
    QUERY_REWRITE_ENABLED: bool = True
    # Only rewrite when conversation has at least this many prior turns
    # (i.e. at least 1 prior exchange = 2 messages: user + assistant).
    QUERY_REWRITE_MIN_TURNS: int = 2
    
    # ===========================================
    # Redis Cache
    # ===========================================
    USE_CACHE: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 10
    CACHE_TTL: int = 3600  # seconds
    CACHE_KEY_PREFIX: str = "rag:"
    
    # ===========================================
    # Storage
    # ===========================================
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"
    
    # ===========================================
    # Observability Settings
    # ===========================================
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    
    # Metrics (Prometheus)
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"
    
    # Tracing (OpenTelemetry)
    TRACING_ENABLED: bool = False
    OTLP_ENDPOINT: Optional[str] = None
    SERVICE_NAME: str = "rag-chatbot"
    
    # Health Check
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_TIMEOUT: int = 10  # seconds
    
    # ===========================================
    # Security Settings
    # ===========================================
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = []  # List of valid API keys (empty = no auth)
    
    @field_validator('SUPPORTED_EXTENSIONS', mode='before')
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            v = v.strip()
            # Handle JSON array format
            if v.startswith('[') and v.endswith(']'):
                import json
                return json.loads(v)
            # Handle comma-separated format
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        return v
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            # Handle JSON array format
            if v.startswith('[') and v.endswith(']'):
                import json
                return json.loads(v)
            # Handle comma-separated format
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @field_validator('API_KEYS', mode='before')
    @classmethod
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            v = v.strip()
            # Handle JSON array format
            if v.startswith('[') and v.endswith(']'):
                import json
                return json.loads(v)
            # Handle comma-separated format
            return [key.strip() for key in v.split(',') if key.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
