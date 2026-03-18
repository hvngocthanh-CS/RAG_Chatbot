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
        # Ưu tiên giá trị từ PROVIDER và VLLM_SERVER_URL nếu được set trong .env
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
    # LLM Settings - vLLM Integration
    # ===========================================
    LLM_PROVIDER: Literal["vllm", "ollama"] = "ollama"

    # OpenAI Settings removed
    
    # Ollama Settings (local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "llama3.2"  # or qwen2.5, mistral, phi3, etc.
    
    # vLLM Settings (Primary)
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
    VLLM_API_KEY: str = "not-needed"  # vLLM often doesn't require API key
    
    # vLLM Server Configuration (for deployment)
    VLLM_TENSOR_PARALLEL_SIZE: int = 1
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.9
    VLLM_MAX_MODEL_LEN: int = 4096
    VLLM_QUANTIZATION: Optional[Literal["awq", "gptq", "squeezellm", "fp8"]] = None
    VLLM_DTYPE: Literal["auto", "half", "float16", "bfloat16", "float32"] = "auto"
    VLLM_TRUST_REMOTE_CODE: bool = True
    
    # Generation Settings
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.95
    LLM_TOP_K: int = 40
    LLM_REPETITION_PENALTY: float = 1.1
    LLM_PRESENCE_PENALTY: float = 0.0
    LLM_FREQUENCY_PENALTY: float = 0.0
    
    # ===========================================
    # Resilience Settings (Circuit Breaker)
    # ===========================================
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 30  # seconds
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: bool = True
    
    # Retry Settings
    RETRY_ENABLED: bool = True
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_INITIAL_DELAY: float = 1.0  # seconds
    RETRY_MAX_DELAY: float = 10.0  # seconds
    RETRY_EXPONENTIAL_BASE: float = 2.0

    # Fallback Settings removed
    
    # ===========================================
    # Embedding Settings
    # ===========================================
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_RETRIES: int = 3
    
    # ===========================================
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
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md"]
    
    # ===========================================
    # Retrieval Settings
    # ===========================================
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.5  # Balance between vector and keyword search
    # Minimum cosine similarity score (0-1) to include a chunk in results.
    # Chunks below this threshold are discarded before reranking.
    # Raise to 0.4+ for stricter answers; lower to 0.2 if recall is too low.
    RETRIEVAL_SCORE_THRESHOLD: float = 0.30

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
    
    @field_validator('VLLM_QUANTIZATION', mode='before')
    @classmethod
    def parse_vllm_quantization(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
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
