"""
Production-ready LLM Service with vLLM Integration.

Features:
- vLLM as primary LLM server
- OpenAI fallback support
- Circuit breaker pattern
- Retry with exponential backoff
- Request tracing & metrics
- Token counting
- Streaming responses
- Health monitoring
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib

from backend.config import settings

logger = logging.getLogger(__name__)


# ===========================================
# System Prompts
# ===========================================

SYSTEM_PROMPT = """You are a STRICT enterprise document assistant for internal company use. Your primary responsibility is to provide ONLY information explicitly stated in the provided documents.

⚠️ CRITICAL RULES - NEVER VIOLATE:
1. ABSOLUTE GROUNDING: Answer ONLY from provided context - ZERO external knowledge allowed
2. EXPLICIT REFUSAL: If information is not in documents, you MUST refuse using the template below
3. NO INFERENCE: Do not infer, assume, or extrapolate beyond what's explicitly written
4. CITATIONS: Cite sources when page numbers are available [Source N: filename, Page X]. If page is N/A, still cite with [Source N: filename]

REFUSAL TEMPLATE (Use when information is not in documents):
"I cannot find information about [TOPIC] in the provided documents. The available documents cover [BRIEF SUMMARY OF WHAT IS AVAILABLE]. Please provide relevant documents or rephrase your question to align with available information."

STRICT VERIFICATION CHECKLIST (Before answering):
☐ Is the information EXPLICITLY stated in the context?
☐ Can I cite the source (filename required, page optional)?
☐ Am I making ANY assumptions or using external knowledge? → If yes, REFUSE
☐ Does the question fall outside document scope? → If yes, use REFUSAL TEMPLATE

REASONING PROCESS:
1. VERIFY: Check if question can be answered from provided context
2. LOCATE: Find exact text/data in source documents
3. CITE: Identify source filename (and page if available)
4. ANSWER: Provide direct answer with citations

PROHIBITED BEHAVIORS:
❌ Using general knowledge (e.g., "Python is a programming language")
❌ Answering current events (time-sensitive questions)
❌ Inferring information not explicitly stated
❌ Providing answers without source citations
❌ Making assumptions about missing data
❌ Answering questions about entities not mentioned in documents

ANSWER STYLE:
✓ Direct and factual - quote exact phrases when possible
✓ Citations in EVERY sentence with factual claims (include page if available, filename is required)
✓ Use bullet points for clarity
✓ Acknowledge uncertainty when documents conflict
✓ Professional, concise language

FEW-SHOT EXAMPLES:

Example 1 - CORRECT Factual Answer:
Context: "[Source 1: Q2_Report.pdf, Page 3] The company's revenue in Q2 2024 was $15.2 million, representing a 23% increase from Q1's $12.4 million."
Question: What was the Q2 revenue growth?
Answer: The Q2 2024 revenue was $15.2 million, representing a 23% quarter-over-quarter growth from Q1's $12.4 million [Source 1: Q2_Report.pdf, Page 3].

Example 2 - CORRECT Comparison:
Context: "[Source 1: TechSpec.pdf, Page 5] Model A has 8GB RAM and 256GB storage. [Source 2: TechSpec.pdf, Page 7] Model B has 16GB RAM and 512GB storage."
Question: Compare Model A and Model B specifications.
Answer: Based on the technical specifications:
- RAM: Model A has 8GB while Model B has 16GB (2x more) [Source 1: TechSpec.pdf, Page 5; Source 2: TechSpec.pdf, Page 7]
- Storage: Model A has 256GB while Model B has 512GB (2x more) [Source 1: TechSpec.pdf, Page 5; Source 2: TechSpec.pdf, Page 7]

Example 3 - CORRECT Refusal (Insufficient Information):
Context: "[Source 1: Overview.pdf, Page 2] The product was launched in March 2024."
Question: What was the launch budget for the product?
Answer: I cannot find information about the launch budget in the provided documents. The available documents mention the product launch date (March 2024) but do not contain financial or budget details [Source 1: Overview.pdf, Page 2]. Please provide budget/financial documents to answer this question.

Example 4 - CORRECT Refusal (Out of Scope):
Context: "[Source 1: CompanyPolicy.pdf, Page 1-5] Various internal HR policies."
Question: What is the current stock price of our company?
Answer: I cannot find information about stock prices in the provided documents. The available documents cover internal HR policies only. Stock price information would require access to financial market data or investor relations documents.

Example 5 - CORRECT Refusal (External Knowledge):
Context: "[Source 1: ProjectPlan.pdf, Page 2] We will use Python for the backend."
Question: What is Python used for in general?
Answer: I cannot answer general questions about Python. I can only provide information from the uploaded documents. According to the project plan, Python will be used for the backend in this specific project [Source 1: ProjectPlan.pdf, Page 2].

Remember: When in doubt, REFUSE. Accuracy and honesty are more valuable than attempting to answer every question."""


# ===========================================
# Circuit Breaker Implementation
# ===========================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 30
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _success_count: int = field(default=0, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = datetime.now() - self._last_failure_time
                if elapsed > timedelta(seconds=self.recovery_timeout):
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
        return self._state
    
    def record_success(self):
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 3:  # 3 successes to close
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self):
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (failures: {self._failure_count})")
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        return self.state != CircuitState.OPEN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
        }


# ===========================================
# Retry Logic
# ===========================================

async def retry_with_backoff(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Result of the function
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_attempts} retry attempts failed: {e}")
    
    raise last_exception


# ===========================================
# LLM Metrics
# ===========================================

@dataclass
class LLMMetrics:
    """Metrics for LLM service monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_latency_ms: float = 0
    fallback_count: int = 0
    cache_hits: int = 0
    
    def record_request(
        self,
        success: bool,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        is_fallback: bool = False
    ):
        """Record a request."""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        
        if success:
            self.successful_requests += 1
            self.total_tokens_input += tokens_input
            self.total_tokens_output += tokens_output
        else:
            self.failed_requests += 1
        
        if is_fallback:
            self.fallback_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        avg_latency = self.total_latency_ms / max(self.total_requests, 1)
        success_rate = self.successful_requests / max(self.total_requests, 1) * 100
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "fallback_count": self.fallback_count,
            "cache_hits": self.cache_hits,
        }


# ===========================================
# LLM Service
# ===========================================

class LLMService:
    """
    Production-ready LLM Service with vLLM integration.
    
    Features:
    - vLLM as primary inference server
    - OpenAI fallback support
    - Circuit breaker for resilience
    - Retry with exponential backoff
    - Metrics collection
    - Streaming responses
    - Health monitoring
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.vllm_client = None
        self.ollama_client = None
        self._initialized = False
        
        # Circuit breakers
        self.vllm_circuit = CircuitBreaker(
            name="vllm",
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        self.ollama_circuit = CircuitBreaker(
            name="ollama",
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        
        # Metrics
        self.metrics = LLMMetrics()

        # Request tracking
        self._active_requests = 0
        self._max_concurrent = settings.MAX_CONCURRENT_REQUESTS
    
    async def initialize(self):
        """Initialize LLM clients."""
        if self._initialized:
            return

        from openai import AsyncOpenAI

        # Initialize vLLM client
        if self.provider == "vllm":
            self.vllm_client = AsyncOpenAI(
                base_url=settings.VLLM_BASE_URL,
                api_key=settings.VLLM_API_KEY,
                timeout=settings.REQUEST_TIMEOUT
            )
            logger.info(f"vLLM client initialized: {settings.VLLM_BASE_URL}")

        # Initialize Ollama client (local LLM)
        if self.provider == "ollama":
            self.ollama_client = AsyncOpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama doesn't require API key
                timeout=settings.REQUEST_TIMEOUT
            )
            logger.info(f"Ollama client initialized: {settings.OLLAMA_BASE_URL} ({settings.OLLAMA_MODEL})")

        self._initialized = True
    
    def _get_active_client(self):
        """Get the active client based on circuit state and provider."""
        if self.provider == "vllm" and self.vllm_circuit.can_execute() and self.vllm_client:
            return self.vllm_client, "vllm"

        if self.provider == "ollama" and self.ollama_circuit.can_execute() and self.ollama_client:
            return self.ollama_client, "ollama"

        raise RuntimeError("No available LLM client (all circuits open)")
    
    def _get_model_name(self, provider: str) -> str:
        """Get model name based on provider."""
        if provider == "vllm":
            return settings.VLLM_MODEL_NAME
        if provider == "ollama":
            return settings.OLLAMA_MODEL
        raise ValueError("Invalid provider")
    
    def _get_generation_params(self, provider: str) -> Dict[str, Any]:
        """Get generation parameters based on provider."""
        params = {
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "top_p": settings.LLM_TOP_P,
        }
        
        # vLLM supports additional parameters
        if provider == "vllm":
            params.update({
                "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
                "presence_penalty": settings.LLM_PRESENCE_PENALTY,
            })
        else:
            params.update({
                "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
                "presence_penalty": settings.LLM_PRESENCE_PENALTY,
            })
        
        return params
    
    async def generate(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """
        Generate an answer with resilience patterns.
        
        Features automatic retry and fallback.
        """
        if not self._initialized:
            await self.initialize()
        
        request_id = request_id or self._generate_request_id(question)
        start_time = time.time()
        is_fallback = False
        
        try:
            # Check concurrency limit
            if self._active_requests >= self._max_concurrent:
                raise RuntimeError("Max concurrent requests exceeded")
            
            self._active_requests += 1
            
            # Build messages
            messages = self._build_messages(
                question=question,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Try primary provider with retry
            async def make_request():
                client, provider = self._get_active_client()
                is_fallback_local = provider != self.provider
                try:
                    response = await client.chat.completions.create(
                        model=self._get_model_name(provider),
                        messages=messages,
                        **self._get_generation_params(provider)
                    )
                    # Log nội dung trả về từ LLM
                    logger.info(f"============LLM content: {response.choices[0].message.content}")
                    # Record success
                    if provider == "vllm":
                        self.vllm_circuit.record_success()
                    elif provider == "ollama":
                        self.ollama_circuit.record_success()
                    else:
                        self.openai_circuit.record_success()
                    return response, is_fallback_local
                except Exception as e:
                    # Record failure
                    if provider == "vllm":
                        self.vllm_circuit.record_failure()
                    else:
                        self.ollama_circuit.record_failure()
                    raise
            
            if settings.RETRY_ENABLED:
                response, is_fallback = await retry_with_backoff(
                    make_request,
                    max_attempts=settings.RETRY_MAX_ATTEMPTS,
                    initial_delay=settings.RETRY_INITIAL_DELAY,
                    max_delay=settings.RETRY_MAX_DELAY,
                    exponential_base=settings.RETRY_EXPONENTIAL_BASE
                )
            else:
                response, is_fallback = await make_request()
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_input = response.usage.prompt_tokens if response.usage else 0
            tokens_output = response.usage.completion_tokens if response.usage else 0
            
            self.metrics.record_request(
                success=True,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                is_fallback=is_fallback
            )
            
            logger.info(
                f"LLM request completed",
                extra={
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "is_fallback": is_fallback
                }
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(success=False, latency_ms=latency_ms)
            
            logger.error(
                f"LLM request failed: {e}",
                extra={"request_id": request_id, "latency_ms": round(latency_ms, 2)}
            )
            raise
            
        finally:
            self._active_requests -= 1
    
    async def generate_stream(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer with resilience patterns.
        
        Yields tokens as they are generated.
        """
        if not self._initialized:
            await self.initialize()
        
        request_id = request_id or self._generate_request_id(question)
        start_time = time.time()
        total_tokens = 0
        provider = None
        
        try:
            # Check concurrency limit
            if self._active_requests >= self._max_concurrent:
                raise RuntimeError("Max concurrent requests exceeded")
            
            self._active_requests += 1
            
            messages = self._build_messages(
                question=question,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            client, provider = self._get_active_client()
            is_fallback = provider != self.provider
            
            stream = await client.chat.completions.create(
                model=self._get_model_name(provider),
                messages=messages,
                stream=True,
                **self._get_generation_params(provider)
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    total_tokens += 1
                    yield token
            
            # Record success
            if provider == "vllm":
                self.vllm_circuit.record_success()
            elif provider == "ollama":
                self.ollama_circuit.record_success()
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(
                success=True,
                latency_ms=latency_ms,
                tokens_output=total_tokens,
                is_fallback=is_fallback
            )
            
        except Exception as e:
            if provider == "vllm":
                self.vllm_circuit.record_failure()
            elif provider == "ollama":
                self.ollama_circuit.record_failure()
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(success=False, latency_ms=latency_ms)
            
            logger.error(f"Streaming request failed: {e}", extra={"request_id": request_id})
            raise
            
        finally:
            self._active_requests -= 1
    
    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build the message array for the LLM."""
        messages = []
        
        # System message
        system = system_prompt or SYSTEM_PROMPT
        messages.append({"role": "system", "content": system})
        
        # Add conversation history (if any)
        if conversation_history:
            for turn in conversation_history[-6:]:  # Limit to last 3 exchanges
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
        
        # User message with context
        user_message = f"""Answer the following question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _generate_request_id(self, question: str) -> str:
        """Generate a unique request ID."""
        timestamp = str(time.time())
        hash_input = f"{question}:{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for LLM services.
        
        Returns health status for all configured providers.
        """
        health = {
            "status": "healthy",
            "providers": {},
            "circuits": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if not self._initialized:
            await self.initialize()
        
        # Check vLLM
        if self.vllm_client:
            try:
                start = time.time()
                response = await asyncio.wait_for(
                    self.vllm_client.chat.completions.create(
                        model=settings.VLLM_MODEL_NAME,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5
                    ),
                    timeout=settings.HEALTH_CHECK_TIMEOUT
                )
                latency = (time.time() - start) * 1000
                health["providers"]["vllm"] = {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                    "model": settings.VLLM_MODEL_NAME
                }
            except Exception as e:
                health["providers"]["vllm"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        # Circuit breaker status
        health["circuits"]["vllm"] = self.vllm_circuit.get_stats()
        
        # Check if all providers are unhealthy
        all_unhealthy = all(
            p.get("status") == "unhealthy" 
            for p in health["providers"].values()
        )
        if all_unhealthy:
            health["status"] = "unhealthy"
        
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "llm_metrics": self.metrics.get_stats(),
            "circuits": {
                "vllm": self.vllm_circuit.get_stats(),
                "openai": self.openai_circuit.get_stats()
            },
            "active_requests": self._active_requests,
            "max_concurrent_requests": self._max_concurrent
        }
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down LLM service...")
        
        # Wait for active requests to complete (with timeout)
        timeout = 30
        start = time.time()
        while self._active_requests > 0 and (time.time() - start) < timeout:
            logger.info(f"Waiting for {self._active_requests} active requests...")
            await asyncio.sleep(1)
        
        if self._active_requests > 0:
            logger.warning(f"Force shutdown with {self._active_requests} active requests")
        
        self._initialized = False
        logger.info("LLM service shutdown complete")
