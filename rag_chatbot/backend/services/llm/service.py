"""
LLM Service with Ollama Integration.

Features:
- Ollama as LLM server (via OpenAI-compatible API)
- Circuit breaker pattern
- Retry with exponential backoff
- Streaming responses
- Health monitoring
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from backend.config import settings
from backend.core.resilience import CircuitBreaker, retry_with_backoff
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service using Ollama via OpenAI-compatible API.

    Ollama runs as a separate process and handles GPU/CPU automatically.
    This service connects to it via HTTP, with circuit breaker and retry
    for resilience.
    """

    def __init__(self):
        self._client = None
        self._initialized = False
        self._circuit = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )
        self._active_requests = 0
        self._max_concurrent = settings.MAX_CONCURRENT_REQUESTS

    async def initialize(self):
        if self._initialized:
            return
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=settings.REQUEST_TIMEOUT,
        )
        self._initialized = True
        logger.info("Ollama client initialized: %s (%s)",
                     settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)

    def _get_active_client(self):
        """Get the Ollama client if circuit is closed."""
        if self._circuit.can_execute() and self._client:
            return self._client, "ollama"
        raise RuntimeError("LLM circuit breaker is open — Ollama unavailable")

    def _get_model_name(self, provider: str) -> str:
        return settings.OLLAMA_MODEL

    def _get_generation_params(self) -> Dict[str, Any]:
        return {
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "top_p": settings.LLM_TOP_P,
            "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
            "presence_penalty": settings.LLM_PRESENCE_PENALTY,
        }

    async def generate(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            if self._active_requests >= self._max_concurrent:
                raise RuntimeError("Max concurrent requests exceeded")
            self._active_requests += 1

            messages = self._build_messages(question, context,
                                            conversation_history, system_prompt)

            async def make_request():
                client, _ = self._get_active_client()
                try:
                    response = await client.chat.completions.create(
                        model=settings.OLLAMA_MODEL,
                        messages=messages,
                        **self._get_generation_params(),
                    )
                    self._circuit.record_success()
                    return response
                except Exception:
                    self._circuit.record_failure()
                    raise

            if settings.RETRY_ENABLED:
                response = await retry_with_backoff(
                    make_request,
                    max_attempts=settings.RETRY_MAX_ATTEMPTS,
                    initial_delay=settings.RETRY_INITIAL_DELAY,
                    max_delay=settings.RETRY_MAX_DELAY,
                    exponential_base=settings.RETRY_EXPONENTIAL_BASE,
                )
            else:
                response = await make_request()

            latency_ms = (time.time() - start_time) * 1000
            logger.info("LLM request completed (%.0fms)", latency_ms)
            return response.choices[0].message.content

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("LLM request failed (%.0fms): %s", latency_ms, e)
            raise

        finally:
            self._active_requests -= 1

    async def generate_stream(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            if self._active_requests >= self._max_concurrent:
                raise RuntimeError("Max concurrent requests exceeded")
            self._active_requests += 1

            messages = self._build_messages(question, context,
                                            conversation_history, system_prompt)
            client, _ = self._get_active_client()

            stream = await client.chat.completions.create(
                model=settings.OLLAMA_MODEL,
                messages=messages,
                stream=True,
                **self._get_generation_params(),
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            self._circuit.record_success()
            logger.info("Streaming completed (%.0fms)", (time.time() - start_time) * 1000)

        except Exception as e:
            self._circuit.record_failure()
            logger.error("Streaming failed: %s", e)
            raise

        finally:
            self._active_requests -= 1

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]

        if conversation_history:
            for turn in conversation_history[-6:]:
                content = turn["content"]
                if turn["role"] == "assistant" and len(content) > 1200:
                    content = content[:1200] + "..."
                messages.append({"role": turn["role"], "content": content})

        user_message = f"""CONTEXT:
{context}

---
QUESTION: {question}"""
        messages.append({"role": "user", "content": user_message})
        return messages

    async def health_check(self) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        health = {"status": "healthy", "timestamp": datetime.now().isoformat()}
        try:
            start = time.time()
            await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=settings.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                ),
                timeout=settings.HEALTH_CHECK_TIMEOUT,
            )
            health["latency_ms"] = round((time.time() - start) * 1000, 2)
            health["model"] = settings.OLLAMA_MODEL
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        return health

    async def shutdown(self):
        logger.info("Shutting down LLM service...")
        timeout = 30
        start = time.time()
        while self._active_requests > 0 and (time.time() - start) < timeout:
            await asyncio.sleep(1)
        self._initialized = False
        logger.info("LLM service shutdown complete")
