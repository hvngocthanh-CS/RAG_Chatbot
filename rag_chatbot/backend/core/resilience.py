"""
Resilience patterns: Circuit Breaker and Retry with Exponential Backoff.

These are cross-cutting concerns used by multiple services (LLM, embedding, etc.).
"""
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ===========================================
# Circuit Breaker
# ===========================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Prevents cascading failures by stopping requests to failing services."""
    failure_threshold: int = 5
    recovery_timeout: int = 30

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _success_count: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = datetime.now() - self._last_failure_time
            if elapsed > timedelta(seconds=self.recovery_timeout):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 3:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN


# ===========================================
# Retry with Exponential Backoff
# ===========================================

async def retry_with_backoff(func, max_attempts=3, initial_delay=1.0,
                              max_delay=10.0, exponential_base=2.0):
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                logger.warning("Retry %d/%d failed: %s. Retrying in %.2fs",
                               attempt + 1, max_attempts, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("All %d retry attempts failed: %s", max_attempts, e)
    raise last_exception
