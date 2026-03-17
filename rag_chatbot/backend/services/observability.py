"""
Observability Module - Structured Logging, Metrics, and Tracing.

Production-ready observability stack:
- Structured JSON logging
- Prometheus metrics
- Request tracing context
- Performance monitoring
"""
import logging
import sys
import json
import time
import uuid
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
import asyncio

from backend.config import settings

# ===========================================
# Request Context
# ===========================================

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]


def get_request_context() -> Dict[str, str]:
    """Get current request context."""
    return {
        "request_id": request_id_var.get(),
        "user_id": user_id_var.get(),
        "trace_id": trace_id_var.get()
    }


# ===========================================
# Structured JSON Logging
# ===========================================

class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by log aggregators
    like ELK Stack, Datadog, or CloudWatch.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.SERVICE_NAME,
            "environment": settings.APP_ENVIRONMENT,
            "version": settings.APP_VERSION,
        }
        
        # Add location info
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add request context
        context = get_request_context()
        if any(context.values()):
            log_data["context"] = {k: v for k, v in context.items() if v}
        
        # Add extra fields
        if hasattr(record, 'extra') and record.extra:
            log_data["extra"] = record.extra
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        context = get_request_context()
        request_id = context.get("request_id", "")
        
        prefix = f"[{timestamp}] [{record.levelname}]"
        if request_id:
            prefix += f" [{request_id}]"
        
        return f"{prefix} {record.name}: {record.getMessage()}"


def setup_logging():
    """
    Configure logging for the application.
    
    Sets up structured JSON logging for production and
    text logging for development.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.LOG_FORMAT == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler (always JSON for log aggregation)
    if settings.LOG_FILE:
        import os
        os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
        
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    logging.info("Logging configured", extra={"format": settings.LOG_FORMAT})


# ===========================================
# Prometheus Metrics
# ===========================================

class PrometheusMetrics:
    """
    Prometheus metrics collector.
    
    Collects:
    - HTTP request metrics (count, latency, errors)
    - LLM inference metrics
    - Document processing metrics
    - Cache hit/miss rates
    - System health metrics
    """
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {
            # Request metrics
            "http_requests_total": {},
            "http_request_duration_seconds": {},
            "http_requests_in_progress": 0,
            
            # LLM metrics
            "llm_requests_total": {"success": 0, "failure": 0},
            "llm_request_duration_seconds": [],
            "llm_tokens_total": {"input": 0, "output": 0},
            "llm_fallback_total": 0,
            
            # Retrieval metrics
            "retrieval_requests_total": 0,
            "retrieval_duration_seconds": [],
            "retrieval_chunks_returned": [],
            
            # Document metrics
            "documents_processed_total": 0,
            "documents_processing_duration_seconds": [],
            "chunks_created_total": 0,
            
            # Cache metrics
            "cache_hits_total": 0,
            "cache_misses_total": 0,
            
            # System metrics
            "active_requests": 0,
            "last_health_check": None
        }
        
        self._start_time = time.time()
    
    def inc_http_requests(self, method: str, endpoint: str, status_code: int):
        """Increment HTTP request counter."""
        key = f"{method}:{endpoint}:{status_code}"
        self._metrics["http_requests_total"][key] = \
            self._metrics["http_requests_total"].get(key, 0) + 1
    
    def observe_http_duration(self, method: str, endpoint: str, duration: float):
        """Record HTTP request duration."""
        key = f"{method}:{endpoint}"
        if key not in self._metrics["http_request_duration_seconds"]:
            self._metrics["http_request_duration_seconds"][key] = []
        self._metrics["http_request_duration_seconds"][key].append(duration)
    
    def inc_llm_requests(self, success: bool):
        """Increment LLM request counter."""
        key = "success" if success else "failure"
        self._metrics["llm_requests_total"][key] += 1
    
    def observe_llm_duration(self, duration: float):
        """Record LLM request duration."""
        self._metrics["llm_request_duration_seconds"].append(duration)
    
    def inc_llm_tokens(self, input_tokens: int, output_tokens: int):
        """Increment token counters."""
        self._metrics["llm_tokens_total"]["input"] += input_tokens
        self._metrics["llm_tokens_total"]["output"] += output_tokens
    
    def inc_llm_fallback(self):
        """Increment fallback counter."""
        self._metrics["llm_fallback_total"] += 1
    
    def inc_cache_hit(self):
        """Increment cache hit counter."""
        self._metrics["cache_hits_total"] += 1
    
    def inc_cache_miss(self):
        """Increment cache miss counter."""
        self._metrics["cache_misses_total"] += 1
    
    def inc_documents_processed(self):
        """Increment documents processed counter."""
        self._metrics["documents_processed_total"] += 1
    
    def inc_chunks_created(self, count: int):
        """Increment chunks created counter."""
        self._metrics["chunks_created_total"] += count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        uptime = time.time() - self._start_time
        
        # Calculate averages
        llm_durations = self._metrics["llm_request_duration_seconds"]
        avg_llm_duration = sum(llm_durations) / len(llm_durations) if llm_durations else 0
        
        cache_total = (self._metrics["cache_hits_total"] + 
                      self._metrics["cache_misses_total"])
        cache_hit_rate = (self._metrics["cache_hits_total"] / cache_total * 100 
                         if cache_total > 0 else 0)
        
        return {
            "uptime_seconds": round(uptime, 2),
            "http": {
                "requests_total": sum(self._metrics["http_requests_total"].values()),
                "requests_by_endpoint": self._metrics["http_requests_total"],
                "requests_in_progress": self._metrics["http_requests_in_progress"]
            },
            "llm": {
                "requests_total": self._metrics["llm_requests_total"],
                "average_duration_seconds": round(avg_llm_duration, 3),
                "tokens_total": self._metrics["llm_tokens_total"],
                "fallback_total": self._metrics["llm_fallback_total"]
            },
            "cache": {
                "hits_total": self._metrics["cache_hits_total"],
                "misses_total": self._metrics["cache_misses_total"],
                "hit_rate_percent": round(cache_hit_rate, 2)
            },
            "documents": {
                "processed_total": self._metrics["documents_processed_total"],
                "chunks_created_total": self._metrics["chunks_created_total"]
            }
        }
    
    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Helper to add metric
        def add_metric(name: str, value: Any, metric_type: str = "gauge", 
                      labels: Optional[Dict] = None):
            label_str = ""
            if labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name}{label_str} {value}")
        
        # Uptime
        add_metric("app_uptime_seconds", time.time() - self._start_time)
        
        # HTTP requests
        for key, count in self._metrics["http_requests_total"].items():
            method, endpoint, status = key.split(":")
            add_metric("http_requests_total", count, "counter",
                      {"method": method, "endpoint": endpoint, "status": status})
        
        # LLM metrics
        add_metric("llm_requests_total", 
                  self._metrics["llm_requests_total"]["success"], "counter",
                  {"status": "success"})
        add_metric("llm_requests_total",
                  self._metrics["llm_requests_total"]["failure"], "counter",
                  {"status": "failure"})
        add_metric("llm_tokens_total",
                  self._metrics["llm_tokens_total"]["input"], "counter",
                  {"type": "input"})
        add_metric("llm_tokens_total",
                  self._metrics["llm_tokens_total"]["output"], "counter",
                  {"type": "output"})
        
        # Cache metrics
        add_metric("cache_hits_total", self._metrics["cache_hits_total"], "counter")
        add_metric("cache_misses_total", self._metrics["cache_misses_total"], "counter")
        
        # Document metrics
        add_metric("documents_processed_total", 
                  self._metrics["documents_processed_total"], "counter")
        add_metric("chunks_created_total",
                  self._metrics["chunks_created_total"], "counter")
        
        return "\n".join(lines)


# Global metrics instance
metrics = PrometheusMetrics()


# ===========================================
# Performance Decorators
# ===========================================

def log_execution_time(operation: str):
    """
    Decorator to log execution time of a function.
    
    Args:
        operation: Name of the operation for logging
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(
                    f"{operation} completed",
                    extra={
                        "operation": operation,
                        "duration_seconds": round(duration, 3),
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logging.error(
                    f"{operation} failed",
                    extra={
                        "operation": operation,
                        "duration_seconds": round(duration, 3),
                        "status": "error",
                        "error": str(e)
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(
                    f"{operation} completed",
                    extra={
                        "operation": operation,
                        "duration_seconds": round(duration, 3),
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logging.error(
                    f"{operation} failed",
                    extra={
                        "operation": operation,
                        "duration_seconds": round(duration, 3),
                        "status": "error",
                        "error": str(e)
                    }
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_metrics(metric_name: str, labels: Optional[Dict] = None):
    """
    Decorator to track metrics for a function.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                return await func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                # Record metrics
                if metric_name == "llm":
                    metrics.inc_llm_requests(success)
                    metrics.observe_llm_duration(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                return func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                if metric_name == "llm":
                    metrics.inc_llm_requests(success)
                    metrics.observe_llm_duration(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ===========================================
# Health Check Utilities
# ===========================================

class HealthChecker:
    """
    Centralized health checker for all services.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, Any] = {}
    
    def register(self, name: str, check_func: Callable):
        """Register a health check function."""
        self._checks[name] = check_func
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {}
        }
        
        for name, check_func in self._checks.items():
            try:
                start = time.time()
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await asyncio.wait_for(
                        check_func(),
                        timeout=settings.HEALTH_CHECK_TIMEOUT
                    )
                else:
                    check_result = check_func()
                
                duration = time.time() - start
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "latency_ms": round(duration * 1000, 2)
                }
                
                if not check_result:
                    results["status"] = "degraded"
                    
            except asyncio.TimeoutError:
                results["checks"][name] = {
                    "status": "timeout",
                    "error": "Health check timed out"
                }
                results["status"] = "degraded"
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["status"] = "degraded"
        
        # If all checks failed, mark as unhealthy
        all_failed = all(
            c.get("status") != "healthy" 
            for c in results["checks"].values()
        )
        if all_failed and results["checks"]:
            results["status"] = "unhealthy"
        
        self._last_results = results
        return results
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get last health check results."""
        return self._last_results


# Global health checker
health_checker = HealthChecker()
