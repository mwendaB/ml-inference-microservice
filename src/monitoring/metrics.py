from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


REQUEST_LATENCY = Histogram("inference_latency_ms", "Inference latency", buckets=(5, 10, 25, 50, 100, 250, 500, 1000))
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
ACTIVE_STREAMS = Gauge("active_streams", "Active RTSP streams")


class MetricsStore:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379) -> None:
        self._client = None
        if redis is not None:
            try:
                self._client = redis.Redis(host=redis_host, port=redis_port)
            except Exception as exc:
                logger.warning("redis_unavailable", extra={"extra": {"error": str(exc)}})

    def push(self, key: str, value: Any) -> None:
        if self._client is None:
            return
        try:
            self._client.lpush(key, value)
            self._client.ltrim(key, 0, 200)
        except Exception as exc:
            logger.warning("redis_push_failed", extra={"extra": {"error": str(exc)}})

    def fetch(self, key: str, limit: int = 200) -> list[Any]:
        if self._client is None:
            return []
        try:
            return [item.decode("utf-8") for item in self._client.lrange(key, 0, limit - 1)]
        except Exception:
            return []


class MetricsRecorder:
    def __init__(self, store: Optional[MetricsStore] = None) -> None:
        self.store = store or MetricsStore()

    def record_inference(self, latency_ms: float, model: str, backend: str) -> None:
        REQUEST_LATENCY.observe(latency_ms)
        REQUEST_COUNT.inc()
        self.store.push("latency_ms", latency_ms)
        self.store.push("model_used", model)
        self.store.push("backend", backend)

    def set_active_streams(self, count: int) -> None:
        ACTIVE_STREAMS.set(count)
        self.store.push("active_streams", count)
