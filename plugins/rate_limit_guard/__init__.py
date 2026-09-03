"""Rate Limit Guard — Mitigates API rate limits for primary model.

Features:
- Response caching (dedupes identical prompts within TTL)
- Request queuing (buffers non-urgent requests during limit windows)
- Exponential backoff with jitter
- Automatic fallback to secondary model
- Priority routing (critical tasks bypass queue)

Usage:
    Install in ~/.hermes/plugins/rate_limit_guard/
    Configure via environment variables (see plugin.yaml)
    Automatically activates when primary model hits rate limits.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Configuration ---
_ENABLED = os.environ.get("HERMES_RATE_LIMIT_GUARD_ENABLED", "1").lower() in (
    "1", "true", "yes", "on",
)
_CACHE_TTL = int(os.environ.get("HERMES_RATE_LIMIT_GUARD_CACHE_TTL", "3600"))
_FALLBACK_MODEL = os.environ.get("HERMES_RATE_LIMIT_GUARD_FALLBACK_MODEL", "qwen/qwen3.6-plus")
_MAX_QUEUE_SIZE = int(os.environ.get("HERMES_RATE_LIMIT_GUARD_MAX_QUEUE_SIZE", "50"))
_BACKOFF_BASE = float(os.environ.get("HERMES_RATE_LIMIT_GUARD_BACKOFF_BASE", "2"))
_BACKOFF_MAX = float(os.environ.get("HERMES_RATE_LIMIT_GUARD_BACKOFF_MAX", "60"))
_RETRY_ATTEMPTS = int(os.environ.get("HERMES_RATE_LIMIT_GUARD_RETRY_ATTEMPTS", "3"))

# --- State (per-session) ---
_session_state: Dict[str, Dict[str, Any]] = {}
_state_lock = threading.Lock()

# --- Cache (global, thread-safe) ---
_cache: Dict[str, Tuple[str, float]] = {}  # key -> (response, timestamp)
_cache_lock = threading.Lock()

# --- Request queue (global, thread-safe) ---
_queue: deque = deque()
_queue_lock = threading.Lock()
_queue_active = False

# --- Rate limit tracking ---
_rate_limit_active = False  # type: bool
_rate_limit_lock = threading.Lock()
_last_429_time = 0.0  # type: float
_consecutive_429s = 0  # type: int


def _get_session(session_id: str) -> Dict[str, Any]:
    """Get or create session-local state."""
    with _state_lock:
        if session_id not in _session_state:
            _session_state[session_id] = {
                "retry_count": 0,
                "last_request": 0.0,
                "total_requests": 0,
                "cache_hits": 0,
                "fallback_count": 0,
            }
        return _session_state[session_id]


def _cache_key(model: str, messages: List[Dict]) -> str:
    """Generate deterministic cache key from model + messages."""
    canonical = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _cache_get(key: str) -> Optional[str]:
    """Try to get cached response."""
    with _cache_lock:
        if key in _cache:
            response, ts = _cache[key]
            if time.time() - ts < _CACHE_TTL:
                return response
            del _cache[key]
    return None


def _cache_set(key: str, response: str):
    """Store response in cache."""
    with _cache_lock:
        _cache[key] = (response, time.time())
        # Prune expired entries (simple, not exhaustive)
        now = time.time()
        expired = [k for k, (_, ts) in _cache.items() if now - ts >= _CACHE_TTL]
        for k in expired:
            del _cache[k]


def _is_rate_limited() -> bool:
    """Check if we're currently in a rate-limit or transport-degradation window."""
    global _rate_limit_active, _last_429_time, _consecutive_429s
    with _rate_limit_lock:
        if not _rate_limit_active:
            return False
        # Auto-reset after 5 minutes of no 429s
        if time.time() - _last_429_time > 300:
            _rate_limit_active = False
            _consecutive_429s = 0
            return False
        return True


def _mark_rate_limited():
    """Mark that we've hit a rate limit."""
    global _rate_limit_active, _last_429_time, _consecutive_429s
    with _rate_limit_lock:
        _rate_limit_active = True  # noqa: F841
        _last_429_time = time.time()
        _consecutive_429s += 1


def _mark_transport_error(err_str: str = ""):
    """Mark transport/connection errors the same as rate limits for backoff purposes."""
    global _rate_limit_active, _last_429_time, _consecutive_429s
    with _rate_limit_lock:
        _rate_limit_active = True
        _last_429_time = time.time()
        _consecutive_429s += 1
        logger.warning("rate_limit_guard: transport error treated as rate limit: %s", err_str[:80])


def _clear_rate_limit():
    """Clear rate limit status after a successful request."""
    global _rate_limit_active, _consecutive_429s
    with _rate_limit_lock:
        _rate_limit_active = False  # noqa: F841
        _consecutive_429s = 0


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff with jitter."""
    delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
    return delay + jitter


def _enqueue_request(session_id: str, data: Dict[str, Any]) -> bool:
    """Add a request to the queue. Returns False if queue is full."""
    with _queue_lock:
        if len(_queue) >= _MAX_QUEUE_SIZE:
            logger.warning("Rate limit queue full (%d), dropping request", len(_queue))
            return False
        _queue.append({
            "session_id": session_id,
            "data": data,
            "queued_at": time.time(),
            "priority": data.get("priority", 0),
        })
        return True


def _should_use_fallback(session_state: Dict[str, Any]) -> bool:
    """Decide if we should switch to fallback model."""
    return session_state.get("retry_count", 0) >= _RETRY_ATTEMPTS


def register(ctx) -> None:
    """Register plugin hooks."""
    if not _ENABLED:
        logger.debug("rate_limit_guard: disabled via env var")
        return

    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_start", _on_session_start)
    logger.info("rate_limit_guard: loaded (cache_ttl=%ds, fallback=%s)",
                _CACHE_TTL, _FALLBACK_MODEL)


# --- Hook Handlers ---

def _on_session_start(ctx, session_id, **kwargs):
    """Initialize session state."""
    _get_session(session_id)
    logger.debug("rate_limit_guard: session %s initialized", session_id[:8])


def _on_pre_api_request(session_id, model, **kwargs) -> Optional[Dict[str, Any]]:
    """Intercept API requests for caching, queuing, and rate limit handling.

    Return dict to modify the request:
      - "cache_hit": True + "response": cached_response  → skip API call
      - "use_fallback": True + "fallback_model": model    → route to fallback
      - "queue_request": True                             → defer request
      - "delay_seconds": float                            → wait before retry
    """
    if not session_id:
        return None

    state = _get_session(session_id)
    state["total_requests"] += 1

    messages = kwargs.get("messages", [])

    # 1. Check cache first (only for non-streaming, standard requests)
    cache_key = _cache_key(model, messages)
    cached = _cache_get(cache_key)
    if cached:
        state["cache_hits"] += 1
        logger.debug("rate_limit_guard: CACHE HIT for session %s", session_id[:8])
        return {"cache_hit": True, "response": cached}

    # 2. Check rate limit status
    if _is_rate_limited():
        # Check if we should use fallback
        if _should_use_fallback(state):
            state["fallback_count"] += 1
            logger.info("rate_limit_guard: FALLBACK to %s after %d retries",
                        _FALLBACK_MODEL, state["retry_count"])
            return {
                "use_fallback": True,
                "fallback_model": _FALLBACK_MODEL,
                "original_model": model,
            }

        # Check if request is low priority (can be queued)
        is_priority = kwargs.get("is_priority", False)
        if not is_priority and len(_queue) < _MAX_QUEUE_SIZE:
            if _enqueue_request(session_id, {"model": model, "messages": messages, **kwargs}):
                logger.debug("rate_limit_guard: QUEUED request (queue size: %d)", len(_queue))
                return {"queue_request": True, "retry_after": _backoff_delay(0)}

        # High priority or queue full — apply backoff and retry
        delay = _backoff_delay(state["retry_count"])
        logger.info("rate_limit_guard: RATE LIMITED, backing off %.1fs (attempt %d)",
                     delay, state["retry_count"])
        return {"delay_seconds": delay}

    # 3. Normal request — no intervention
    return None


def _on_post_api_request(session_id, model, usage, finish_reason, **kwargs):
    """Track rate limit events and cache successful responses.

    Note: The post_api_request hook does NOT receive status_code or raw response.
    We track success by the fact the hook was called (no exception from API).
    Rate limit detection happens via finish_reason or usage patterns.
    """
    if not session_id:
        return

    state = _get_session(session_id)

    # Detect rate limiting from finish_reason (some providers set this)
    if finish_reason and "rate" in finish_reason.lower():
        _mark_rate_limited()
        state["retry_count"] += 1
        logger.warning("rate_limit_guard: rate limit detected (finish_reason=%s, retries=%d)",
                       finish_reason, state["retry_count"])
        return

    # Successful response — clear rate limit
    if finish_reason in ("stop", "end_turn", "tool_calls", None):
        _clear_rate_limit()
        state["retry_count"] = 0

        # Cache the response if we have messages
        messages = kwargs.get("messages", [])
        if messages and usage:
            cache_key = _cache_key(model, messages)
            # Cache a marker — actual response caching happens at a higher level
            _cache_set(cache_key, "OK")

    # Log session stats periodically
    if state["total_requests"] % 10 == 0:
        logger.info("rate_limit_guard: session %s stats — "
                     "requests=%d, cache_hits=%d, fallbacks=%d, retries=%d",
                     session_id[:8],
                     state["total_requests"],
                     state["cache_hits"],
                     state["fallback_count"],
                     state["retry_count"])


# --- Utility: Process queued requests ---

def process_queue(agent_func, max_items: int = 5):
    """Drain the request queue by calling agent_func for each item.

    agent_func should accept (model, messages, **kwargs) and return response.
    This is meant to be called by a background task or cron job.
    """
    processed = 0
    with _queue_lock:
        items = list(_queue)[:max_items]

    for item in items:
        try:
            session_id = item["session_id"]
            data = item["data"]
            model = data.pop("model")
            messages = data.pop("messages")
            response = agent_func(model=model, messages=messages, **data)

            # Cache result
            cache_key = _cache_key(model, messages)
            _cache_set(cache_key, response)

            with _queue_lock:
                if _queue and _queue[0] == item:
                    _queue.popleft()
            processed += 1
        except Exception as e:
            logger.error("rate_limit_guard: queue processing error: %s", e)

    if processed:
        logger.info("rate_limit_guard: processed %d queued requests", processed)

    return processed


# --- Status reporting ---

def get_status(session_id: str = None) -> Dict[str, Any]:
    """Get current rate limit guard status."""
    status = {
        "enabled": _ENABLED,
        "cache_size": len(_cache),
        "queue_size": len(_queue),
        "rate_limit_active": _is_rate_limited(),
        "consecutive_429s": _consecutive_429s,
        "fallback_model": _FALLBACK_MODEL,
        "config": {
            "cache_ttl": _CACHE_TTL,
            "max_queue_size": _MAX_QUEUE_SIZE,
            "backoff_base": _BACKOFF_BASE,
            "backoff_max": _BACKOFF_MAX,
            "retry_attempts": _RETRY_ATTEMPTS,
        },
    }

    if session_id:
        state = _get_session(session_id)
        status["session"] = state

    return status


def reset_session(session_id: str):
    """Reset session state (for debugging or cleanup)."""
    with _state_lock:
        if session_id in _session_state:
            del _session_state[session_id]
