"""Tests for rate_limit_guard plugin."""

import time
import pytest
from plugins.rate_limit_guard import (
    _cache_key,
    _cache_get,
    _cache_set,
    _cache,
    _cache_lock,
    _is_rate_limited,
    _mark_rate_limited,
    _clear_rate_limit,
    _backoff_delay,
    _get_session,
    get_status,
    reset_session,
    _session_state,
    _state_lock,
    _ENABLED,
    _CACHE_TTL,
    _FALLBACK_MODEL,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset all global state between tests."""
    # Clear cache
    with _cache_lock:
        _cache.clear()
    # Clear session state
    with _state_lock:
        _session_state.clear()
    # Clear rate limit
    _clear_rate_limit()
    yield


class TestCacheKey:
    def test_same_messages_same_key(self):
        messages = [{"role": "user", "content": "hello"}]
        key1 = _cache_key("model-a", messages)
        key2 = _cache_key("model-a", messages)
        assert key1 == key2

    def test_different_models_different_keys(self):
        messages = [{"role": "user", "content": "hello"}]
        key1 = _cache_key("model-a", messages)
        key2 = _cache_key("model-b", messages)
        assert key1 != key2

    def test_different_messages_different_keys(self):
        key1 = _cache_key("model", [{"role": "user", "content": "hello"}])
        key2 = _cache_key("model", [{"role": "user", "content": "world"}])
        assert key1 != key2


class TestCache:
    def test_cache_set_and_get(self):
        key = _cache_key("test", [{"role": "user", "content": "test"}])
        _cache_set(key, "response")
        assert _cache_get(key) == "response"

    def test_cache_miss(self):
        assert _cache_get("nonexistent") is None

    def test_cache_expiration(self, monkeypatch):
        key = _cache_key("test", [{"role": "user", "content": "test"}])
        _cache_set(key, "response")

        # Fast-forward time past TTL
        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + _CACHE_TTL + 10)
        assert _cache_get(key) is None


class TestRateLimit:
    def test_not_rate_limited_initially(self):
        assert _is_rate_limited() is False

    def test_mark_rate_limited(self):
        _mark_rate_limited()
        assert _is_rate_limited() is True

    def test_clear_rate_limit(self):
        _mark_rate_limited()
        _clear_rate_limit()
        assert _is_rate_limited() is False

    def test_auto_reset_after_5_min(self, monkeypatch):
        _mark_rate_limited()
        assert _is_rate_limited() is True

        # Fast-forward 5+ minutes
        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + 301)
        assert _is_rate_limited() is False


class TestBackoff:
    def test_backoff_increases_exponentially(self):
        delay0 = _backoff_delay(0)
        delay1 = _backoff_delay(1)
        delay2 = _backoff_delay(2)
        assert delay0 < delay1 < delay2

    def test_backoff_respects_max(self):
        delay = _backoff_delay(20)  # Would be huge without cap
        assert delay <= 70  # max + 10% jitter


class TestSessionState:
    def test_session_created(self):
        state = _get_session("test-session")
        assert "total_requests" in state
        assert state["total_requests"] == 0

    def test_session_persists_across_calls(self):
        state1 = _get_session("test-session")
        state1["total_requests"] = 5
        state2 = _get_session("test-session")
        assert state2["total_requests"] == 5

    def test_reset_session(self):
        _get_session("test-session")
        reset_session("test-session")
        with _state_lock:
            assert "test-session" not in _session_state


class TestGetStatus:
    def test_status_structure(self):
        status = get_status()
        assert "enabled" in status
        assert "cache_size" in status
        assert "queue_size" in status
        assert "config" in status

    def test_status_with_session(self):
        _get_session("status-test")
        status = get_status("status-test")
        assert "session" in status
        assert "total_requests" in status["session"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
