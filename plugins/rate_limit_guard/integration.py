"""Rate Limit Guard — Helper module for core integration.

This module provides the interface that core code calls to respect
plugin decisions from pre_api_request hook. It lives in the plugin
directory but is imported by core code (conditional import, safe if
plugin not installed).
"""

from typing import Any, Dict, List, Optional


def apply_rate_limit_decision(
    decision: Optional[Dict[str, Any]],
    model: str,
    messages: List[Dict],
    make_api_call,
) -> str:
    """Apply the rate limit guard's decision.

    Args:
        decision: Return value from pre_api_request hook
        model: Original model name
        messages: Conversation messages
        make_api_call: Function(model, messages) -> response string

    Returns:
        API response string (from cache, fallback, or direct call)
    """
    if decision is None:
        # No intervention — proceed normally
        return make_api_request(model, messages)

    if decision.get("cache_hit"):
        # Return cached response — skip API call
        return decision["response"]

    if decision.get("use_fallback"):
        # Route to fallback model
        fallback_model = decision["fallback_model"]
        return make_api_call(fallback_model, messages)

    if decision.get("queue_request"):
        # Request is queued — caller should handle retry_after
        # This is handled by the agent's retry logic
        raise RateLimitQueuedError(
            delay=decision.get("retry_after", 2.0),
            message="Request queued due to rate limit"
        )

    if decision.get("delay_seconds"):
        # Apply backoff delay
        import time
        time.sleep(decision["delay_seconds"])
        # Retry the same request
        return make_api_call(model, messages)

    # Default: proceed normally
    return make_api_call(model, messages)


class RateLimitQueuedError(Exception):
    """Raised when a request is queued for later processing."""
    def __init__(self, delay: float, message: str = ""):
        self.delay = delay
        self.message = message
        super().__init__(f"Queued (retry after {delay}s): {message}")
