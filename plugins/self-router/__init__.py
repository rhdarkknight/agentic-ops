"""Self-Router plugin — Hermes delegating on its own initiative.

WS4 of the self-router plan. Wires the router's decision layer into the
``pre_tool_call`` hook so that, before the first mutating call of a task, the
router can recommend (or, when both auto_dispatch and a verified executor are
set, dispatch) away from Hermes to a specialist/stateless harness.

Safety defaults (trust-preserving):
- Default is RECOMMEND-only. The hook emits a ``warn`` decision with a
  "recommend dispatch to <agent>" reason; it NEVER auto-dispatches unless
  ``self_router.auto_dispatch`` is true AND the target harness is in the
  verified ``self_router.executors`` set (populated by the Phase-0 probe).
- Read-only tools never trigger routing.
- Cooldown gates the per-tool-call hook so it does not assess on every call.

The router's decision logic (self_assess + anchoring + router) is pure and
unit-tested; the dispatch is a separate, gated layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Per-session ACTIVE model capture. on_session_start fires with model=agent.model
# (the runtime model for that session, which may differ from config.yaml's
# model.default). The cascade reads this so it routes on the ACTIVE model, not
# the config default. Keyed by session_id; pruned to bound growth.
_SESSION_MODELS: Dict[str, str] = {}
_MAX_SESSION_MODELS = 500


def _record_session_model(session_id: str = "", model: str = "", **kwargs: Any) -> None:
    """on_session_start: record the ACTIVE session model."""
    if not session_id or not model:
        return
    _SESSION_MODELS[session_id] = model
    if len(_SESSION_MODELS) > _MAX_SESSION_MODELS:
        # Drop oldest (dict preserves insertion order).
        for _ in range(len(_SESSION_MODELS) - _MAX_SESSION_MODELS):
            _SESSION_MODELS.pop(next(iter(_SESSION_MODELS)), None)


def _refresh_session_model(session_id: str = "", model: str = "", **kwargs: Any) -> None:
    """pre_llm_call: refresh the ACTIVE session model every turn.

    on_session_start only fires at session creation; a mid-session /model switch
    changes agent.model WITHOUT re-firing it. pre_llm_call fires every turn with
    the current model, so this keeps the capture in sync so a live /model swap
    cascades to the executor immediately. Returns None (observer; no context
    injection)."""
    if not session_id or not model:
        return None
    _SESSION_MODELS[session_id] = model
    if len(_SESSION_MODELS) > _MAX_SESSION_MODELS:
        for _ in range(len(_SESSION_MODELS) - _MAX_SESSION_MODELS):
            _SESSION_MODELS.pop(next(iter(_SESSION_MODELS)), None)
    return None


def get_active_session_model(session_id: str = "") -> str:
    """Return the ACTIVE model for a session (from on_session_start capture)."""
    if not session_id:
        return ""
    return _SESSION_MODELS.get(session_id, "")

# Dual-namespace import so the plugin works under both the editable-install
# (`plugins.self_router`) and production (`hermes_plugins.self_router`) layouts.
_config_mod = None
_router_mod = None
for _name in ("hermes_plugins.self_router.config", "plugins.self_router.config"):
    try:
        _config_mod = __import__(_name, fromlist=["*"])
        break
    except ImportError:
        continue
for _name in ("hermes_plugins.self_router.router", "plugins.self_router.router"):
    try:
        _router_mod = __import__(_name, fromlist=["*"])
        break
    except ImportError:
        continue

if _config_mod is None or _router_mod is None:
    logger.warning("self-router: could not import config/router modules; plugin inert")


def _pre_tool_call(tool_name: str = "", args: Optional[dict] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """pre_tool_call hook: recommend a route before the first mutating call.

    Returns None (inert) unless the router says to recommend/dispatch. When it
    recommends, returns a ``warn`` decision so the user sees the suggestion
    without the call being blocked.
    """
    if _router_mod is None or _config_mod is None:
        return None
    try:
        router = _router_mod.get_router()
        if not router.should_check(tool_name):
            return None
        task_desc = tool_name
        if isinstance(args, dict):
            for k in ("path", "command", "url", "goal"):
                v = args.get(k)
                if isinstance(v, str) and v:
                    task_desc = f"{tool_name}: {v[:200]}"
                    break
        task_signature = _router_mod.anchoring.build_task_signature(tool_name, args)
        closest_match = router._store.closest_match(task_signature)
        decision = router.maybe_route(
            task=task_desc,
            closest_match=closest_match,
            task_signature=task_signature,
            description=task_desc,
            session_id=kwargs.get("session_id", ""),
        )
        if decision["decision"] in ("dispatch", "recommend"):
            return {
                "decision": "warn",
                "reason": decision["reason"],
                "snapshot": {
                    "self_router": True,
                    "target": decision.get("harness", ""),
                    "mode": decision["decision"],
                },
            }
        return None
    except Exception as exc:  # never let the router break a tool call
        logger.warning("self-router pre_tool_call failed: %s", exc)
        return None


def register(ctx) -> None:
    """Register the pre_tool_call hook. Without this, the plugin is dead."""
    register_hook = getattr(ctx, "register_hook", None)
    if not callable(register_hook):
        logger.warning(
            "self-router: ctx has no register_hook (type=%s); plugin inert",
            type(ctx).__name__,
        )
        return
    register_hook("pre_tool_call", _pre_tool_call)
    register_hook("on_session_start", _record_session_model)
    register_hook("pre_llm_call", _refresh_session_model)
    logger.info("self-router registered")