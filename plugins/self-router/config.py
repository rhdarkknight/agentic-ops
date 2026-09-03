"""Self-Router configuration.

Reads `self_router` from `~/.hermes/config.yaml` with safe defaults, plus
env overrides for testability. The dispatch layer is gated on the verified
executor set (`executors`); an empty set means the router recommends but
never auto-dispatches.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    # Specialist must beat keep_self by this margin to win.
    "win_margin": 0.15,
    # Recall similarity below this = "no prior art" (novelty flag).
    "novelty_threshold": 0.6,
    # Anchoring risk above this (with no-prior-art) triggers stateless route.
    "anchoring_risk_threshold": 0.7,
    # Stateless harness for genuinely novel tasks.
    "route_novel_to": "codex",
    # Cascade the executor's model from the ACTIVE session (non-negotiable):
    # personal/professional inference costs must never overlap. When true, the
    # router writes the active provider/model (mapped to litellm) before dispatch.
    "model_cascade": True,
    # Task-type -> specialist harness map.
    "route_specialist_map": {
        "narrow_coding": "claude_code",
        "security_review": "codex",
        "sandboxed_execution": "codex",
        "second_opinion": "opencode",
    },
    # VERIFIED executor set (Phase 0 probe). Empty = recommend-only.
    # Populated by the executor probe; never route to an unverified agent.
    "executors": [],
    "counterfactual_cron": True,
    "trigger": {
        "hook": "pre_tool_call",
        # pre_tool_call fires on EVERY tool call — gate it.
        "cooldown_sec": 30,
        # Default: RECOMMEND a route, don't auto-dispatch.
        "require_user_confirmation": True,
    },
    "auto_dispatch": False,
}

# Paths that never trigger routing (read-only).
READ_ONLY_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "browser_snapshot", "browser_navigate", "skill_view", "skills_list",
    "session_search", "hindsight_recall", "hindsight_reflect",
    "mcp_codebase_memory_search_graph", "mcp_codebase_memory_query_graph",
    "mcp_codebase_memory_get_code_snippet", "mcp_codebase_memory_get_architecture",
})

# First-mutating-call tools that SHOULD trigger routing consideration.
MUTATING_TOOLS = frozenset({
    "write_file", "patch", "terminal", "execute_code", "delegate_task",
    "skill_manage", "memory", "hindsight_retain", "process",
})


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _load_file_config() -> Dict[str, Any]:
    """Read the user's `self_router` block from `~/.hermes/config.yaml`.

    Returns {} if unreadable or absent. Uses a dynamic import so the plugin
    works under both the editable-install and production namespaces.
    """
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "config.yaml"
        if not cfg_path.is_file():
            return {}
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        block = data.get("self_router") or {}
        return block if isinstance(block, dict) else {}
    except Exception as exc:  # never let a config read crash the router
        logger.warning("self-router: could not read config.yaml self_router block: %s", exc)
        return {}


def load_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load self_router config: deep-merge defaults, then the user's
    `~/.hermes/config.yaml → self_router` block, then env overrides, then
    explicit overrides (highest precedence)."""
    cfg = _deep_merge(DEFAULT_CONFIG, {})

    # Read the user's self_router block from config.yaml (the source of truth
    # for runtime values like executors / auto_dispatch).
    file_cfg = _load_file_config()
    if file_cfg:
        cfg = _deep_merge(cfg, file_cfg)

    cfg = _deep_merge(cfg, overrides or {})

    # Env overrides (testability + runtime toggles).
    cfg["enabled"] = _truthy(os.environ.get("HERMES_SELF_ROUTER_ENABLED", None)) or cfg["enabled"]
    cfg["win_margin"] = _float_env("HERMES_SELF_ROUTER_WIN_MARGIN", cfg["win_margin"])
    cfg["novelty_threshold"] = _float_env("HERMES_SELF_ROUTER_NOVELTY_THRESHOLD", cfg["novelty_threshold"])
    cfg["anchoring_risk_threshold"] = _float_env("HERMES_SELF_ROUTER_ANCHORING_RISK_THRESHOLD", cfg["anchoring_risk_threshold"])
    auto = os.environ.get("HERMES_SELF_ROUTER_AUTO_DISPATCH")
    if auto is not None:
        cfg["auto_dispatch"] = _truthy(auto)
    cd = os.environ.get("HERMES_SELF_ROUTER_COOLDOWN_SEC")
    if cd is not None:
        cfg["trigger"]["cooldown_sec"] = _int_env("HERMES_SELF_ROUTER_COOLDOWN_SEC", cfg["trigger"]["cooldown_sec"])
    req = os.environ.get("HERMES_SELF_ROUTER_REQUIRE_CONFIRMATION")
    if req is not None:
        cfg["trigger"]["require_user_confirmation"] = _truthy(req)

    # Sanity clamps.
    if not isinstance(cfg["win_margin"], (int, float)) or cfg["win_margin"] < 0:
        cfg["win_margin"] = 0.15
    if not isinstance(cfg["executors"], list):
        cfg["executors"] = []

    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out