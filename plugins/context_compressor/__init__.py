"""
Context Compressor Enhancements Plugin
======================================

Compression enhancement plugins that work alongside the stock
ContextCompressor without modifying core files.

Plugins included:
1. Task State Persistence    — survives across sessions (valid hooks)
2. Summary Versioning         — tracks summary hashes (valid hooks)
3. Hierarchical Memory (LTM)  — long-term memory tier (valid hooks)
4. Salience Scoring           — importance-weighted preservation (pre_llm_call)
5. Structural Preservation    — protects code/config blocks (pre_llm_call)

NOTE (2026-08-19): Quality Guardrails was REMOVED. Its old `post_compress`
hook does not exist in VALID_HOOKS, and post-compaction fact-survival
verification is already handled by the `validated-compaction` plugin (enabled).

All registered hooks are valid per hermes_cli.plugins.VALID_HOOKS.

Hook kwarg contracts (verified against installed source):
  - pre_llm_call:  session_id, task_id, turn_id, user_message,
                   conversation_history, is_first_turn, model, platform,
                   sender_id  → return str appended to user context
  - post_llm_call: session_id, task_id, turn_id, user_message,
                   assistant_response, conversation_history, model, platform
  - on_session_start: session_id, model, platform
  - on_session_end:   (varies by callsite; adapters accept **kwargs)

Config via environment variables:
  HERMES_CC_ENHANCEMENTS_ENABLED     — 0 to disable (default: 1)
  HERMES_CC_LTM_REFRESH_INTERVAL     — LTM refresh every N compactions (default: 4)
  HERMES_CC_TASK_STATE_FILE          — task state path (default: ~/.hermes/task_state.json)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state (survives across plugin instances)
# ---------------------------------------------------------------------------

_enabled = os.environ.get("HERMES_CC_ENHANCEMENTS_ENABLED", "1").lower() in (
    "1", "true", "yes", "on"
)

# Lazy imports — initialized on first use
_task_state_mgr = None
_version_tracker = None
_ltm_manager = None
_salience_scorer = None
_structure_preserver = None

# Shared compaction-summary detection (matches production prefixes).
from .markers import find_compaction_summary  # noqa: E402

# Last summary processed by the task-state / versioning adapters — used to
# avoid re-processing the persistent compaction summary on every turn.
_last_processed_task_summary: str = ""


def _get_task_state_mgr():
    global _task_state_mgr
    if _task_state_mgr is None:
        from .task_state import TaskStateManager
        _task_state_mgr = TaskStateManager()
    return _task_state_mgr


def _get_version_tracker():
    global _version_tracker
    if _version_tracker is None:
        from .summary_versioning import SummaryVersionTracker
        _version_tracker = SummaryVersionTracker()
    return _version_tracker


def _get_ltm_manager():
    global _ltm_manager
    if _ltm_manager is None:
        from .hierarchical_memory import LTMManager
        _ltm_manager = LTMManager()
    return _ltm_manager


def _get_salience_scorer():
    global _salience_scorer
    if _salience_scorer is None:
        from .salience_scoring import SalienceScorer
        _salience_scorer = SalienceScorer()
    return _salience_scorer


def _get_structure_preserver():
    global _structure_preserver
    if _structure_preserver is None:
        from .structural_preservation import StructurePreserver
        _structure_preserver = StructurePreserver()
    return _structure_preserver


# ---------------------------------------------------------------------------
# Hook adapters — wrap the backup module classes with hook-compatible interfaces
# ---------------------------------------------------------------------------

def _task_state_on_session_start(session_id: str = "", **_kwargs: Any) -> None:
    global _last_processed_task_summary
    # A new session may carry a different summary; reset the dedupe so the
    # first summary of this session is processed.
    _last_processed_task_summary = ""
    mgr = _get_task_state_mgr()
    mgr.load()


def _task_state_on_session_end(**_kwargs: Any) -> None:
    mgr = _get_task_state_mgr()
    # Only persist when something changed — on_session_end fires per-turn, so
    # an unconditional save would rewrite task_state.json on every turn.
    if getattr(mgr, "_dirty", False):
        mgr.save()


def _task_state_pre_llm_call(session_id: str = "", conversation_history=None,
                             **_kwargs: Any) -> str:
    mgr = _get_task_state_mgr()
    if mgr.is_empty():
        return ""
    return mgr.get_context()


def _task_state_post_llm_call(session_id: str = "", conversation_history=None,
                              **_kwargs: Any) -> None:
    # Extract summary from messages and update task state. Skip cron
    # sessions entirely: their input is dominated by the
    # [IMPORTANT: cron job ...] wrapper which the LLM summaries treat
    # as "the task" and save to the persistent goal — then it re-injects
    # as the "## Active Task" context in every subsequent turn. Cron work
    # is ephemeral by nature; it should not bleed into long-term state.
    if os.environ.get("HERMES_CRON_SESSION"):
        logger.debug("TaskState: skipping update_from_summary (cron session)")
        return
    try:
        summary = find_compaction_summary(conversation_history or [])
        if not summary:
            return
        global _last_processed_task_summary
        # The compaction summary persists in history on every turn; only
        # re-parse when the summary content actually changed.
        if summary == _last_processed_task_summary:
            return
        _last_processed_task_summary = summary
        mgr = _get_task_state_mgr()
        mgr.update_from_summary(summary, session_id=session_id)
    except Exception:
        logger.exception("task_state: post_llm_call failed")


def _version_post_llm_call(session_id: str = "", conversation_history=None,
                           **_kwargs: Any) -> None:
    try:
        tracker = _get_version_tracker()
        tracker.load()
        tracker.track_from_messages(conversation_history or [])
        if tracker._dirty:
            tracker.save()
    except Exception:
        logger.exception("summary_versioning: post_llm_call failed")


def _version_on_session_end(**_kwargs: Any) -> None:
    tracker = _get_version_tracker()
    tracker.save()


def _version_on_session_start(session_id: str = "", **_kwargs: Any) -> None:
    """Reset the version tracker for a new session.

    get_last_hash() loads from disk and would otherwise inject a PREVIOUS
    session's compression hash on the first turn of a new session. Load fresh
    and clear hashes so the injection only appears after this session's first
    compression.
    """
    try:
        tracker = _get_version_tracker()
        tracker.load()
        tracker._hashes = []
        tracker._dirty = False
    except Exception:
        logger.exception("summary_versioning: on_session_start failed")


def _version_pre_llm_call(session_id: str = "", conversation_history=None,
                          model: str = "", **_kwargs: Any) -> str:
    tracker = _get_version_tracker()
    last_hash = tracker.get_last_hash()
    if last_hash:
        return f"\n\n## Summary Version\nLast compression hash: `{last_hash.get('hash', '')}` (v{tracker.version})\n"
    return ""


def _salience_pre_llm_call(session_id: str = "", conversation_history=None,
                           model: str = "", **_kwargs: Any) -> str:
    """Score turns for salience and return a preservation prompt block.

    Adapts the old `pre_compress` behavior to the valid `pre_llm_call` hook.
    The hook contract returns a string appended to ephemeral context; we
    return a compact directive telling the model which high-salience turns
    (corrections, decisions, file modifications) must survive compression.
    """
    try:
        salience = _get_salience_scorer()
        return salience.score_conversation(conversation_history or [])
    except Exception:
        logger.exception("salience_scoring: pre_llm_call failed")
        return ""


def _structure_pre_llm_call(session_id: str = "", conversation_history=None,
                            model: str = "", **_kwargs: Any) -> str:
    """Tag structured content (code/config blocks) for preservation.

    Adapts the old `pre_compress` structural-preservation behavior to the
    valid `pre_llm_call` hook. Returns a prompt block only when structured
    content was detected, so the model keeps those blocks intact during
    summarization.
    """
    try:
        preserver = _get_structure_preserver()
        return preserver.build_preservation_note(conversation_history or [])
    except Exception:
        logger.exception("structural_preservation: pre_llm_call failed")
        return ""


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register all context compressor enhancement hooks (valid hooks only)."""
    if not _enabled:
        logger.debug("context_compressor_enhancements: disabled via HERMES_CC_ENHANCEMENTS_ENABLED=0")
        return

    logger.info("Registering context_compressor_enhancements plugin (5 sub-plugins)")

    # --- Task State Persistence (valid hooks) ---
    ctx.register_hook("on_session_start", _task_state_on_session_start)
    ctx.register_hook("on_session_end", _task_state_on_session_end)
    ctx.register_hook("pre_llm_call", _task_state_pre_llm_call)
    ctx.register_hook("post_llm_call", _task_state_post_llm_call)
    logger.debug("  Registered task_state plugin")

    # --- Summary Versioning (valid hooks) ---
    ctx.register_hook("post_llm_call", _version_post_llm_call)
    ctx.register_hook("on_session_end", _version_on_session_end)
    ctx.register_hook("on_session_start", _version_on_session_start)
    ctx.register_hook("pre_llm_call", _version_pre_llm_call)
    logger.debug("  Registered summary_versioning plugin")

    # --- Hierarchical Memory / LTM (valid hooks) ---
    ltm_mgr = _get_ltm_manager()
    ctx.register_hook("on_session_start", ltm_mgr.on_session_start)
    ctx.register_hook("on_session_end", ltm_mgr.on_session_end)
    ctx.register_hook("post_llm_call", ltm_mgr.post_llm_call)
    ctx.register_hook("pre_llm_call", ltm_mgr.pre_llm_call)
    logger.debug("  Registered hierarchical_memory plugin")

    # --- Salience Scoring (valid pre_llm_call hook) ---
    ctx.register_hook("pre_llm_call", _salience_pre_llm_call)
    logger.debug("  Registered salience_scoring plugin")

    # --- Structural Preservation (valid pre_llm_call hook) ---
    ctx.register_hook("pre_llm_call", _structure_pre_llm_call)
    logger.debug("  Registered structural_preservation plugin")

    logger.info("All context_compressor enhancements registered successfully")
