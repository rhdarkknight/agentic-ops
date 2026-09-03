"""Loop-state persistence — the "note on the desk at nine."

A structured ``~/.hermes/loop-state/STATUS.json`` written at every
``pre_exit_verify`` call and read at session start. Implements PDF 1's
"state is the only thing the next run inherits" principle: the loop's
handoff state lives on disk, survives session reset, and doubles as a
dead-man heartbeat (a cron can detect "no update in N hours → stuck").

State is **advisory only** — never sole grounds for an approval. The
gate's approval logic still derives from message-history prose evidence
and the session-scoped hint cache. This file is the cross-session
*memory* of what happened, not a source of truth for what to do next.

Schema (v1):
    {
      "version": 1,
      "session_id": str,
      "task": str | null,           # best-effort: latest user message
      "build_count": int,           # cumulative mutating tool calls this session
      "review_pass_count": int,     # highest review pass seen this session
      "last_review_clean": bool,    # was the last review clean?
      "open_blockers": int,         # from the last parsed review
      "open_majors": int,
      "consecutive_clean": int,     # consecutive clean passes at last write
      "turn_count": int,            # total tool calls this session (for step cap)
      "circuit_breaker_tripped": bool,
      "last_heartbeat": str,        # ISO-8601 UTC timestamp of this write
      "last_exit_verdict": str,     # "approved" | "rejected" | "stalled" | "cap"
      "single_safe_next_action": str | null,
      "off_limits": list[str],      # things the loop must never touch
    }
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VERSION = 1
_DEFAULT_DIR = Path.home() / ".hermes" / "loop-state"
_DEFAULT_PATH = _DEFAULT_DIR / "STATUS.json"


def _state_path() -> Path:
    """Resolve the state file path, honoring an env override for tests."""
    override = os.environ.get("HERMES_LOOP_STATE_FILE")
    if override:
        return Path(override)
    return _DEFAULT_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_state() -> Optional[Dict[str, Any]]:
    """Read the persisted loop state. Returns None if absent or corrupt.

    Advisory only — callers must never approve solely on this state.
    """
    path = _state_path()
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("loop-state read failed: %s", exc)
        return None


def _snapshot_session_id(snapshot: Dict[str, Any]) -> str:
    """Return the durable identity used to decide whether snapshots may merge.

    The empty string is the legacy representation for an anonymous caller.
    It is intentionally distinct from every named session, so an anonymous
    write can neither inherit nor be inherited by a named session snapshot.
    """
    value = snapshot.get("session_id")
    return str(value) if value else ""


def write_state(snapshot: Dict[str, Any]) -> None:
    """Write a session-isolated snapshot into the persisted loop state.

    Numeric fields climb monotonically only when the existing file belongs to
    the same session. A different session (including named versus anonymous)
    starts from the supplied snapshot, preventing stale counters from leaking
    across conversations. A ``last_heartbeat`` timestamp is always stamped.
    Failure is best-effort: a write error is logged and swallowed — the gate
    must never block on a state-file write.
    """
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_state() or {}
        current = existing if _snapshot_session_id(existing) == _snapshot_session_id(snapshot) else {}
        # Merge: numeric fields climb monotonically within one session only;
        # other fields remain latest-wins.
        for key, value in snapshot.items():
            if isinstance(value, (int, float)) and isinstance(current.get(key), (int, float)):
                current[key] = max(current[key], value) if value >= 0 else value
            else:
                current[key] = value
        current["version"] = _VERSION
        current["last_heartbeat"] = _now_iso()
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("loop-state write failed: %s", exc)


def clear_state(session_id: Optional[str] = None) -> bool:
    """Clear persisted state, optionally only when it belongs to ``session_id``.

    The no-argument form preserves the original administrative API. Lifecycle
    callers must pass the finalized/reset session ID so they cannot delete a
    newer conversation's singleton STATUS.json snapshot. Returns whether a
    file was removed.
    """
    path = _state_path()
    try:
        if not path.exists():
            return False
        if session_id is not None:
            current = read_state()
            if not current or _snapshot_session_id(current) != str(session_id):
                return False
        path.unlink()
        return True
    except OSError as exc:
        logger.debug("loop-state clear failed: %s", exc)
        return False


def heartbeat_age_seconds() -> Optional[float]:
    """Return seconds since the last heartbeat, or None if no state."""
    state = read_state()
    if not state or not state.get("last_heartbeat"):
        return None
    try:
        last = datetime.fromisoformat(state["last_heartbeat"])
        return (datetime.now(timezone.utc) - last).total_seconds()
    except (TypeError, ValueError):
        return None