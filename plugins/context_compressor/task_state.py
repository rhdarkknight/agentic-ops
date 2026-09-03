"""
Task State Persistence
======================

Extracts the Task State Persistence enhancement from context_compressor.py
into a standalone plugin module.

This module replaces the inline _load_task_state(), _save_task_state(),
and _extract_task_state_from_summary() methods of ContextCompressor.

Features:
- Loads/saves persistent task state to/from a JSON file
- Extracts structured task information (Goal, Done, Next Steps, Files)
  from structured summary text using section-heading parsing
- Provides a get_context() method that formats state for injection
  into the user message via the pre_llm_call hook

The task state survives across sessions and gives the agent continuity
when resuming work after a session ends.

Config: HERMES_CC_TASK_STATE_FILE (default: ~/.hermes/task_state.json)

Usage:
    from plugins.context_compressor.task_state import TaskStateManager
    mgr = TaskStateManager()
    mgr.load()
    mgr.save()
    ctx = mgr.get_context()   # formatted for user message injection
    mgr.update_from_summary(summary_text)  # called after compression
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum summary length to attempt extraction
_MIN_SUMMARY_CHARS = 200

# Directive-style bracketed injections the gateway or plugins add to the user
# message (e.g. [System note: ...] for resume continuity, [CAVEMAN FULL: ...]
# from caveman_enforcer, [Role: ...] from role-system, [IMPORTANT: ...] for
# cron delivery). The compressor summarizer can lift these from a prior turn
# and save them as the persistent "goal" — which then gets re-injected into
# every future user message via get_context(). Strip them at extraction time
# so the cross-session state file never carries directive text.
_INJECTION_PATTERNS = [
    re.compile(r"^\[System note:[^\]]*\]", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[CAVEMAN[^\]]*:[^\]]*\]", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[Role:[^\]]*\]", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[Active Task[^\]]*\]", re.MULTILINE | re.DOTALL),
    # Cron context wrapper is the trap: it contains NESTED `[SILENT]` brackets
    # inside the outer [IMPORTANT: ...] block, so a `[^\]]*\]` pattern
    # terminates at the first inner `]` and leaves the tail ("suppress
    # delivery. Never combine [SILENT]...") in the goal field. Two patterns:
    # the long form matches the canonical cron wrapper end-marker
    # (`nothing more.]`); the short form matches a truncated wrapper that
    # ends with `and no` (sometimes the prompt template is cut off).
    re.compile(
        r"^\[IMPORTANT:.*?nothing more\.\]",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"^\[IMPORTANT:.*?and no\b[^\]]*\]",
        re.MULTILINE | re.DOTALL,
    ),
    # Fallback for any other cron-style [IMPORTANT: ...] block — matches up
    # to the first balanced close using a single-level nested-bracket pattern.
    re.compile(
        r"^\[IMPORTANT:[^\]]*(?:\[[^\]]*\][^\]]*)*\]",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(r"^\[SILENT\][^\n]*", re.MULTILINE),
    re.compile(r"^## Active Task.*?(?=\n## |\Z)", re.MULTILINE | re.DOTALL),
]


def _strip_injections(text: str) -> str:
    """Remove directive-style bracketed injections from extracted state.

    Conservative: only strips bracketed text that occupies a line by itself
    (the canonical shape for these directives). Real user content with
    bracket tokens in the middle of sentences is preserved.
    """
    if not text:
        return text
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TaskStateManager:
    """Manages persistent task state across sessions.

    The task state is a lightweight dict extracted from structured
    summaries after context compression. It survives across sessions
    and is injected as context at the start of each turn.

    State schema:
        {
            "goal": "What the user is trying to accomplish",
            "done": ["completed item 1", "completed item 2", ...],
            "next_steps": ["item 1", "item 2", ...],
            "relevant_files": ["file1.py", "file2.yaml", ...],
            "last_updated": "<ISO timestamp>",
            "session_id": "<last session ID>"
        }
    """

    def __init__(self, path: str | None = None):
        """Initialize the TaskStateManager.

        Args:
            path: Path to the JSON state file.
                  Default: ~/.hermes/task_state.json
                  Override with HERMES_CC_TASK_STATE_FILE env var.
        """
        env_path = os.environ.get("HERMES_CC_TASK_STATE_FILE", "")
        if env_path:
            self._path = Path(os.path.expanduser(env_path))
        elif path:
            self._path = Path(os.path.expanduser(path))
        else:
            self._path = Path("~/.hermes/task_state.json").expanduser()

        self.state: Dict[str, Any] = {}
        self._dirty = False  # True when state has been modified since last save

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self) -> None:
        """Load task state from disk. Idempotent (no-op if file missing)."""
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        self.state = data
                        self._dirty = False
                        logger.debug("TaskState: loaded from %s", self._path)
                    else:
                        logger.warning("TaskState: unexpected data type in %s", self._path)
        except json.JSONDecodeError as e:
            logger.warning("TaskState: could not parse %s: %s", self._path, e)
        except OSError as e:
            logger.debug("TaskState: could not load %s: %s", self._path, e)

    def save(self) -> None:
        """Save current task state to disk. Atomic write (tmp + rename)."""
        if not self._dirty:
            logger.debug("TaskState: skip save (no changes)")
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(self._path)
            self._dirty = False
            logger.debug("TaskState: saved %s (%d bytes)", self._path, self._path.stat().st_size)
        except OSError as e:
            logger.warning("TaskState: could not save to %s: %s", self._path, e)

    def is_empty(self) -> bool:
        """Return True if no task state is loaded/saved yet."""
        return not bool(self.state.get("goal") or self.state.get("done") or self.state.get("next_steps"))

    def get_context(self) -> str:
        """Format current task state for injection into user message.

        Returns empty string if no state is available.

        Format:
            ## Task State (from previous session)
            [goal, done, next steps, files]
        """
        if self.is_empty():
            return ""

        parts: list[str] = []
        # Defensive: strip any directive injections that may have landed
        # in goal from a prior cycle before re-injecting. This is the
        # second line of defense — the primary filter is at extraction
        # time in _extract_task_state_from_summary, but this catches
        # any state that was written before the filter was added.
        goal = _strip_injections(self.state.get("goal", "")).strip()
        if goal:
            parts.append(f"**Goal:** {goal}")

        done = self.state.get("done", [])
        if done:
            done_items = "\n".join(f"  - {d}" for d in done[:10])  # cap at 10
            parts.append(f"**Done:**\n{done_items}")

        next_steps = self.state.get("next_steps", [])
        if next_steps:
            steps_items = "\n".join(f"  - {s}" for s in next_steps[:10])
            parts.append(f"**Next Steps:**\n{steps_items}")

        files = self.state.get("relevant_files", [])
        if files:
            file_list = ", ".join(files[:15])  # cap at 15 files
            parts.append(f"**Relevant Files:** {file_list}")

        if not parts:
            return ""

        last_updated = self.state.get("last_updated", "")
        header = "## Task State (from previous session)"
        if last_updated:
            header += f" (updated {last_updated})"

        return header + "\n" + "\n".join(parts)

    def update_from_summary(self, summary: str, session_id: str = "") -> bool:
        """Extract and update task state from a structured summary.

        Parses the summary text for ## Goal, ## Progress, ## Next Steps,
        and ## Relevant Files sections, then updates self.state.

        Args:
            summary: The structured summary text (from compression).
            session_id: Optional session ID to store alongside state.

        Returns:
            True if state was updated, False if summary was too short
            or parsing failed.
        """
        if not summary or len(summary) < _MIN_SUMMARY_CHARS:
            return False

        new_state = _extract_task_state_from_summary(summary)
        if not new_state.get("goal") and not new_state.get("done") and not new_state.get("next_steps"):
            logger.debug("TaskState: no extractable state from summary (%.0f chars)", len(summary))
            return False

        import datetime
        new_state["last_updated"] = datetime.datetime.now().isoformat()
        if session_id:
            new_state["session_id"] = session_id

        # Merge: keep existing done items that aren't in the new state
        # (allows accumulated done history across multiple compactions)
        existing_done = set(self.state.get("done", []))
        new_done = new_state.get("done", [])
        merged_done = _merge_list(existing_done, set(new_done))

        self.state = new_state
        self.state["done"] = merged_done
        self._dirty = True
        logger.debug("TaskState: updated from summary (%s, %d done items)", session_id or "?", len(merged_done))
        return True

    def mark_dirty(self) -> None:
        """Mark state as modified so next save() writes to disk."""
        self._dirty = True

    def clear(self) -> None:
        """Clear all task state."""
        self.state = {}
        self._dirty = True


# ---------------------------------------------------------------------------
# Internal extraction logic (ported from ContextCompressor)
# ---------------------------------------------------------------------------

def _extract_task_state_from_summary(summary: str) -> dict:
    """Extract structured task information from a summary string.

    Parses the structured summary sections into a lightweight task
    state dict that survives across sessions.

    The compaction summary carries the SUMMARY_PREFIX directive header and a
    trailing SUMMARY_END_MARKER; both are stripped before parsing so quoted
    heading text inside the prefix cannot pollute extracted sections.

    Returns:
        dict with keys: goal, done, next_steps, relevant_files
    """
    try:
        from .markers import SUMMARY_END_MARKER, strip_summary_prefix
    except ImportError:  # standalone test import (no package context)
        from markers import SUMMARY_END_MARKER, strip_summary_prefix

    summary = strip_summary_prefix(summary)

    state: dict = {
        "goal": "",
        "done": [],
        "next_steps": [],
        "relevant_files": [],
    }

    def _extract_section(text: str, heading: str) -> str:
        marker = f"## {heading}"
        idx = text.find(marker)
        if idx == -1:
            return ""
        rest = text[idx + len(marker):]
        # Stop at the next ## heading
        next_heading = rest.find("\n## ")
        if next_heading != -1:
            rest = rest[:next_heading]
        # Also stop at the summary end marker (production)
        end_idx = rest.find(SUMMARY_END_MARKER)
        if end_idx != -1:
            rest = rest[:end_idx]
        # Also stop at [Summary vN | hash:...] metadata (legacy)
        meta_marker = "[Summary v"
        meta_idx = rest.find(meta_marker)
        if meta_idx != -1:
            rest = rest[:meta_idx]
        return rest.strip()

    # --- Goal ---
    # Production uses "## Historical Task Snapshot" (the compacted task) and
    # the live summary template's "## Goal". Prefer Goal, fall back to the
    # historical snapshot, then to "## Active Task" (the template's field).
    goal = _extract_section(summary, "Goal")
    if not goal:
        goal = _extract_section(summary, "Historical Task Snapshot")
    if not goal:
        goal = _extract_section(summary, "Active Task")
    if goal:
        # Take first non-blank line after stripping any captured directive
        # injections (e.g. [System note: ...], [CAVEMAN FULL: ...]).
        cleaned = _strip_injections(goal)
        first = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), "")
        state["goal"] = first

    # --- Progress / Done ---
    # Production uses "## Completed Actions" with a numbered list
    # ("1. ACTION target — outcome [tool: name]").
    progress_text = _extract_section(summary, "Completed Actions")
    if not progress_text:
        progress_text = _extract_section(summary, "Progress")
    if not progress_text:
        progress_text = _extract_section(summary, "Done")

    if progress_text:
        for line in progress_text.splitlines():
            stripped = line.strip()
            # Numbered-list item: "1. READ config.py:45 — found X [tool: read_file]"
            m = re.match(r"^\d+[.)]\s+(.+)$", stripped)
            if m:
                state["done"].append(_strip_injections(m.group(1).strip()))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                state["done"].append(_strip_injections(stripped[2:].strip()))
            elif stripped and not stripped.startswith("### "):
                # Fallback: non-empty line that isn't a sub-heading
                state["done"].append(_strip_injections(stripped))
        # Cap done list
        state["done"] = [d for d in state["done"] if d][:20]

    # --- Next Steps ---
    # Production uses "## Historical Remaining Work" (stale) and the live
    # template's "## Historical In-Progress State" / "## Historical Pending
    # User Asks". Also accept the legacy "## Next Steps".
    next_text = _extract_section(summary, "Next Steps")
    if not next_text:
        next_text = _extract_section(summary, "Historical Remaining Work")
    if not next_text:
        next_text = _extract_section(summary, "Historical In-Progress State")
    if next_text:
        for line in next_text.splitlines():
            stripped = line.strip().replace("- ", "").replace("* ", "")
            m = re.match(r"^\d+[.)]\s+(.+)$", stripped)
            if m:
                stripped = m.group(1).strip()
            if stripped:
                state["next_steps"].append(_strip_injections(stripped))
        state["next_steps"] = [s for s in state["next_steps"] if s][:20]

    # --- Relevant Files ---
    files_text = _extract_section(summary, "Relevant Files")
    if files_text:
        for line in files_text.splitlines():
            stripped = line.strip().replace("- ", "").replace("* ", "")
            if stripped:
                state["relevant_files"].append(stripped)

    return state


def _merge_list(existing: set, new: set, max_size: int = 50) -> list:
    """Merge two sets of done items, preserving order and capping size.

    New items are prepended before existing ones to show recency.
    Total capped at max_size to prevent unbounded growth.
    """
    merged_list: list[str] = []
    seen: set[str] = set()

    # New items first (most recent)
    for item in reversed(list(new)):
        if item and item not in seen:
            merged_list.insert(0, item)
            seen.add(item)

    # Then existing items not in new
    for item in existing:
        if item and item not in seen and len(merged_list) < max_size:
            merged_list.append(item)
            seen.add(item)

    return merged_list[:max_size]
