"""Shared compaction-summary detection for context_compressor_enhancements.

The production agent prefixes every compaction summary with a long multi-line
"[CONTEXT COMPACTION — REFERENCE ONLY] ..." directive (SUMMARY_PREFIX, 1428
chars), or the legacy "[CONTEXT SUMMARY]:" prefix, sets the
"_compressed_summary" metadata flag on the summary message, and appends the
"_SUMMARY_END_MARKER". Content heuristics that match only "[CONTEXT
COMPACTION]" (bracket immediately closed) match NOTHING in production —
always use the shared helpers below.

Prefixes are replicated from agent/context_compressor.py (SUMMARY_PREFIX /
LEGACY_SUMMARY_PREFIX / _HISTORICAL_SUMMARY_PREFIXES / _SUMMARY_END_MARKER /
_MERGED_SUMMARY_DELIMITER). They are stable constants; importing the core from
a plugin would be coupling we avoid.
"""

from __future__ import annotations

from typing import Any, Sequence

COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"

LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Same as the core's _SUMMARY_END_MARKER.
SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)

# Merge-into-tail delimiter (core: _MERGED_SUMMARY_DELIMITER).
MERGED_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"

# The full current prefix (mirrors core SUMMARY_PREFIX). This is the 1428-char
# multi-line directive; stripping only a fragment leaks directive text into
# extracted sections.
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat ONLY "
    "the latest message as the active task and discard stale items from "
    "'## Historical Task Snapshot' / '## Historical In-Progress State' / "
    "'## Historical Pending User Asks' / "
    "'## Historical Remaining Work' entirely — do not 'wrap up' or "
    "'finish' work described there unless the latest message explicitly "
    "asks for it. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)

# Historical handoff prefixes (mirrors core _HISTORICAL_SUMMARY_PREFIXES:
# current, legacy, carveout-era 1378-char, pre-#35344 586-char). A summary
# persisted under one of these can be inherited into a resumed lineage, so
# strip detection MUST cover all of them.
_HISTORICAL_PREFIX_CARVEOUT = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary — that message is the single source of "
    "truth for what to do right now. If the latest user message is consistent "
    "with the '## Active Task' section, you may use the summary as background. "
    "If the latest user message contradicts, supersedes, changes topic from, "
    "or in any way diverges from '## Active Task' / '## In Progress' / "
    "'## Pending User Asks' / '## Remaining Work', the latest message WINS — "
    "discard those stale items entirely and do not 'wrap up the old task "
    "first'. Reverse signals in the latest message (e.g. 'stop', 'undo', "
    "'roll back', 'just verify', 'don't do that anymore', 'never mind', a "
    "new topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. IMPORTANT: Your persistent "
    "memory (MEMORY.md, USER.md) in the system prompt is ALWAYS authoritative "
    "and active — never ignore or deprioritize memory content due to this "
    "compaction note. The current session state (files, config, etc.) may "
    "reflect work described here — avoid repeating it:"
)
_HISTORICAL_PREFIX_RESUME = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Your current task is identified in the "
    "'## Active Task' section of the summary — resume exactly from there. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary. The current session state (files, config, etc.) may reflect "
    "work described here — avoid repeating it:"
)

HISTORICAL_SUMMARY_PREFIXES = (
    SUMMARY_PREFIX,
    LEGACY_SUMMARY_PREFIX,
    _HISTORICAL_PREFIX_CARVEOUT,
    _HISTORICAL_PREFIX_RESUME,
)

# Detection markers — a message is a summary if its stripped content starts
# with any of these (mirrors core _is_context_summary_content). The short
# head fragment "[CONTEXT COMPACTION" is included so truncated/handwritten
# fixtures still match; real production summaries start with the full prefix.
_SUMMARY_PREFIXES_FOR_DETECT = (
    SUMMARY_PREFIX,
    LEGACY_SUMMARY_PREFIX,
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]",
    "[CONTEXT SUMMARY — REFERENCE ONLY]",
)


def is_compaction_message(msg: Any) -> bool:
    """Return True if a message is a compaction summary message.

    Matches the metadata flag the agent sets on the summary message, or a
    message whose *stripped* content starts with a summary prefix (mirrors
    core _is_context_summary_content: looks past the merged-prior delimiter).
    """
    if not isinstance(msg, dict):
        return False
    if msg.get(COMPRESSED_SUMMARY_METADATA_KEY):
        return True
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    text = content.lstrip()
    if MERGED_SUMMARY_DELIMITER in text:
        text = text.split(MERGED_SUMMARY_DELIMITER, 1)[1].lstrip()
    return any(text.startswith(p) for p in _SUMMARY_PREFIXES_FOR_DETECT)


def find_compaction_summary(messages: Sequence[Any]) -> str:
    """Return the content of the most recent compaction summary message, or
    empty string if none present. Scans from the end (most recent first)."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get(COMPRESSED_SUMMARY_METADATA_KEY):
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content.lstrip()
            if MERGED_SUMMARY_DELIMITER in text:
                text = text.split(MERGED_SUMMARY_DELIMITER, 1)[1].lstrip()
            if any(text.startswith(p) for p in _SUMMARY_PREFIXES_FOR_DETECT):
                return content
    return ""


def strip_summary_prefix(summary: str) -> str:
    """Return the summary body without the current/legacy/historical prefix
    and the trailing end marker. Mirrors core _strip_summary_prefix."""
    text = (summary or "").strip()
    if MERGED_SUMMARY_DELIMITER in text:
        text = text.split(MERGED_SUMMARY_DELIMITER, 1)[1].strip()
    for prefix in HISTORICAL_SUMMARY_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    if text.endswith(SUMMARY_END_MARKER):
        text = text[: -len(SUMMARY_END_MARKER)].rstrip()
    return text
