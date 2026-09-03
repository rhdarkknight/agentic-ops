"""Hierarchical Memory (LTM) Plugin
===================================

Maintains a two-tier memory system:
- Working summary: current task state (updated every compression)
- Long-term memory: stable facts (refreshed every N compactions)

LTM is extracted from compression summaries and injected into every turn.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .markers import find_compaction_summary

logger = logging.getLogger(__name__)

LTM_FILE = Path.home() / ".hermes" / "ltm_state.json"


def _refresh_interval() -> int:
    """Parse HERMES_CC_LTM_REFRESH_INTERVAL defensively (never crash import)."""
    raw = os.environ.get("HERMES_CC_LTM_REFRESH_INTERVAL", "")
    if not raw:
        return 4
    try:
        val = int(raw)
        return val if val > 0 else 1
    except ValueError:
        logger.warning("LTM: invalid HERMES_CC_LTM_REFRESH_INTERVAL=%r, using 4", raw)
        return 4


LTM_REFRESH_INTERVAL = _refresh_interval()

# Sections to extract as long-term memory — MUST match the production summary
# template headings (agent/context_compressor.py:1574-1617, 1757-1806).
LTM_SECTIONS = [
    "## Constraints & Preferences",
    "## Key Decisions",
    "## Completed Actions",
    "## Active State",
    "## Blocked",
    "## Critical Context",
    "## Goal",
    "## Relevant Files",
]


class LTMManager:
    """Manages long-term memory across compression cycles."""

    def __init__(self):
        self._long_term_memory: Optional[str] = None
        self._compressions_since_refresh = 0
        self._working_summary: Optional[str] = None
        self._last_processed_summary: str = ""

    def on_session_start(self, session_id: str = "", **kwargs):
        """Load LTM from disk on session start."""
        if LTM_FILE.exists():
            try:
                data = json.loads(LTM_FILE.read_text())
                self._long_term_memory = data.get("ltm")
                self._compressions_since_refresh = data.get("compressions_since_refresh", 0)
                logger.debug("Loaded LTM from disk (%d chars)", len(self._long_term_memory or ""))
            except Exception as e:
                logger.warning("Failed to load LTM: %s", e)

    def on_session_end(self, **kwargs):
        """Save LTM state to disk on session end.

        NOTE: the core fires the plugin on_session_end hook at the end of
        EVERY turn (turn_finalizer). Skip the write when nothing changed to
        avoid per-turn disk churn.
        """
        try:
            data = {
                "ltm": self._long_term_memory,
                "compressions_since_refresh": self._compressions_since_refresh,
            }
            new_blob = json.dumps(data, indent=2)
            if LTM_FILE.exists() and LTM_FILE.read_text() == new_blob:
                return
            LTM_FILE.write_text(new_blob)
        except Exception as e:
            logger.warning("Failed to save LTM: %s", e)

    def post_compress(self, ctx, session_id, original_messages, compressed_messages, summary_text, compression_count):
        """Extract or refresh LTM from compression summary.

        DEPRECATED (2026-08-19): the post_compress hook does not exist in
        VALID_HOOKS. Kept for backward compatibility; the plugin now uses
        post_llm_call() via the valid post_llm_call hook.
        """
        if not summary_text:
            return

        self._working_summary = summary_text
        self._compressions_since_refresh += 1

        # Refresh LTM every N compressions
        if self._compressions_since_refresh >= LTM_REFRESH_INTERVAL:
            self._refresh_ltm(summary_text)
            self._compressions_since_refresh = 0

    def post_llm_call(self, session_id: str = "", conversation_history=None, **kwargs):
        """Extract or refresh LTM from a compaction summary in the conversation.

        Called from the valid post_llm_call hook. Scans the conversation for
        the compaction summary marker and refreshes LTM from it. Dedupes on the
        last-processed summary so the persistent summary (present on every
        turn) does not advance the refresh counter or rewrite state each turn.
        """
        summary_text = find_compaction_summary(conversation_history or [])
        if not summary_text:
            return
        if summary_text == self._last_processed_summary:
            return
        self._last_processed_summary = summary_text

        self._working_summary = summary_text
        self._compressions_since_refresh += 1

        if self._compressions_since_refresh >= LTM_REFRESH_INTERVAL:
            self._refresh_ltm(summary_text)
            self._compressions_since_refresh = 0

    def pre_llm_call(self, session_id: str = "", conversation_history=None,
                     model: str = "", **kwargs):
        """Inject LTM into the user message context."""
        if not self._long_term_memory:
            return ""

        return f"\n\n## Long-Term Context (from previous compressions)\n{self._long_term_memory}\n"

    def _refresh_ltm(self, summary_text: str):
        """Extract stable sections from summary to form long-term memory.

        MERGE section-by-section: a section the latest summary omits keeps its
        previously stored content (never silently erases older LTM). The
        summary prefix + end marker are stripped first so directive text
        quoted inside them can never land in LTM.
        """
        from .markers import strip_summary_prefix

        body = strip_summary_prefix(summary_text)

        # Parse existing LTM into per-section dict (header -> body)
        existing: Dict[str, str] = {}
        if self._long_term_memory:
            blocks = re.finditer(r"(## [^\n]+)\n(.*?)(?=\n## |\Z)", self._long_term_memory, re.DOTALL)
            for m in blocks:
                existing[m.group(1).strip()] = m.group(2).strip()

        for section_header in LTM_SECTIONS:
            pattern = re.escape(section_header) + r"\n(.*?)(?=\n##|\Z)"
            match = re.search(pattern, body, re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                if section_content:
                    existing[section_header] = section_content

        if existing:
            self._long_term_memory = "\n\n".join(
                f"{h}\n{b}" for h, b in existing.items() if b
            )
            logger.info("Refreshed LTM (%d sections, %d chars)", len(existing), len(self._long_term_memory or ""))
