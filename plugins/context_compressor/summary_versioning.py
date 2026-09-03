"""
Summary Versioning
=================

Extracts the Summary Versioning enhancement from context_compressor.py
into a standalone plugin module.

Features:
- Computes SHA256 hashes of summary content for change detection
- Tracks summary history across compactions
- Provides version metadata for injection
- Integrates with TaskStateManager for cross-session continuity

This replaces the inline _format_summary_for_output() and _summary_hashes
tracking in ContextCompressor.

Usage:
    from plugins.context_compressor.summary_versioning import SummaryVersionTracker
    tracker = SummaryVersionTracker()
    tracker.track_from_messages(messages)    # scan for summaries
    tracker.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .markers import is_compaction_message

logger = logging.getLogger(__name__)

_VERSION = "1.0"
_DEFAULT_PATH = "~/.hermes/summary_versions.json"


class SummaryVersionTracker:
    """Tracks summary hashes for version history across compression cycles.

    Maintains a list of (hash, timestamp, session_id) entries for all
    summaries generated in the current and previous sessions.

    A summary hash uniquely identifies the content of a compressed
    handoff summary. Identical hashes indicate no meaningful change
    occurred during a compression cycle.
    """

    def __init__(self, path: str | None = None):
        self._path = Path(os.path.expanduser(path or os.environ.get(
            "HERMES_CC_VERSION_FILE", _DEFAULT_PATH
        )))
        self._hashes: List[dict] = []  # list of {hash, timestamp, session_id}
        self._version = _VERSION
        self._dirty = False

    @property
    def version(self) -> str:
        return self._version

    @property
    def hashes(self) -> List[dict]:
        return self._hashes

    def load(self) -> None:
        """Load version history from disk."""
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        self._version = data.get("version", _VERSION)
                        self._hashes = data.get("hashes", [])
                    elif isinstance(data, list):
                        self._hashes = data
                    self._dirty = False
                    logger.debug("SummaryVersion: loaded %d entries", len(self._hashes))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("SummaryVersion: load failed: %s", e)

    def save(self) -> None:
        """Save version history to disk. Atomic write."""
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "version": self._version,
                    "hashes": self._hashes,
                }, fh, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
            self._dirty = False
            logger.debug("SummaryVersion: saved %d hashes", len(self._hashes))
        except OSError as e:
            logger.warning("SummaryVersion: save failed: %s", e)

    def track_from_messages(self, messages: List[dict], session_id: str = "") -> int:
        """Scan messages for compaction summary messages and record their hashes.

        Called from post_llm_call. The production agent prefixes summaries
        with "[CONTEXT COMPACTION" / "[CONTEXT SUMMARY]:" and sets the
        "_compressed_summary" metadata flag — there is NO "[Summary vN |
        hash:...]" metadata in the current core. We hash the summary body
        (SHA256, first 8 chars) for change detection.

        Returns:
            Number of new hashes added.
        """
        if not messages:
            return 0

        seen_hashes = {h["hash"] for h in self._hashes}
        added = 0
        timestamp = datetime.now().isoformat()

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if not is_compaction_message(msg):
                continue
            content = msg.get("content", "") or ""
            if not isinstance(content, str) or not content.strip():
                continue
            h = self.compute_hash(content.strip())
            if h not in seen_hashes:
                self._hashes.append({
                    "hash": h,
                    "version": self._version,
                    "timestamp": timestamp,
                    "session_id": session_id or "",
                })
                seen_hashes.add(h)
                self._dirty = True
                added += 1

        if added:
            logger.debug("SummaryVersion: added %d new hash(es)", added)

        return added

    def get_last_hash(self) -> Optional[dict]:
        """Return the most recent summary hash entry, or None."""
        return self._hashes[-1] if self._hashes else None

    def format_metadata_line(self) -> str:
        """Return a version/hash metadata line for the current summary.

        This replicates the [Summary vX.X | hash:XXXXXXXX] format used
        by ContextCompressor._format_summary_for_output().

        If no hashes are tracked, returns the current version line.
        """
        last = self.get_last_hash()
        if last:
            return f"[Summary v{last.get('version', self._version)} | hash:{last['hash']}]"
        return f"[Summary v{self._version} | hash:unknown]"

    def is_duplicate(self, content_hash: str) -> bool:
        """Return True if this content hash has been seen before."""
        return content_hash in {h["hash"] for h in self._hashes}

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash (first 8 chars) of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:8]

    def mark_dirty(self) -> None:
        self._dirty = True
