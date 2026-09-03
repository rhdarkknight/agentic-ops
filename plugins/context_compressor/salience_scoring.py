"""
Salience Scoring Plugin
=======================

Scores conversation turns by importance. High-salience turns are
preserved verbatim during compression; low-salience turns are
candidates for summarization.

Scoring factors:
- User corrections ("wrong", "actually", "revert")
- File modifications (write_file, patch tool calls)
- Error tracebacks
- TODO/DECISION markers
- File paths mentioned
- Recency (more recent = more important)
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

HIGH_VALUE_TOOLS = {'write_file', 'patch', 'terminal'}
CORRECTION_PATTERNS = [
    r'\b(wrong|incorrect|no\s+not|actually|undo|revert)\b',
    r'\b(don\'t|do not|stop|never|avoid)\b',
    r'\b(must|important)\b',
]

TODO_PATTERNS = [
    r'\b(TODO|FIXME|NOTE|DECISION|HACK)\b',
    r'\b(action item|next step|follow up)\b',
]


class SalienceScorer:
    """Scores turns for importance-based preservation during compression."""

    def pre_compress(self, ctx, session_id, messages, compression_count, approx_tokens, **kwargs):
        """Score all turns and return salience map.

        DEPRECATED (2026-08-19): the pre_compress hook does not exist in
        VALID_HOOKS. Kept for backward compatibility; the plugin now uses
        score_conversation() via the pre_llm_call hook.
        """
        total = len(messages)
        salience_scores = {}

        for i, msg in enumerate(messages):
            score = self._score_turn(msg, i, total)
            if score > 0.5:  # Only track high-salience turns
                salience_scores[i] = score

        return {"salience_scores": salience_scores}

    def score_conversation(self, messages) -> str:
        """Score conversation turns and return a preservation prompt block.

        Called from the valid pre_llm_call hook. Returns a compact directive
        listing high-salience turns (user corrections, decisions, file
        modifications, errors) that must survive context compression.
        """
        if not messages:
            return ""
        total = len(messages)
        high: list[tuple[int, str]] = []
        for i, msg in enumerate(messages):
            score = self._score_turn(msg, i, total)
            if score > 0.6:
                role = msg.get("role", "?")
                content = (msg.get("content") or "")
                if isinstance(content, str):
                    content = " ".join(content.split())[:160]
                elif isinstance(content, list):
                    content = "[multimodal content]"
                else:
                    content = str(content)[:160]
                if content:
                    high.append((i, f"[turn {i} {role}] {content}"))
        if not high:
            return ""
        lines = ["[context_compressor_enhancements: high-salience turns — preserve these "
                 "details across any compression:]"]
        lines += [f"- {text}" for _, text in high[:8]]
        return "\n".join(lines)

    def _score_turn(self, message: Dict, turn_index: int, total_turns: int) -> float:
        """Score a single turn for salience (0.0 to 1.0)."""
        score = 0.0
        raw_content = message.get("content") or ""
        # Multimodal / tool-result content can be a list of parts — never
        # crash on non-str content.
        if isinstance(raw_content, str):
            content = raw_content.lower()
        elif isinstance(raw_content, list):
            content = " ".join(
                str(p.get("text", "")) for p in raw_content if isinstance(p, dict)
            ).lower()
        else:
            content = str(raw_content).lower()

        # User messages get base score
        if message.get("role") == "user":
            score += 0.2

        # Recent turns are more important
        recency_bonus = turn_index / max(total_turns, 1)
        score += recency_bonus * 0.3

        # Check for corrections
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, content):
                score += 0.3
                break

        # Check for TODOs
        for pattern in TODO_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.2
                break

        # Check tool calls
        for tc in message.get("tool_calls") or []:
            fn_name = tc.get("function", {}).get("name", "")
            if fn_name in HIGH_VALUE_TOOLS:
                score += 0.25

        # Check for file paths
        if re.search(r'[/\w.-]+\.(py|yaml|yml|json|toml|md|txt|sh|rs|go|js|ts)', content):
            score += 0.1

        # Check for error indicators
        if any(kw in content for kw in ['error', 'exception', 'traceback', 'failed', 'crash']):
            score += 0.15

        return min(score, 1.0)
