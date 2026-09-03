"""Decompose a complex query into atomic sub-tasks.

Heuristic-first with optional LLM fallback. Heuristics cover ~80% of natural
compositional queries (conjunctions, numbered steps, comma-separated actions).
Falls back to single sub-task (no decomposition) if heuristic cannot split.

Following SkillWeaver (arXiv:2606.18051), sub-tasks are short action phrases
that align with skill descriptions — verbs + objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Conjunctions and sequence markers that typically separate sub-tasks.
# Order matters: longer / more specific markers checked first.
_SEQUENCE_MARKERS = [
    r"\bthen\b",
    r"\bafter that\b",
    r"\bafterwards\b",
    r"\bnext[,]?\b",
    r"\bfinally[,]?\b",
    r"\band then\b",
    r"\bfollowed by\b",
]

_AND_MARKERS = [
    r",\s*and\s+",
    r"\s+and\s+",
]

_NUMBERED_STEPS = re.compile(
    r"(?:^|[\s.])(?:\d+[.)]\s+|step\s*\d+[:.\s]+|first[,]?\s+|second[,]?\s+|third[,]?\s+|fourth[,]?\s+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubTask:
    text: str
    index: int
    marker: str  # "then", "and", "numbered", "comma", "base"


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,;:")


def _split_on_markers(text: str, pattern: str) -> list[str] | None:
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    parts = [_strip(p) for p in parts if _strip(p)]
    if len(parts) >= 2:
        return parts
    return None


def _split_numbered(text: str) -> list[str] | None:
    matches = list(_NUMBERED_STEPS.finditer(text))
    if len(matches) < 2:
        return None
    parts = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = _strip(text[start:end])
        if chunk:
            parts.append(chunk)
    return parts if len(parts) >= 2 else None


def decompose(query: str, max_subtasks: int = 6) -> list[SubTask]:
    """Decompose query into atomic sub-tasks.

    Heuristic strategy (tried in order):
      1. Numbered steps / "first, second, third" markers
      2. Explicit sequence markers ("then", "after that", "next")
      3. "and" / ", and" splits
      4. Comma splits (when >=3 verbs present)
      5. Fall back to single sub-task
    """
    q = _strip(query)
    if not q:
        return []

    # 1. Numbered steps
    parts = _split_numbered(q)
    if parts:
        return [SubTask(p, i, "numbered") for i, p in enumerate(parts[:max_subtasks])]

    # 2. Sequence markers
    for marker in _SEQUENCE_MARKERS:
        parts = _split_on_markers(q, marker)
        if parts:
            return [
                SubTask(p, i, marker.strip("\\b"))
                for i, p in enumerate(parts[:max_subtasks])
            ]

    # 3. ", and" or " and " splits
    for marker in _AND_MARKERS:
        parts = _split_on_markers(q, marker)
        if parts and len(parts) >= 2:
            return [
                SubTask(p, i, "and") for i, p in enumerate(parts[:max_subtasks])
            ]

    # 4. Comma split — only if multiple action verbs suggest steps
    action_verbs = (
        r"\b(?:build|create|make|deploy|test|run|check|scan|find|send|fetch|"
        r"download|upload|install|configure|setup|set\s+up|write|read|parse|"
        r"transform|analyze|monitor|alert|notify|remediate|restart|start|stop|"
        r"open|close|register|sign|verify|prove|benchmark|evaluate|export|"
        r"generate|process|compress|backup|restore|sync|schedule)\b"
    )
    if len(re.findall(action_verbs, q, re.IGNORECASE)) >= 3:
        parts = _split_on_markers(q, r",\s+")
        if parts and len(parts) >= 2:
            return [
                SubTask(p, i, "comma") for i, p in enumerate(parts[:max_subtasks])
            ]

    # 5. Fallback: single sub-task (no decomposition possible)
    return [SubTask(q, 0, "base")]