"""Task classifier — two-stage: keyword heuristics → lightweight LLM."""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


COMPOSITE_PATTERNS = [
    r"\b(fix|bug|issue|patch|PR|pull.request)\b",
    r"\b(implement|add|feature|refactor)\b.*\b(code|file|function)\b",
    r"\b(write|create|generate)\b.*\b(test|unittest|coverage)",
    r"\b(test|unittest|coverage).*\b(write|create|generate)\b",
    r"\b(review|appraise|evaluate)\b.*\b(PR|pull.request|change)\b",
    r"\b(reproduce|replicate)\b.*\b(bug|issue|error|failure|crash)\b",
    r"\b(reproduce|replicate)\b",
]

COMPOSITE_KEYWORDS = {
    "fix", "bug", "issue", "patch", "PR", "pull request",
    "implement", "feature", "refactor", "code change",
    "write test", "generate test", "unit test",
    "review", "code review", "appraise",
    "reproduce", "reproduction script",
}

SKILL_KEYWORD_MAP = {
    "code_localization": {"file", "path", "locate", "find", "relevant", "which file", "where"},
    "code_editing": {"fix", "patch", "edit", "change", "modify", "update", "refactor"},
    "unit_test_generation": {"test", "unittest", "coverage", "assertion", "mutation", "write"},
    "issue_reproduction": {"reproduce", "replicate", "trigger", "failure", "script", "demo", "crash", "bug"},
    "code_review": {"review", "appraise", "evaluate", "PR", "pull request", "judge"},
}

CLASSIFIER_PROMPT = """Classify this request. Output JSON only:
{{"is_composite": bool, "detected_skills": [skill_names], "confidence": float, "reasoning": "brief explanation"}}

Skills: code_localization, code_editing, unit_test_generation, issue_reproduction, code_review

Request: {user_message}"""


@dataclass
class ClassificationResult:
    is_composite: bool
    detected_skills: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    suggested_order: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_composite": self.is_composite,
            "detected_skills": self.detected_skills,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_order": self.suggested_order,
        }


class TaskClassifier:
    """Two-stage task classifier for atomic pipeline."""

    def __init__(self):
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in COMPOSITE_PATTERNS]

    def classify(self, user_message: str, llm_callback=None) -> ClassificationResult:
        """Classify a user request. Stage 1: heuristics. Stage 2: LLM if ambiguous."""
        heuristic_score = self._heuristic_score(user_message)

        if heuristic_score > 0.7:
            skills = self._detect_skills(user_message)
            return ClassificationResult(
                is_composite=True,
                detected_skills=skills,
                confidence=min(heuristic_score + 0.1, 1.0),
                reasoning="High heuristic score — composite task detected",
                suggested_order=self._suggest_order(skills),
            )
        elif heuristic_score > 0.3:
            skills = self._detect_skills(user_message)
            if llm_callback:
                return self._llm_classify(user_message, llm_callback)
            return ClassificationResult(
                is_composite=len(skills) >= 1,
                detected_skills=skills,
                confidence=heuristic_score,
                reasoning="Ambiguous — heuristic score in middle range",
                suggested_order=self._suggest_order(skills),
            )
        else:
            return ClassificationResult(
                is_composite=False,
                detected_skills=[],
                confidence=1.0 - heuristic_score,
                reasoning="Low heuristic score — simple query",
            )

    def _heuristic_score(self, message: str) -> float:
        """Compute composite score from keyword + pattern matching."""
        lower = message.lower()
        score = 0.0

        for pattern in self._compiled_patterns:
            if pattern.search(lower):
                score += 0.3

        keyword_hits = sum(1 for kw in COMPOSITE_KEYWORDS if kw.lower() in lower)
        score += min(keyword_hits * 0.12, 0.5)

        return min(score, 1.0)

    def _detect_skills(self, message: str) -> List[str]:
        """Detect which atomic skills are relevant."""
        lower = message.lower()
        detected = []
        for skill, keywords in SKILL_KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw in lower)
            if hits >= 1:
                detected.append(skill)
        return detected

    def _suggest_order(self, skills: List[str]) -> List[str]:
        """Suggest execution order based on dependencies."""
        priority = {
            "code_localization": 0,
            "issue_reproduction": 1,
            "code_editing": 2,
            "unit_test_generation": 3,
            "code_review": 4,
        }
        return sorted(skills, key=lambda s: priority.get(s, 99))

    def _llm_classify(self, message: str, llm_callback) -> ClassificationResult:
        """Fallback LLM classification for ambiguous cases."""
        prompt = CLASSIFIER_PROMPT.format(user_message=message)
        try:
            response = llm_callback(prompt)
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            return ClassificationResult(
                is_composite=data.get("is_composite", False),
                detected_skills=data.get("detected_skills", []),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", "LLM classification"),
                suggested_order=self._suggest_order(data.get("detected_skills", [])),
            )
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            skills = self._detect_skills(message)
            return ClassificationResult(
                is_composite=len(skills) > 1,
                detected_skills=skills,
                confidence=0.5,
                reasoning="LLM fallback failed — using heuristic detection",
                suggested_order=self._suggest_order(skills),
            )
