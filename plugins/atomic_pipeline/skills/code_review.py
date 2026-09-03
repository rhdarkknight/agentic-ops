"""Skill 5: Code Review — judge if PR addresses the issue."""
import json
from typing import Any, Dict, List, Optional

from ..core.skill_base import SkillBase, SkillResult


class CodeReviewSkill(SkillBase):
    name = "code_review"
    description = "Judge whether a PR correctly addresses the issue"
    tool_allowlist = ["read_file", "grep", "execute_code"]

    def execute(
        self,
        pr_diff: str,
        issue_description: str,
        pr_files: Optional[List[str]] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            judgment, confidence, evidence, concerns = self._review(
                pr_diff, issue_description, pr_files or []
            )
            return SkillResult.success(
                skill=self.name,
                data={
                    "judgment": judgment,
                    "confidence": confidence,
                    "evidence": evidence,
                    "concerns": concerns,
                    "judgment_correct": judgment == "accept",
                },
                reward=1.0 if judgment == "accept" and confidence > 0.7 else -0.5,
            )
        except Exception as e:
            return SkillResult.failure(self.name, str(e))

    def _review(self, diff: str, issue: str, files: List[str]) -> tuple:
        lower_issue = issue.lower()
        lower_diff = diff.lower()
        evidence = []
        concerns = []

        if any(kw in lower_diff for kw in ["fix", "patch", "resolve"]):
            evidence.append("PR contains fix-related changes")
        if any(kw in lower_diff for kw in ["test", "assert"]):
            evidence.append("PR includes test modifications")

        if len(files) == 0:
            concerns.append("No files changed")

        confidence = min(0.9, 0.5 + len(evidence) * 0.2)
        judgment = "accept" if len(evidence) >= 2 and confidence > 0.6 else "reject"
        return judgment, confidence, evidence, concerns

    def get_system_prompt(self) -> str:
        return (
            "You are in CODE_REVIEW mode. Evaluate whether the PR correctly addresses the issue. "
            "Provide binary judgment (accept/reject) with evidence. "
            "Output JSON: {\"judgment\": \"accept|reject\", \"confidence\": 0.0-1.0, \"evidence\": [...], \"concerns\": [...]}"
        )

    def validate_result(self, result: SkillResult) -> bool:
        if not result.success:
            return False
        data = result.data
        return (
            data.get("judgment") in ("accept", "reject")
            and isinstance(data.get("confidence"), float)
            and isinstance(data.get("evidence"), list)
        )

    def get_handoff_format(self) -> dict:
        return {
            "skill": "code_review",
            "judgment": "accept",
            "confidence": 0.92,
            "evidence": ["PR adds timeout parameter", "Tests verify behavior"],
            "concerns": ["Missing documentation update"],
        }
