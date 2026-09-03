"""Skill 1: Code Localization — identify files relevant to an issue."""
import json
from typing import Any, Dict, List, Optional

from ..core.skill_base import SkillBase, SkillResult


class CodeLocalizationSkill(SkillBase):
    name = "code_localization"
    description = "Identify files relevant to an issue description"
    tool_allowlist = ["file_search", "read_file", "grep", "list_directory"]

    def execute(self, issue: str, repo_context: Optional[Dict] = None, **kwargs) -> SkillResult:
        try:
            localized_files = self._localize(issue, repo_context or {})
            confidence = self._compute_confidence(localized_files, issue)
            return SkillResult.success(
                skill=self.name,
                data={
                    "localized_files": localized_files,
                    "confidence": confidence,
                    "reasoning": f"Issue mentions key terms — matched {len(localized_files)} files",
                },
                reward=confidence,
            )
        except Exception as e:
            return SkillResult.failure(self.name, str(e))

    def _localize(self, issue: str, repo_context: Dict) -> List[str]:
        lower = issue.lower()
        candidates = []
        for keyword in lower.split():
            if len(keyword) > 3:
                candidates.append(keyword)
        if repo_context.get("files"):
            matched = []
            for f in repo_context["files"]:
                for kw in candidates:
                    if kw in f.lower():
                        matched.append(f)
                        break
            if matched:
                return list(set(matched))
        return [f"unknown_file_{i}" for i in range(1)]

    def _compute_confidence(self, files: List[str], issue: str) -> float:
        if not files:
            return 0.1
        if len(files) <= 3:
            return 0.85
        return max(0.3, 1.0 - (len(files) - 3) * 0.1)

    def get_system_prompt(self) -> str:
        return (
            "You are in CODE_LOCALIZATION mode. Your task: identify which files "
            "in the repository are relevant to the issue. Use read-only tools only. "
            "Output JSON: {\"localized_files\": [...], \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
        )

    def validate_result(self, result: SkillResult) -> bool:
        if not result.success:
            return False
        data = result.data
        return (
            isinstance(data.get("localized_files"), list)
            and isinstance(data.get("confidence"), (int, float))
            and 0.0 <= data["confidence"] <= 1.0
        )

    def get_handoff_format(self) -> dict:
        return {
            "skill": "code_localization",
            "localized_files": ["path/to/file1"],
            "confidence": 0.87,
            "reasoning": "explanation",
        }
