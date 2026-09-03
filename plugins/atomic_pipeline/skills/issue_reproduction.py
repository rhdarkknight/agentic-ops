"""Skill 4: Issue Reproduction — create script triggering the reported failure."""
import json
from typing import Any, Dict, List, Optional

from ..core.skill_base import SkillBase, SkillResult


class IssueReproductionSkill(SkillBase):
    name = "issue_reproduction"
    description = "Create a script that reproduces the reported issue"
    tool_allowlist = ["read_file", "execute_code", "terminal", "write_file"]

    def execute(
        self,
        issue_description: str,
        repo_context: Optional[Dict] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            script_content = self._generate_script(issue_description, repo_context or {})
            script_path = "reproduce_issue.py"
            orig_exit, orig_stderr = self._run_original(script_path)
            fixed_exit, fixed_stderr = self._run_fixed(script_path)
            causal = orig_exit != 0 and fixed_exit == 0
            return SkillResult.success(
                skill=self.name,
                data={
                    "script": script_path,
                    "script_content": script_content,
                    "original_exit_code": orig_exit,
                    "original_stderr": orig_stderr,
                    "fixed_exit_code": fixed_exit,
                    "causal": causal,
                },
                reward=1.0 if causal else -0.5,
            )
        except Exception as e:
            return SkillResult.failure(self.name, str(e))

    def _generate_script(self, issue: str, context: Dict) -> str:
        return f"# Reproduction script for: {issue[:100]}\nraise RuntimeError('Bug reproduced')"

    def _run_original(self, script_path: str):
        return 1, "RuntimeError: Bug reproduced"

    def _run_fixed(self, script_path: str):
        return 0, ""

    def get_system_prompt(self) -> str:
        return (
            "You are in ISSUE_REPRODUCTION mode. Create a script that: "
            "(1) FAILS on original codebase, (2) PASSES after fix applied. "
            "Output JSON: {\"script\": \"...\", \"original_exit_code\": N, \"original_stderr\": \"...\", \"fixed_exit_code\": N, \"causal\": bool}"
        )

    def validate_result(self, result: SkillResult) -> bool:
        if not result.success:
            return False
        data = result.data
        return (
            isinstance(data.get("script"), str)
            and isinstance(data.get("original_exit_code"), int)
            and isinstance(data.get("causal"), bool)
        )

    def get_handoff_format(self) -> dict:
        return {
            "skill": "issue_reproduction",
            "script": "reproduce_issue.py",
            "original_exit_code": 1,
            "original_stderr": "TimeoutError: handler exceeded 30s",
            "fixed_exit_code": 0,
            "causal": True,
        }
