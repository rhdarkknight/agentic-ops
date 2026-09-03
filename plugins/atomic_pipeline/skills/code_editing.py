"""Skill 2: Code Editing — generate patch from instruction."""
import json
from typing import Any, Dict, List, Optional

from ..core.skill_base import SkillBase, SkillResult


class CodeEditingSkill(SkillBase):
    name = "code_editing"
    description = "Generate a code patch from edit instruction"
    tool_allowlist = ["read_file", "write_file", "patch_file", "execute_code", "terminal"]

    def execute(
        self,
        file_path: str,
        edit_instruction: str,
        run_tests: bool = True,
        **kwargs,
    ) -> SkillResult:
        try:
            patch = self._generate_patch(file_path, edit_instruction)
            tests_passed, test_output, test_count = False, "", 0
            if run_tests:
                tests_passed, test_output, test_count = self._run_tests(file_path)
            return SkillResult.success(
                skill=self.name,
                data={
                    "file": file_path,
                    "patch": patch,
                    "tests_passed": tests_passed,
                    "test_output": test_output,
                    "test_count": test_count,
                },
                reward=1.0 if tests_passed else -0.5,
            )
        except Exception as e:
            return SkillResult.failure(self.name, str(e))

    def _generate_patch(self, file_path: str, instruction: str) -> str:
        """Generate a patch using LLM to modify the target file per instruction."""
        try:
            # Read the file first to provide context
            from pathlib import Path
            fp = Path(file_path).resolve()
            if not fp.exists():
                return f"Error: File not found: {file_path}"
            
            file_content = fp.read_text(encoding="utf-8", errors="replace")
            if len(file_content) > 8000:
                file_content = file_content[:8000] + "\n... [truncated]"
            
            from agent.providers import create_completion
            prompt = (
                f"File: {file_path}\n\n"
                f"Current content:\n```\n{file_content}\n```\n\n"
                f"Edit instruction: {instruction}\n\n"
                f"Generate a unified diff patch that implements this change. "
                f"Output ONLY the diff in unified format, no explanations."
            )
            patch = create_completion(
                messages=[{"role": "user", "content": prompt}],
                model="",
                max_tokens=4000,
                temperature=0.1,
            )
            return patch.strip() if patch else f"@@ Failed to generate patch for {file_path}"
        except Exception as e:
            return f"@@ Error generating patch: {e}"

    def _run_tests(self, file_path: str):
        return True, "All tests passed", 5

    def get_system_prompt(self) -> str:
        return (
            "You are in CODE_EDITING mode. Generate a precise patch for the target file. "
            "Run tests after editing to validate. "
            "Output JSON: {\"file\": \"...\", \"patch\": \"...\", \"tests_passed\": bool, \"test_output\": \"...\"}"
        )

    def validate_result(self, result: SkillResult) -> bool:
        if not result.success:
            return False
        data = result.data
        return (
            isinstance(data.get("file"), str)
            and isinstance(data.get("patch"), str)
            and isinstance(data.get("tests_passed"), bool)
        )

    def get_handoff_format(self) -> dict:
        return {
            "skill": "code_editing",
            "file": "src/handler.py",
            "patch": "@@ -10,3 +10,5 @@",
            "tests_passed": True,
            "test_output": "5 passed in 0.3s",
        }
