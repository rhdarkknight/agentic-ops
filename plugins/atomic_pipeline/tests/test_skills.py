"""Tests for atomic skills."""
import pytest
import json

from ..skills.code_localization import CodeLocalizationSkill
from ..skills.code_editing import CodeEditingSkill
from ..skills.unit_test_gen import UnitTestGenSkill
from ..skills.issue_reproduction import IssueReproductionSkill
from ..skills.code_review import CodeReviewSkill


class TestCodeLocalizationSkill:

    def test_localize_returns_files(self):
        skill = CodeLocalizationSkill()
        result = skill.execute(issue="Fix the timeout bug in handler.py")
        assert result.success
        assert "localized_files" in result.data
        assert 0.0 <= result.data["confidence"] <= 1.0

    def test_localize_with_repo_context(self):
        skill = CodeLocalizationSkill()
        result = skill.execute(
            issue="Fix auth module",
            repo_context={"files": ["auth.py", "handler.py", "test_auth.py"]},
        )
        assert result.success
        assert len(result.data["localized_files"]) > 0

    def test_validate_result_valid(self):
        skill = CodeLocalizationSkill()
        result = skill.execute(issue="test")
        assert skill.validate_result(result)

    def test_validate_result_invalid(self):
        skill = CodeLocalizationSkill()
        from ..core.skill_base import SkillResult
        bad = SkillResult.failure("code_localization", "error")
        assert not skill.validate_result(bad)

    def test_system_prompt(self):
        skill = CodeLocalizationSkill()
        prompt = skill.get_system_prompt()
        assert "CODE_LOCALIZATION" in prompt

    def test_handoff_format(self):
        skill = CodeLocalizationSkill()
        fmt = skill.get_handoff_format()
        assert "localized_files" in fmt
        assert "confidence" in fmt


class TestCodeEditingSkill:

    def test_edit_generates_patch(self):
        skill = CodeEditingSkill()
        result = skill.execute(
            file_path="src/handler.py",
            edit_instruction="Add timeout parameter",
        )
        assert result.success
        assert "patch" in result.data
        assert "handler.py" in result.data["file"]

    def test_edit_with_tests(self):
        skill = CodeEditingSkill()
        result = skill.execute(
            file_path="src/main.py",
            edit_instruction="Fix bug",
            run_tests=True,
        )
        assert result.success
        assert "tests_passed" in result.data

    def test_validate_result(self):
        skill = CodeEditingSkill()
        result = skill.execute(file_path="test.py", edit_instruction="fix")
        assert skill.validate_result(result)

    def test_system_prompt(self):
        skill = CodeEditingSkill()
        assert "CODE_EDITING" in skill.get_system_prompt()


class TestUnitTestGenSkill:

    def test_generate_tests(self):
        skill = UnitTestGenSkill()
        result = skill.execute(
            function_code="def add(a, b): return a + b",
            specification="Adds two numbers",
        )
        assert result.success
        assert result.data["test_cases"] >= 1
        assert 0.0 <= result.data["mutation_score"] <= 1.0

    def test_validate_result(self):
        skill = UnitTestGenSkill()
        result = skill.execute(function_code="def f(): pass", specification="test")
        assert skill.validate_result(result)

    def test_mutation_score_range(self):
        skill = UnitTestGenSkill()
        result = skill.execute(function_code="def f(): pass", specification="test")
        assert 0.0 <= result.data["mutation_score"] <= 1.0


class TestIssueReproductionSkill:

    def test_reproduce_generates_script(self):
        skill = IssueReproductionSkill()
        result = skill.execute(issue_description="App crashes on startup")
        assert result.success
        assert "script" in result.data
        assert "causal" in result.data

    def test_validate_result(self):
        skill = IssueReproductionSkill()
        result = skill.execute(issue_description="test issue")
        assert skill.validate_result(result)

    def test_causal_validation(self):
        skill = IssueReproductionSkill()
        result = skill.execute(issue_description="test")
        assert isinstance(result.data["causal"], bool)


class TestCodeReviewSkill:

    def test_review_accept(self):
        skill = CodeReviewSkill()
        result = skill.execute(
            pr_diff="--- a/file.py\n+++ b/file.py\n+fixed bug",
            issue_description="Fix the timeout bug",
        )
        assert result.success
        assert result.data["judgment"] in ("accept", "reject")
        assert 0.0 <= result.data["confidence"] <= 1.0

    def test_review_with_evidence(self):
        skill = CodeReviewSkill()
        result = skill.execute(
            pr_diff="--- a/file.py\n+++ b/file.py\n+added test\n+fixed issue",
            issue_description="Fix the bug",
        )
        assert result.success
        assert isinstance(result.data["evidence"], list)

    def test_validate_result(self):
        skill = CodeReviewSkill()
        result = skill.execute(pr_diff="diff content", issue_description="fix bug")
        assert skill.validate_result(result)

    def test_judgment_values(self):
        skill = CodeReviewSkill()
        result = skill.execute(pr_diff="test", issue_description="test")
        assert result.data["judgment"] in ("accept", "reject")
