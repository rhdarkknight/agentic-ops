"""Tests for AtomicOrchestrator and pipeline integration."""
import pytest

from ..core.orchestrator import AtomicOrchestrator, PipelineState, SkillExecutionRecord
from ..core.classifier import TaskClassifier, ClassificationResult
from ..core.skill_base import SkillResult
from ..skills import (
    CodeLocalizationSkill,
    CodeEditingSkill,
    UnitTestGenSkill,
    IssueReproductionSkill,
    CodeReviewSkill,
)


@pytest.fixture(autouse=True)
def reset_orchestrator():
    """Reset orchestrator state between tests."""
    AtomicOrchestrator._instances = {}
    AtomicOrchestrator._current_state = None
    yield
    AtomicOrchestrator._instances = {}
    AtomicOrchestrator._current_state = None


class TestAtomicOrchestrator:

    def test_get_instance(self):
        orch = AtomicOrchestrator.get_instance()
        assert isinstance(orch, AtomicOrchestrator)
        assert AtomicOrchestrator.get_instance() is orch

    def test_init_session(self):
        state = AtomicOrchestrator.init_session("test-session-1")
        assert isinstance(state, PipelineState)
        assert state.session_id == "test-session-1"
        assert not state.completed

    def test_finalize_session(self):
        AtomicOrchestrator.init_session("test-session-2")
        AtomicOrchestrator.finalize_session("test-session-2")
        assert AtomicOrchestrator._current_state is None

    def test_register_skill(self):
        orch = AtomicOrchestrator.get_instance()
        orch.register_skill("test_skill", CodeLocalizationSkill())
        assert "test_skill" in orch.skills

    def test_execute_unknown_skill(self):
        orch = AtomicOrchestrator.get_instance()
        result = orch.execute_skill("nonexistent")
        assert not result.success
        assert "Unknown skill" in result.error

    def test_execute_known_skill(self):
        AtomicOrchestrator._instances = {}
        orch = AtomicOrchestrator.get_instance()
        orch.register_skill("localization", CodeLocalizationSkill())
        result = orch.execute_skill("localization", issue="test issue")
        assert result.success
        assert result.skill == "code_localization"

    def test_classify_request_composite(self):
        AtomicOrchestrator._instances = {}
        orch = AtomicOrchestrator.get_instance()
        result = orch.classify_request("Fix the timeout bug in handler.py")
        assert result.is_composite

    def test_classify_request_simple(self):
        orch = AtomicOrchestrator.get_instance()
        result = orch.classify_request("What is Python?")
        assert not result.is_composite

    def test_get_trajectory_empty(self):
        orch = AtomicOrchestrator.get_instance()
        traj = orch.get_trajectory()
        assert traj == {}

    def test_get_trajectory_with_session(self):
        AtomicOrchestrator.init_session("test-session-3")
        orch = AtomicOrchestrator.get_instance()
        orch.register_skill("loc", CodeLocalizationSkill())
        orch.execute_skill("loc", issue="test")
        traj = orch.get_trajectory()
        assert "skills_executed" in traj
        assert len(traj["skills_executed"]) >= 1

    def test_execute_composite_simple(self):
        orch = AtomicOrchestrator.get_instance()
        result = orch.execute_composite("What is 2+2?")
        assert result["success"]
        assert not result["composite"]

    def test_skill_result_to_json(self):
        import json
        result = SkillResult.success("test_skill", {"key": "value"}, reward=0.8)
        parsed = json.loads(result.to_json())
        assert parsed["skill"] == "test_skill"
        assert parsed["success"] is True
        assert parsed["data"] == {"key": "value"}

    def test_skill_result_failure(self):
        result = SkillResult.failure("test_skill", "error message")
        assert not result.success
        assert result.error == "error message"
        assert result.reward == -1.0

    def test_pipeline_state_to_dict(self):
        state = PipelineState(session_id="test")
        state.active_skill = "code_editing"
        state.is_composite = True
        d = state.to_dict()
        assert d["session_id"] == "test"
        assert d["active_skill"] == "code_editing"
        assert d["is_composite"] is True
        assert d["completed"] is False

    def test_skill_execution_record(self):
        result = SkillResult.success("test", {})
        record = SkillExecutionRecord(skill="test", result=result, duration_ms=100.0)
        d = record.to_dict()
        assert d["skill"] == "test"
        assert d["duration_ms"] == 100.0
        assert d["success"] is True

    def test_full_pipeline_with_all_skills(self):
        """End-to-end: register all skills, execute one of each."""
        orch = AtomicOrchestrator.get_instance()
        orch.register_skill("code_localization", CodeLocalizationSkill())
        orch.register_skill("code_editing", CodeEditingSkill())
        orch.register_skill("unit_test_generation", UnitTestGenSkill())
        orch.register_skill("issue_reproduction", IssueReproductionSkill())
        orch.register_skill("code_review", CodeReviewSkill())

        AtomicOrchestrator.init_session("full-test")

        r1 = orch.execute_skill("code_localization", issue="Fix bug in auth")
        assert r1.success

        r2 = orch.execute_skill("code_editing", file_path="auth.py", edit_instruction="fix")
        assert r2.success

        r3 = orch.execute_skill("unit_test_generation", function_code="def f(): pass", specification="test")
        assert r3.success

        r4 = orch.execute_skill("issue_reproduction", issue_description="crash on login")
        assert r4.success

        r5 = orch.execute_skill("code_review", pr_diff="diff", issue_description="fix")
        assert r5.success

        traj = orch.get_trajectory()
        assert len(traj["skills_executed"]) == 5

        AtomicOrchestrator.finalize_session("full-test")
