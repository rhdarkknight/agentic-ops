"""Tests for TaskClassifier."""
import pytest

from ..core.classifier import TaskClassifier, ClassificationResult


@pytest.fixture
def classifier():
    return TaskClassifier()


class TestTaskClassifier:

    def test_composite_detection_fix_bug(self, classifier):
        result = classifier.classify("Fix the timeout bug in handler.py")
        assert result.is_composite
        assert "code_editing" in result.detected_skills or "code_localization" in result.detected_skills

    def test_simple_query_not_composite(self, classifier):
        result = classifier.classify("What is Python?")
        assert not result.is_composite
        assert result.detected_skills == []

    def test_skill_detection_write_tests(self, classifier):
        result = classifier.classify("Write tests for the auth module")
        assert result.is_composite
        assert "unit_test_generation" in result.detected_skills

    def test_skill_detection_code_review(self, classifier):
        result = classifier.classify("Review this PR for the authentication fix")
        assert result.is_composite
        assert "code_review" in result.detected_skills

    def test_skill_detection_issue_reproduction(self, classifier):
        result = classifier.classify("Reproduce the crash when uploading large files")
        assert result.is_composite
        assert "issue_reproduction" in result.detected_skills

    def test_heuristic_score_high_for_fix(self, classifier):
        score = classifier._heuristic_score("Fix the bug in the login handler")
        assert score > 0.3

    def test_heuristic_score_low_for_simple(self, classifier):
        score = classifier._heuristic_score("What is 2+2?")
        assert score < 0.3

    def test_suggested_order_priority(self, classifier):
        skills = ["code_review", "code_editing", "code_localization"]
        order = classifier._suggest_order(skills)
        assert order[0] == "code_localization"
        assert order[-1] == "code_review"

    def test_classification_result_to_dict(self, classifier):
        result = ClassificationResult(
            is_composite=True,
            detected_skills=["code_editing"],
            confidence=0.8,
            reasoning="test",
        )
        d = result.to_dict()
        assert d["is_composite"] is True
        assert d["detected_skills"] == ["code_editing"]
        assert d["confidence"] == 0.8

    def test_llm_fallback_on_parse_error(self, classifier):
        def bad_callback(prompt):
            raise ValueError("bad")
        result = classifier._llm_classify("test message", bad_callback)
        assert isinstance(result, ClassificationResult)
        assert result.confidence == 0.5
