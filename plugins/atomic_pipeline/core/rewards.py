"""Reward calculators for each atomic skill — RL-style scoring."""
from typing import Any, Dict, List, Optional


class RewardCalculator:
    """Computes rewards for atomic skill outcomes."""

    @staticmethod
    def code_localization_reward(
        predicted_files: List[str],
        ground_truth_files: Optional[List[str]] = None,
        confidence: float = 0.0,
    ) -> float:
        """Binary reward: +1 if exact match, else -1. For inference, use confidence."""
        if ground_truth_files is None:
            return confidence * 2 - 1  # Map [0,1] to [-1,1]
        pred_set = set(predicted_files)
        truth_set = set(ground_truth_files)
        if pred_set == truth_set:
            return 1.0
        elif pred_set.issubset(truth_set) and len(pred_set) > 0:
            return 0.5 * (len(pred_set) / len(truth_set))
        elif len(pred_set.intersection(truth_set)) > 0:
            return -0.5
        return -1.0

    @staticmethod
    def code_editing_reward(
        tests_passed: bool,
        test_count: int = 0,
        regression: bool = False,
    ) -> float:
        """Reward: +1 if all tests pass, -1 if tests fail, -2 if regression."""
        if regression:
            return -2.0
        if tests_passed and test_count > 0:
            return 1.0
        if tests_passed and test_count == 0:
            return 0.5  # No tests available
        return -1.0

    @staticmethod
    def unit_test_generation_reward(
        mutation_score: float,
        bugs_caught: int,
        total_mutants: int,
        test_passes_on_correct: bool,
    ) -> float:
        """Reward: r = I[test passes on f] ∧ ∀f'∈B(f): test fails on f']."""
        if not test_passes_on_correct:
            return -1.0
        if total_mutants == 0:
            return 0.0
        return mutation_score  # Already [0,1]

    @staticmethod
    def issue_reproduction_reward(
        original_fails: bool,
        fixed_passes: bool,
        causal: bool,
    ) -> float:
        """Causal validation: fail original, pass fixed."""
        if original_fails and fixed_passes and causal:
            return 1.0
        if original_fails and fixed_passes:
            return 0.5
        if original_fails:
            return -0.5
        return -1.0

    @staticmethod
    def code_review_reward(
        judgment_correct: bool,
        confidence: float = 0.5,
    ) -> float:
        """Accuracy reward: +1 if matches ground truth, scaled by confidence."""
        if judgment_correct:
            return 0.5 + 0.5 * confidence
        return -1.0

    @staticmethod
    def compute_all(
        skill_name: str,
        result_data: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Dispatch to appropriate reward calculator."""
        if skill_name == "code_localization":
            return RewardCalculator.code_localization_reward(
                predicted_files=result_data.get("localized_files", []),
                ground_truth_files=ground_truth.get("files") if ground_truth else None,
                confidence=result_data.get("confidence", 0.0),
            )
        elif skill_name == "code_editing":
            return RewardCalculator.code_editing_reward(
                tests_passed=result_data.get("tests_passed", False),
                test_count=result_data.get("test_count", 0),
                regression=result_data.get("regression", False),
            )
        elif skill_name == "unit_test_generation":
            return RewardCalculator.unit_test_generation_reward(
                mutation_score=result_data.get("mutation_score", 0.0),
                bugs_caught=result_data.get("bugs_caught", 0),
                total_mutants=result_data.get("total_mutants", 0),
                test_passes_on_correct=result_data.get("test_passes_on_correct", False),
            )
        elif skill_name == "issue_reproduction":
            return RewardCalculator.issue_reproduction_reward(
                original_fails=result_data.get("original_fails", False),
                fixed_passes=result_data.get("fixed_passes", False),
                causal=result_data.get("causal", False),
            )
        elif skill_name == "code_review":
            return RewardCalculator.code_review_reward(
                judgment_correct=result_data.get("judgment_correct", False),
                confidence=result_data.get("confidence", 0.5),
            )
        return 0.0
