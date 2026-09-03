"""Tool: atomic_unit_test_gen — wrapper for UnitTestGenSkill."""
import json
from typing import Any, Dict, Optional

from ..skills.unit_test_gen import UnitTestGenSkill

_skill = UnitTestGenSkill()


def atomic_unit_test_gen(
    function_code: str,
    specification: str,
    test_file: Optional[str] = None,
    **kwargs,
) -> str:
    """Atomic skill: generate unit tests with mutation-based fault detection.
    
    Args:
        function_code: Source code of the function to test
        specification: Description of expected behavior
        test_file: Optional path for the generated test file
    
    Returns:
        JSON string with test_file, test_cases count, mutation_score, bugs_caught, total_mutants
    """
    result = _skill.execute(
        function_code=function_code,
        specification=specification,
        test_file=test_file,
        **kwargs,
    )
    return result.to_json()
