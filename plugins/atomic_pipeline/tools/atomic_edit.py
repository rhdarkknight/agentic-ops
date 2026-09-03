"""Tool: atomic_code_edit — wrapper for CodeEditingSkill."""
import json
from typing import Any, Dict, Optional

from ..skills.code_editing import CodeEditingSkill

_skill = CodeEditingSkill()


def atomic_code_edit(
    file_path: str,
    edit_instruction: str,
    run_tests: bool = True,
    **kwargs,
) -> str:
    """Atomic skill: generate a code patch from edit instruction.
    
    Args:
        file_path: Path to the file to edit
        edit_instruction: Natural language description of the change
        run_tests: Whether to run tests after editing (default: True)
    
    Returns:
        JSON string with file, patch, tests_passed, test_output
    """
    result = _skill.execute(
        file_path=file_path,
        edit_instruction=edit_instruction,
        run_tests=run_tests,
        **kwargs,
    )
    return result.to_json()
