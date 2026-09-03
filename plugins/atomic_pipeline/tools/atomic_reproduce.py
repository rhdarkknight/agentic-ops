"""Tool: atomic_issue_reproduce — wrapper for IssueReproductionSkill."""
import json
from typing import Any, Dict, Optional

from ..skills.issue_reproduction import IssueReproductionSkill

_skill = IssueReproductionSkill()


def atomic_issue_reproduce(
    issue_description: str,
    repo_context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """Atomic skill: create a script that reproduces the reported failure.
    
    Args:
        issue_description: Description of the bug/issue
        repo_context: Optional dict with repo context (files, structure)
    
    Returns:
        JSON string with script, original_exit_code, original_stderr, fixed_exit_code, causal
    """
    result = _skill.execute(
        issue_description=issue_description,
        repo_context=repo_context,
        **kwargs,
    )
    return result.to_json()
