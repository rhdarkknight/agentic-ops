"""Tool: atomic_code_localize — wrapper for CodeLocalizationSkill."""
import json
from typing import Any, Dict, Optional

from ..skills.code_localization import CodeLocalizationSkill

_skill = CodeLocalizationSkill()


def atomic_code_localize(
    issue: str,
    repo_context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """Atomic skill: identify files relevant to an issue.
    
    Args:
        issue: Issue description text
        repo_context: Optional dict with 'files' key listing repo files
    
    Returns:
        JSON string with localized_files, confidence, reasoning
    """
    result = _skill.execute(issue=issue, repo_context=repo_context, **kwargs)
    return result.to_json()
