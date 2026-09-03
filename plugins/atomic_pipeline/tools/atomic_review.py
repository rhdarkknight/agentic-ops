"""Tool: atomic_code_review — wrapper for CodeReviewSkill."""
import json
from typing import Any, Dict, List, Optional

from ..skills.code_review import CodeReviewSkill

_skill = CodeReviewSkill()


def atomic_code_review(
    pr_diff: str,
    issue_description: str,
    pr_files: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Atomic skill: judge whether a PR correctly addresses the issue.
    
    Args:
        pr_diff: Unified diff of the PR changes
        issue_description: Original issue/bug description
        pr_files: Optional list of files changed in the PR
    
    Returns:
        JSON string with judgment, confidence, evidence, concerns
    """
    result = _skill.execute(
        pr_diff=pr_diff,
        issue_description=issue_description,
        pr_files=pr_files,
        **kwargs,
    )
    return result.to_json()
