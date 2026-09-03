"""Base class for atomic skills — handoff format, tool allowlist, success criteria."""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillResult:
    """Standardized result from any atomic skill execution."""
    skill: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "skill": self.skill,
            "success": self.success,
            "data": self.data,
            "reward": self.reward,
            "error": self.error,
        })

    @classmethod
    def failure(cls, skill: str, error: str) -> "SkillResult":
        return cls(skill=skill, success=False, data={}, reward=-1.0, error=error)

    @classmethod
    def success(cls, skill: str, data: Dict[str, Any], reward: float = 1.0) -> "SkillResult":
        return cls(skill=skill, success=True, data=data, reward=reward)


class SkillBase(ABC):
    """Abstract base for all atomic skills."""

    name: str = ""
    description: str = ""
    tool_allowlist: List[str] = []

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """Execute the skill. Returns SkillResult with standardized handoff format."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return skill-specific system prompt for LLM injection."""
        pass

    @abstractmethod
    def validate_result(self, result: SkillResult) -> bool:
        """Validate that result meets success criteria."""
        pass

    def get_handoff_format(self) -> dict:
        """Return the expected handoff JSON structure for this skill."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "tool_allowlist": self.tool_allowlist,
        }
