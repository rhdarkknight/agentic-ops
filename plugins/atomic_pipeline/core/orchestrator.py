"""Orchestrator — manages state machine across atomic skills."""
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .classifier import ClassificationResult, TaskClassifier
from .rewards import RewardCalculator
from .skill_base import SkillResult


@dataclass
class SkillExecutionRecord:
    skill: str
    result: SkillResult
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "success": self.result.success,
            "reward": self.result.reward,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class PipelineState:
    session_id: str
    active_skill: Optional[str] = None
    skill_records: List[SkillExecutionRecord] = field(default_factory=list)
    is_composite: bool = False
    completed: bool = False
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "active_skill": self.active_skill,
            "skill_records": [r.to_dict() for r in self.skill_records],
            "is_composite": self.is_composite,
            "completed": self.completed,
        }


class AtomicOrchestrator:
    """Pipeline coordinator — manages state machine across atomic skills."""

    _instances: Dict[str, "AtomicOrchestrator"] = {}
    _current_state: Optional[PipelineState] = None

    def __init__(self):
        self.classifier = TaskClassifier()
        self.skills: Dict[str, Any] = {}
        self.llm_callback: Optional[Callable] = None

    @classmethod
    def get_instance(cls) -> "AtomicOrchestrator":
        if "default" not in cls._instances:
            cls._instances["default"] = cls()
        return cls._instances["default"]

    @classmethod
    def init_session(cls, session_id: str):
        state = PipelineState(session_id=session_id)
        cls._current_state = state
        return state

    @classmethod
    def get_state(cls) -> Optional[PipelineState]:
        return cls._current_state

    @classmethod
    def finalize_session(cls, session_id: str):
        if cls._current_state and cls._current_state.session_id == session_id:
            cls._current_state.completed = True
            cls._current_state = None

    def register_skill(self, name: str, skill_instance):
        self.skills[name] = skill_instance

    def set_llm_callback(self, callback: Callable):
        self.llm_callback = callback

    def classify_request(self, user_message: str) -> ClassificationResult:
        return self.classifier.classify(user_message, self.llm_callback)

    def execute_skill(self, skill_name: str, **kwargs) -> SkillResult:
        if skill_name not in self.skills:
            return SkillResult.failure(skill_name, f"Unknown skill: {skill_name}")

        if self._current_state:
            self._current_state.active_skill = skill_name

        skill = self.skills[skill_name]
        start = time.time()
        result = skill.execute(**kwargs)
        duration_ms = (time.time() - start) * 1000

        result.reward = RewardCalculator.compute_all(skill_name, result.data)

        if self._current_state:
            record = SkillExecutionRecord(skill=skill_name, result=result, duration_ms=duration_ms)
            self._current_state.skill_records.append(record)
            self._current_state.active_skill = None

        return result

    def execute_composite(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        classification = self.classify_request(user_message)

        if not classification.is_composite:
            return {
                "success": True,
                "composite": False,
                "message": "Simple query — no decomposition needed",
                "classification": classification.to_dict(),
            }

        results = []
        for skill_name in classification.suggested_order:
            result = self.execute_skill(skill_name, query=user_message, context=context or {})
            results.append(result)
            if not result.success:
                break

        total_reward = sum(r.reward for r in results)
        return {
            "success": all(r.success for r in results),
            "composite": True,
            "classification": classification.to_dict(),
            "skill_results": [r.to_dict() for r in results],
            "total_reward": total_reward,
        }

    def get_trajectory(self) -> Dict[str, Any]:
        if not self._current_state:
            return {}
        return {
            "session_id": self._current_state.session_id,
            "skills_executed": [r.to_dict() for r in self._current_state.skill_records],
            "composite_outcome": "success" if self._current_state.completed else "in_progress",
        }
