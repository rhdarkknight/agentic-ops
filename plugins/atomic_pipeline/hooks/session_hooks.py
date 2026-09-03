"""Session hooks for pipeline initialization and true session teardown."""
from typing import Any, Dict, Tuple

from ..core.orchestrator import AtomicOrchestrator
from ..skills import (
    CodeLocalizationSkill,
    CodeEditingSkill,
    UnitTestGenSkill,
    IssueReproductionSkill,
    CodeReviewSkill,
)


def on_session_start(session_id: str, **kwargs) -> Dict[str, Any]:
    """Initialize pipeline state at session start.

    Creates orchestrator instance, registers all 5 atomic skills,
    and loads default configuration.
    """
    state = AtomicOrchestrator.init_session(session_id)
    orchestrator = AtomicOrchestrator.get_instance()

    orchestrator.register_skill("code_localization", CodeLocalizationSkill())
    orchestrator.register_skill("code_editing", CodeEditingSkill())
    orchestrator.register_skill("unit_test_generation", UnitTestGenSkill())
    orchestrator.register_skill("issue_reproduction", IssueReproductionSkill())
    orchestrator.register_skill("code_review", CodeReviewSkill())

    return {
        "session_id": session_id,
        "pipeline_initialized": True,
        "skills_registered": list(orchestrator.skills.keys()),
    }


def _session_ids_for_cleanup(kwargs: Dict[str, Any]) -> Tuple[str, ...]:
    """Return each lifecycle session ID once, tolerating dispatcher metadata."""
    session_ids = []
    for key in ("old_session_id", "session_id", "new_session_id"):
        session_id = kwargs.get(key)
        if isinstance(session_id, str) and session_id and session_id not in session_ids:
            session_ids.append(session_id)
    return tuple(session_ids)


def _finalize_supplied_sessions(**kwargs: Any) -> None:
    for session_id in _session_ids_for_cleanup(kwargs):
        AtomicOrchestrator.finalize_session(session_id)


def on_session_end(**kwargs: Any) -> None:
    """Compatibility no-op: the core invokes this after every turn/run."""
    return None


def on_session_reset(**kwargs: Any) -> None:
    """Discard pipeline state for all IDs supplied by a session replacement."""
    _finalize_supplied_sessions(**kwargs)
    return None


def on_session_finalize(**kwargs: Any) -> Dict[str, Any]:
    """Finalize pipeline state only when the conversation lifetime ends.

    ``**kwargs`` permits lifecycle dispatchers to add metadata or provide old,
    current, and replacement IDs without breaking the callback ABI.
    """
    session_ids = _session_ids_for_cleanup(kwargs)
    trajectory = AtomicOrchestrator.get_instance().get_trajectory()
    _finalize_supplied_sessions(**kwargs)

    return {
        "session_id": kwargs.get("session_id") or (session_ids[0] if session_ids else None),
        "trajectory": trajectory,
        "finalized": bool(session_ids),
    }
