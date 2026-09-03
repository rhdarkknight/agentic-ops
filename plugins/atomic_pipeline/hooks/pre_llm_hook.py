"""pre_llm_call hook — inject pipeline context into the user message.

CONTRACT (per agent/turn_context.py:431-456):
    pre_llm_call hooks may return a dict with a ``context`` field. ONLY
    the ``context`` field is consumed — the runtime appends it to the
    current turn's user message after the API call boundary. ``model``,
    ``messages``, and any other keys are silently dropped.

    Per-skill model routing via this hook is NOT supported. If you need
    per-skill models, route them through the core ``model`` config or a
    cron job's per-job override, not through this hook.
"""
from typing import Any, Dict, List, Optional

from ..core.orchestrator import AtomicOrchestrator


def on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    is_first_turn: bool = False,
    model: Optional[str] = None,
    platform: str = "",
    sender_id: str = "",
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Inject atomic pipeline context into the user message.

    Note on the kwarg name: ``conversation_history`` is the dispatcher's
    actual kwarg (was previously ``messages``). The body aliases to
    ``messages`` for readability, but only the context string is
    returned — the runtime ignores any messages/model keys.
    """
    state = AtomicOrchestrator.get_state()
    if not state or not state.active_skill:
        # Silent no-op: no active skill means no pipeline context to inject.
        # Returning an empty dict matches the hook contract — anything
        # truthy would be appended, so empty is correct.
        return {}

    skill_name = state.active_skill
    skills = state.__dict__.get("_skills", {})
    if skill_name in skills:
        skill_prompt = skills[skill_name].get_system_prompt()
    else:
        skill_prompt = f"Execute skill: {skill_name}"

    return {"context": f"[Atomic Pipeline]\n{skill_prompt}"}
