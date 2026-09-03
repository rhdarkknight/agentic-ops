"""post_tool_call hook — track skill execution and update scores."""
import json
from typing import Any, Dict

from ..core.orchestrator import AtomicOrchestrator


def on_post_tool_call(
    tool_name: str,
    result: Any,
    **kwargs,
) -> Dict[str, Any]:
    """Track skill execution and update scores after tool call.

    Intercepts atomic_* tool results, parses them, and records
    the skill execution in the orchestrator state.

    NOTE: dispatcher passes ``result`` kwarg, not ``tool_result``.
    Previously mismatched → TypeError on every post_tool_call, WARNING
    every turn.
    """
    if not tool_name.startswith("atomic_"):
        return {}

    state = AtomicOrchestrator.get_state()
    if not state:
        return {}

    try:
        result_data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {}

    skill_name = tool_name.replace("atomic_", "")
    AtomicOrchestrator.record_skill_result(skill_name, result_data)

    return {}


# Add record_skill_result to orchestrator if not present
_original_init = AtomicOrchestrator.__init__

def _patched_init(self):
    _original_init(self)
    self._skill_scores = {}

AtomicOrchestrator.__init__ = _patched_init

_original_record = getattr(AtomicOrchestrator, "record_skill_result", None)

def record_skill_result(cls, skill_name, result_data):
    if not hasattr(cls, "_skill_scores"):
        cls._skill_scores = {}
    cls._skill_scores[skill_name] = result_data

if not _original_record:
    AtomicOrchestrator.record_skill_result = classmethod(record_skill_result)
