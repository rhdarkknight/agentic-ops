"""Tests for atomic_pipeline pre_llm_call hook contract.

The hook's return value is consumed by agent/turn_context.py:431-456,
which reads only the ``context`` field. This test pins that contract so
the dead model-routing code does not get reintroduced.
"""
from types import SimpleNamespace
from unittest.mock import patch

from ..hooks.pre_llm_hook import on_pre_llm_call
from ..core.orchestrator import AtomicOrchestrator


class _StubSkill:
    def get_system_prompt(self) -> str:
        return "STUB_SKILL_PROMPT"


def test_hook_returns_only_context_when_skill_active():
    """When a skill is active, hook returns dict with exactly the context key."""
    fake_state = SimpleNamespace(
        active_skill="code_localization",
        _skills={"code_localization": _StubSkill()},
    )
    with patch.object(AtomicOrchestrator, "get_state", return_value=fake_state):
        result = on_pre_llm_call(
            session_id="s1",
            user_message="hi",
            conversation_history=[{"role": "user", "content": "hi"}],
            is_first_turn=True,
            model="some-model",
            platform="cli",
            sender_id="user",
        )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"context"}, f"unexpected keys: {set(result.keys())}"
    assert "STUB_SKILL_PROMPT" in result["context"]
    # Must NOT return model/messages keys (runtime drops them anyway,
    # but the hook should not promise them)
    assert "model" not in result
    assert "messages" not in result


def test_hook_returns_empty_dict_when_no_skill():
    """When no skill is active, hook returns empty dict (no context)."""
    fake_state = SimpleNamespace(active_skill=None, _skills={})
    with patch.object(AtomicOrchestrator, "get_state", return_value=fake_state):
        result = on_pre_llm_call(
            session_id="s1",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
        )
    assert result == {}, f"expected empty dict, got {result}"


def test_hook_handles_unknown_skill_gracefully():
    """If active_skill is set but not in _skills, hook still returns context."""
    fake_state = SimpleNamespace(active_skill="nonexistent_skill", _skills={})
    with patch.object(AtomicOrchestrator, "get_state", return_value=fake_state):
        result = on_pre_llm_call(
            session_id="s1",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
        )
    assert "context" in result
    assert "nonexistent_skill" in result["context"]
