"""Regression coverage for atomic pipeline session lifecycle boundaries."""
from pathlib import Path

import pytest

from .. import register
from ..core.orchestrator import AtomicOrchestrator
from ..hooks.session_hooks import on_session_end, on_session_finalize, on_session_reset


@pytest.fixture(autouse=True)
def reset_orchestrator():
    """Keep the singleton state isolated across lifecycle regressions."""
    AtomicOrchestrator._instances = {}
    AtomicOrchestrator._current_state = None
    yield
    AtomicOrchestrator._instances = {}
    AtomicOrchestrator._current_state = None


class _RecordingContext:
    def __init__(self):
        self.hooks = {}

    def register_tool(self, **_kwargs):
        pass

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_registers_only_true_session_boundary_hooks():
    context = _RecordingContext()
    register(context)

    assert context.hooks["on_session_start"]
    assert context.hooks["on_session_reset"] is on_session_reset
    assert context.hooks["on_session_finalize"] is on_session_finalize
    assert "on_session_end" not in context.hooks

    manifest = (Path(__file__).resolve().parents[1] / "plugin.yaml").read_text()
    assert "version: 1.0.2" in manifest
    for hook_name in context.hooks:
        assert f"  - {hook_name}" in manifest
    assert "  - on_session_end" not in manifest


def test_session_end_is_abi_safe_non_destructive_compatibility_no_op():
    state = AtomicOrchestrator.init_session("multi-turn-session")

    result = on_session_end(
        session_id="multi-turn-session",
        completed=True,
        interrupted=False,
        platform="test",
        future_lifecycle_metadata="ignored",
    )

    assert result is None
    assert AtomicOrchestrator.get_state() is state
    assert state.completed is False


def test_session_finalize_clears_matching_state_with_extra_lifecycle_kwargs():
    AtomicOrchestrator.init_session("terminal-session")

    result = on_session_finalize(
        session_id="terminal-session",
        completed=True,
        reason="conversation_closed",
        future_lifecycle_metadata="ignored",
    )

    assert result["session_id"] == "terminal-session"
    assert result["finalized"] is True
    assert AtomicOrchestrator.get_state() is None


@pytest.mark.parametrize("id_key", ["old_session_id", "session_id", "new_session_id"])
def test_session_reset_clears_each_supported_session_id_with_abi_safe_kwargs(id_key):
    session_id = f"reset-{id_key}"
    AtomicOrchestrator.init_session(session_id)

    result = on_session_reset(
        **{
            id_key: session_id,
            "reason": "session_replaced",
            "future_lifecycle_metadata": "ignored",
        }
    )

    assert result is None
    assert AtomicOrchestrator.get_state() is None
