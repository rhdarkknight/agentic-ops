"""Smoke tests for the runtime loop hooks.

These tests verify that ``post_tool_batch_reflect`` and ``pre_exit_verify``
are registered in the agent's ``VALID_HOOKS`` and that the conversation
loop module imports cleanly.

They are tolerant of layout: when run from inside the hermes-agent tree
(plugins/compounding-loops/tests/), the agent modules are importable
directly. When run from the standalone plugin dir
(~/.hermes/plugins/compounding-loops/tests/), the agent tree is not on
the path, so the hook-registration assertions are skipped via
``importorskip`` rather than failing — the unit tests in
``test_compounding_loops.py`` cover the plugin logic independent of the
agent tree.
"""

import sys
from pathlib import Path

import pytest


_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent


def _agent_on_path() -> bool:
    try:
        import hermes_cli.plugins  # noqa: F401
        return True
    except Exception:
        return False


def test_post_tool_batch_reflect_hook_is_registered():
    if not _agent_on_path():
        pytest.skip("hermes_cli not importable from this layout")
    from hermes_cli.plugins import VALID_HOOKS
    assert "post_tool_batch_reflect" in VALID_HOOKS


def test_pre_exit_verify_hook_is_registered():
    if not _agent_on_path():
        pytest.skip("hermes_cli not importable from this layout")
    from hermes_cli.plugins import VALID_HOOKS
    assert "pre_exit_verify" in VALID_HOOKS


def test_conversation_loop_imports_cleanly():
    if not _agent_on_path():
        pytest.skip("agent.conversation_loop not importable from this layout")
    from agent import conversation_loop as cl
    assert hasattr(cl, "run_conversation")