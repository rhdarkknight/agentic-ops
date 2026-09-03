"""Regression coverage for session-scoped harness-conductor hook state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def plugin():
    spec = importlib.util.spec_from_file_location(
        "harness_conductor_state_isolation", _PLUGIN_DIR / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._HARNESS_VERDICTS.clear()
    module._HARNESS_TOOL_ERRORS.clear()
    return module


def test_anonymous_stateful_hooks_do_not_retain_or_consume_state(plugin):
    plugin._HARNESS_VERDICTS["known"] = {"approved": False, "reason": "retry"}

    assert plugin._post_api_request(
        finish_reason="stop", assistant_message=SimpleNamespace(content="")
    ) is None
    assert plugin._transform_llm_output(response_text="done") is None
    assert plugin._HARNESS_VERDICTS == {
        "known": {"approved": False, "reason": "retry"}
    }


def _seed_all_state_maps(plugin, *session_ids: str) -> None:
    for session_id in session_ids:
        plugin._HARNESS_VERDICTS[session_id] = {"approved": False}
        plugin._HARNESS_TOOL_ERRORS[session_id] = ["tool error"]


def test_on_session_end_is_not_registered_and_preserves_cross_turn_state(plugin):
    registered = {}
    plugin.register(SimpleNamespace(register_hook=registered.__setitem__))
    _seed_all_state_maps(plugin, "turn")

    # on_session_end is a turn boundary, not a conversation lifetime event.
    assert "on_session_end" not in registered
    assert registered["on_session_finalize"] is plugin._on_session_finalize
    assert plugin._on_session_end(
        session_id="turn", finalizer_reason="turn-complete", telemetry={"n": 1}
    ) is None
    assert "turn" in plugin._HARNESS_VERDICTS
    assert "turn" in plugin._HARNESS_TOOL_ERRORS


@pytest.mark.parametrize("callback", ["_on_session_reset", "_on_session_finalize"])
def test_lifecycle_cleanup_removes_old_and_new_session_state(plugin, callback):
    _seed_all_state_maps(plugin, "old", "new")

    getattr(plugin, callback)(
        old_session_id="old",
        new_session_id="new",
        finalizer_reason="conversation-complete",
        finalizer_metadata={"safe": True},
    )

    assert not plugin._HARNESS_VERDICTS
    assert not plugin._HARNESS_TOOL_ERRORS
