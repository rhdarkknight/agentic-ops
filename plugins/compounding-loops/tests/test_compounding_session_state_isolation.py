"""Regression coverage for session-scoped compounding-loop hook state."""

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
        "compounding_loops_state_isolation", _PLUGIN_DIR / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._LOOP_VERDICTS.clear()
    module._SESSION_CLEAN_STREAK.clear()
    module._VERDICT_GENERATION.clear()
    module._REJECTION_STREAK.clear()
    module._CAP_QUIET_STREAK.clear()
    module._LAST_REFLECT_TAIL.clear()
    return module


def _review_message() -> SimpleNamespace:
    return SimpleNamespace(
        content="Review pass 1: 0 blockers, 0 majors. Review clean.",
        tool_calls=None,
    )


def _manifest_hooks(section: str) -> list[str]:
    values = []
    in_section = False
    for line in (_PLUGIN_DIR / "plugin.yaml").read_text().splitlines():
        if line == f"{section}:":
            in_section = True
            continue
        if in_section and line.startswith("  - "):
            values.append(line.removeprefix("  - "))
        elif in_section:
            break
    return values


def test_anonymous_stateful_hooks_are_noops(plugin):
    plugin._LOOP_VERDICTS["known"] = {"verdict": "rejected", "generation": 1}
    plugin._VERDICT_GENERATION["known"] = 1

    assert plugin._post_api_request(
        finish_reason="stop", assistant_message=_review_message()
    ) is None
    assert plugin._pre_llm_call() is None
    assert plugin._transform_llm_output(response_text="done") is None
    assert plugin._post_llm_call() is None
    assert plugin._post_tool_batch_reflect(
        [{"role": "user", "content": "build it"}]
    ) == {"reflect": False}

    assert set(plugin._LOOP_VERDICTS) == {"known"}
    assert plugin._VERDICT_GENERATION == {"known": 1}
    assert not plugin._SESSION_CLEAN_STREAK
    assert not plugin._LAST_REFLECT_TAIL


def test_session_b_verdict_does_not_invalidate_pending_session_a_verdict(plugin):
    plugin._post_api_request(
        finish_reason="stop", assistant_message=_review_message(), session_id="A"
    )
    plugin._post_api_request(
        finish_reason="stop", assistant_message=_review_message(), session_id="B"
    )

    injected = plugin._pre_llm_call(session_id="A")

    assert injected is not None
    assert "[compounding-loops]" in injected["context"]
    assert "A" not in plugin._LOOP_VERDICTS
    assert "B" in plugin._LOOP_VERDICTS


def test_same_session_generation_refresh_still_drops_stale_verdict(plugin):
    session_id = "same-session"
    plugin._LOOP_VERDICTS[session_id] = {
        "verdict": "rejected",
        "reason": "old verdict",
        "generation": 1,
    }
    plugin._VERDICT_GENERATION[session_id] = 2

    assert plugin._pre_llm_call(session_id=session_id) is None
    assert session_id not in plugin._LOOP_VERDICTS


def _seed_all_state_maps(plugin, *session_ids: str) -> None:
    for session_id in session_ids:
        plugin._LOOP_VERDICTS[session_id] = {}
        plugin._SESSION_CLEAN_STREAK[session_id] = []
        plugin._VERDICT_GENERATION[session_id] = 1
        plugin._REJECTION_STREAK[session_id] = {}
        plugin._CAP_QUIET_STREAK[session_id] = 1
        plugin._LAST_REFLECT_TAIL[session_id] = ()


def _all_state_maps(plugin):
    return (
        plugin._LOOP_VERDICTS,
        plugin._SESSION_CLEAN_STREAK,
        plugin._VERDICT_GENERATION,
        plugin._REJECTION_STREAK,
        plugin._CAP_QUIET_STREAK,
        plugin._LAST_REFLECT_TAIL,
    )


def test_manifest_matches_registered_lifecycle_hooks(plugin):
    registered = {}
    plugin.register(SimpleNamespace(register_hook=registered.__setitem__))

    assert set(_manifest_hooks("hooks")) == set(registered)
    assert set(_manifest_hooks("provides_hooks")) == set(registered)
    assert "on_session_end" not in registered
    assert registered["on_session_reset"] is plugin._on_session_reset
    assert registered["on_session_finalize"] is plugin._on_session_finalize


def test_on_session_end_is_not_registered_and_preserves_cross_turn_state(plugin):
    registered = {}
    plugin.register(SimpleNamespace(register_hook=registered.__setitem__))
    _seed_all_state_maps(plugin, "turn")

    # The core fires this after every run, so neither registration nor the
    # compatibility callback may destructively clear state needed next turn.
    assert "on_session_end" not in registered
    assert plugin._on_session_end(
        session_id="turn", finalizer_reason="turn-complete", telemetry={"n": 1}
    ) is None
    assert all("turn" in state for state in _all_state_maps(plugin))


@pytest.mark.parametrize("callback", ["_on_session_reset", "_on_session_finalize"])
def test_lifecycle_cleanup_removes_old_and_new_session_state(plugin, callback):
    _seed_all_state_maps(plugin, "old", "new")

    getattr(plugin, callback)(
        old_session_id="old",
        session_id="new",
        finalizer_reason="conversation-complete",
        finalizer_metadata={"safe": True},
    )

    for state in _all_state_maps(plugin):
        assert not state
