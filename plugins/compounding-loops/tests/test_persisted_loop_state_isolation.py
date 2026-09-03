"""Regression coverage for session-isolated STATUS.json persistence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loop_state():
    return _load_module("compounding_loops_loop_state_isolation", _PLUGIN_DIR / "loop_state.py")


@pytest.fixture
def plugin():
    return _load_module("compounding_loops_persisted_state_isolation", _PLUGIN_DIR / "__init__.py")


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "STATUS.json"
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(path))
    return path


def test_named_sessions_do_not_inherit_numeric_counters(loop_state, state_file):
    loop_state.write_state(
        {"session_id": "session-a", "build_count": 9, "review_pass_count": 4, "turn_count": 12}
    )
    loop_state.write_state(
        {"session_id": "session-b", "build_count": 1, "review_pass_count": 1, "turn_count": 2}
    )

    state = loop_state.read_state()
    assert state is not None
    assert state["session_id"] == "session-b"
    assert state["build_count"] == 1
    assert state["review_pass_count"] == 1
    assert state["turn_count"] == 2

    # Monotonic merging remains available for subsequent writes in session B.
    loop_state.write_state({"session_id": "session-b", "build_count": 0, "turn_count": 1})
    same_session = loop_state.read_state()
    assert same_session is not None
    assert same_session["build_count"] == 1
    assert same_session["turn_count"] == 2


def test_anonymous_and_named_snapshots_do_not_cross_contaminate(loop_state, state_file):
    loop_state.write_state({"session_id": "", "build_count": 13, "turn_count": 21})
    loop_state.write_state({"session_id": "named", "build_count": 2, "turn_count": 3})

    named = loop_state.read_state()
    assert named is not None
    assert named["session_id"] == "named"
    assert named["build_count"] == 2
    assert named["turn_count"] == 3

    loop_state.write_state({"session_id": "", "build_count": 1, "turn_count": 1})
    anonymous = loop_state.read_state()
    assert anonymous is not None
    assert anonymous["session_id"] == ""
    assert anonymous["build_count"] == 1
    assert anonymous["turn_count"] == 1


@pytest.mark.parametrize("callback", ["_on_session_finalize", "_on_session_reset"])
def test_lifecycle_cleanup_only_removes_its_own_persisted_snapshot(
    plugin, loop_state, state_file, callback
):
    loop_state.write_state({"session_id": "current", "build_count": 3})

    # A delayed lifecycle event for another conversation must leave the newer
    # singleton STATUS.json snapshot in place.
    getattr(plugin, callback)(session_id="stale")
    assert state_file.exists()
    assert loop_state.read_state()["session_id"] == "current"

    # The matching event removes the persisted snapshot, just as it clears the
    # corresponding in-memory state maps.
    getattr(plugin, callback)(session_id="current")
    assert not state_file.exists()
