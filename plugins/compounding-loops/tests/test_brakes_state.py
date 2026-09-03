"""Tests for #1 (STATUS.json + heartbeat) and #2 (circuit breaker + step cap)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_HERMES_HOME = _PLUGIN_DIR.parent.parent  # ~/.hermes


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    os.environ.pop("HERMES_LOOPS_MAX_TURNS", None)
    os.environ.pop("HERMES_LOOPS_CIRCUIT_BREAKER", None)
    # Put ~/.hermes on the path so loop_state is importable.
    if str(_HERMES_HOME) not in sys.path:
        sys.path.insert(0, str(_HERMES_HOME))
    spec = importlib.util.spec_from_file_location(
        "plugins.cl_brakes", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.cl_brakes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """Isolate the loop-state file to a temp path."""
    p = tmp_path / "STATUS.json"
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(p))
    return p


def _pc(cid, name, args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


def _build_msgs(user, tool_calls=None, assistant_texts=None):
    msgs = [{"role": "user", "content": user}]
    if tool_calls:
        msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for tc in tool_calls:
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "tc"), "content": "ok"})
    if assistant_texts:
        for t in assistant_texts:
            msgs.append({"role": "assistant", "content": t})
    return msgs


# -----------------------------------------------------------------------------
# #1: STATUS.json write
# -----------------------------------------------------------------------------

def test_state_written_on_rejection(plugin, state_file):
    """A rejected exit must write a state snapshot with last_exit_verdict='rejected'."""
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[_pc("w1", "write_file"), _pc("w2", "patch")],
    )
    plugin._pre_exit_verify("Done.", messages=msgs, session_id="sess-1")
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["last_exit_verdict"] == "rejected"
    assert state["session_id"] == "sess-1"
    assert state["build_count"] >= 2
    assert "last_heartbeat" in state


def test_state_written_on_approval(plugin, state_file):
    """An approved exit (two clean passes) writes verdict='approved'."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    plugin._pre_exit_verify("Review pass 2: 0 blockers, 0 majors. Done.", messages=msgs)
    state = json.loads(state_file.read_text())
    assert state["last_exit_verdict"] == "approved"
    assert state["review_pass_count"] >= 2


def test_state_monotonic_build_count(plugin, state_file):
    """Build count climbs monotonically across calls (never regresses)."""
    msgs1 = _build_msgs("Implement X", tool_calls=[_pc("w1", "write_file"), _pc("w2", "patch")])
    plugin._pre_exit_verify("Review pass 1: 1 blocker.", messages=msgs1)
    after1 = json.loads(state_file.read_text())["build_count"]

    # Second call with fewer visible calls (simulating compaction) —
    # the state must not regress.
    msgs2 = _build_msgs("Implement X", tool_calls=[_pc("w1", "write_file")])
    plugin._pre_exit_verify("Still working.", messages=msgs2)
    after2 = json.loads(state_file.read_text())["build_count"]
    assert after2 >= after1


def test_state_write_failure_does_not_block_gate(plugin, monkeypatch):
    """If the state write fails, the gate must still return its decision."""
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", "/nonexistent/dir/that/cannot/exist/STATUS.json")
    msgs = _build_msgs("Implement X", tool_calls=[_pc("w1", "write_file"), _pc("w2", "patch")])
    # Should not raise.
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is False  # rejected for no review


def test_loop_state_read_returns_none_when_absent(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(tmp_path / "nope.json"))
    from loop_state import read_state
    assert read_state() is None


def test_loop_state_heartbeat_age(plugin, state_file):
    from loop_state import write_state, heartbeat_age_seconds
    write_state({"session_id": "s1", "build_count": 1})
    age = heartbeat_age_seconds()
    assert age is not None
    assert age >= 0
    assert age < 5  # just written


# -----------------------------------------------------------------------------
# #2: Step cap (HERMES_LOOPS_MAX_TURNS)
# -----------------------------------------------------------------------------

def test_step_cap_approves_when_exceeded(plugin, monkeypatch, state_file):
    """When total tool calls >= MAX_TURNS, the gate approves with a cap reason."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_TURNS", "2")
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[_pc("w1", "write_file"), _pc("w2", "patch")],
    )
    decision = plugin._pre_exit_verify("Review pass 1: 1 blocker.", messages=msgs)
    assert decision["approved"] is True
    assert "step cap" in decision["reason"].lower()
    state = json.loads(state_file.read_text())
    assert state["last_exit_verdict"] == "cap"


def test_step_cap_disabled_by_default(plugin, state_file):
    """MAX_TURNS=0 (default) means no step cap — gate engages normally."""
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[_pc("w1", "write_file"), _pc("w2", "patch")],
    )
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is False  # rejected, not capped


# -----------------------------------------------------------------------------
# #2: Circuit breaker (HERMES_LOOPS_CIRCUIT_BREAKER)
# -----------------------------------------------------------------------------

def test_circuit_breaker_trips_on_repeated_same_args(plugin, monkeypatch, state_file):
    """Same tool + same args 3x in a row → circuit breaker trips."""
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "3")
    same_args = '{"file": "foo.py", "content": "print(1)"}'
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[
            _pc("c1", "write_file", same_args),
            _pc("c2", "write_file", same_args),
            _pc("c3", "write_file", same_args),
        ],
    )
    decision = plugin._pre_exit_verify("Review pass 1: 1 blocker.", messages=msgs)
    assert decision["approved"] is True
    assert "circuit breaker" in decision["reason"].lower()
    state = json.loads(state_file.read_text())
    assert state["circuit_breaker_tripped"] is True


def test_circuit_breaker_does_not_trip_on_different_args(plugin, monkeypatch, state_file):
    """Different args → no trip."""
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "3")
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[
            _pc("c1", "write_file", '{"file": "a.py"}'),
            _pc("c2", "write_file", '{"file": "b.py"}'),
            _pc("c3", "write_file", '{"file": "c.py"}'),
        ],
    )
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    # Not a circuit breaker trip — normal rejection for no review.
    assert decision["approved"] is False
    assert "circuit breaker" not in decision.get("reason", "").lower()


def test_circuit_breaker_disabled_by_default(plugin, state_file):
    msgs = _build_msgs(
        "Implement X",
        tool_calls=[_pc("c1", "write_file", '{"x": 1}')] * 5,
    )
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert "circuit breaker" not in decision.get("reason", "").lower()


def test_circuit_breaker_detect_signatures(plugin):
    """The _detect_circuit_breaker helper returns (tool, args) on trip."""
    same = '{"file": "x.py"}'
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("a", "write_file", same), _pc("b", "write_file", same)]},
    ]
    tool, args = plugin._detect_circuit_breaker(msgs, 2)
    assert tool == "write_file"
    assert "x.py" in args

    diff_msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("a", "write_file", '{"f": "1"}'), _pc("b", "write_file", '{"f": "2"}')]},
    ]
    tool, args = plugin._detect_circuit_breaker(diff_msgs, 2)
    assert tool is None


def test_step_cap_safe_fallback_on_garbage(plugin, monkeypatch):
    """Garbage MAX_TURNS value falls back to 0 (disabled), not crash."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_TURNS", "not-a-number")
    assert plugin._config()["max_turns"] == 0


def test_circuit_breaker_safe_fallback_on_garbage(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "garbage")
    assert plugin._config()["circuit_breaker"] == 0