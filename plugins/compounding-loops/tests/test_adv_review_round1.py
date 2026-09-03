"""Adversarial probes for #1/#2/#8/#5/#6 — interaction edges and bug hunting."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_HERMES_HOME = _PLUGIN_DIR.parent.parent


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    os.environ.pop("HERMES_LOOPS_MAX_TURNS", None)
    os.environ.pop("HERMES_LOOPS_CIRCUIT_BREAKER", None)
    os.environ.pop("HERMES_LOOPS_PROOF_OF_FIX", None)
    if str(_HERMES_HOME) not in sys.path:
        sys.path.insert(0, str(_HERMES_HOME))
    spec = importlib.util.spec_from_file_location(
        "plugins.cl_adv2", _PLUGIN_DIR / "__init__.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["plugins.cl_adv2"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    p = tmp_path / "STATUS.json"
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(p))
    return p


def _pc(cid, name, args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


# -----------------------------------------------------------------------------
# #1: state file — concurrency / corruption resilience
# -----------------------------------------------------------------------------

def test_state_survives_corrupt_json(plugin, tmp_path, monkeypatch):
    """A corrupt STATUS.json must not crash the gate — read returns None,
    the gate proceeds normally, and the next write replaces the corrupt file."""
    p = tmp_path / "STATUS.json"
    p.write_text("{corrupt json")
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(p))
    from loop_state import read_state
    assert read_state() is None  # corrupt file → None, no crash
    # The gate still works
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
    ]
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is False  # rejected, no crash


def test_state_write_replaces_corrupt_file(plugin, tmp_path, monkeypatch):
    """After a gate write, a previously-corrupt file is replaced with valid JSON."""
    p = tmp_path / "STATUS.json"
    p.write_text("{corrupt")
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(p))
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
    ]
    plugin._pre_exit_verify("Done.", messages=msgs)
    # File should now be valid JSON
    state = json.loads(p.read_text())
    assert state["last_exit_verdict"] == "rejected"


def test_state_monotonic_climb_with_negative_values(plugin, state_file):
    """The monotonic-merge logic should not climb on negative values
    (there are none in practice, but verify the logic doesn't break)."""
    from loop_state import write_state, read_state
    write_state({"turn_count": 10})
    assert read_state()["turn_count"] == 10
    write_state({"turn_count": 5})  # lower
    # Monotonic: should stay at 10, not regress to 5
    assert read_state()["turn_count"] == 10


# -----------------------------------------------------------------------------
# #2: circuit breaker — args truncation edge cases
# -----------------------------------------------------------------------------

def test_circuit_breaker_truncates_long_args(plugin):
    """Very long args should be truncated in the signature but still
    detected as identical across calls."""
    long_args = '{"content": "' + "x" * 5000 + '"}'
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("a", "write_file", long_args),
                        _pc("b", "write_file", long_args),
                        _pc("c", "write_file", long_args)]},
    ]
    tool, args = plugin._detect_circuit_breaker(msgs, 3)
    assert tool == "write_file"  # same long args → trip


def test_circuit_breaker_different_tools_same_args_no_trip(plugin):
    """Same args but different tool names → NOT a trip (different signatures)."""
    same_args = '{"file": "x.py"}'
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("a", "write_file", same_args),
                        _pc("b", "patch", same_args),
                        _pc("c", "write_file", same_args)]},
    ]
    tool, args = plugin._detect_circuit_breaker(msgs, 3)
    assert tool is None  # signatures differ (tool name differs)


def test_circuit_breaker_spans_multiple_messages(plugin):
    """The circuit breaker should detect repetition across multiple
    assistant messages, not just within one."""
    same = '{"f": "x.py"}'
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [_pc("a", "write_file", same)]},
        {"role": "assistant", "content": None, "tool_calls": [_pc("b", "write_file", same)]},
        {"role": "assistant", "content": None, "tool_calls": [_pc("c", "write_file", same)]},
    ]
    tool, args = plugin._detect_circuit_breaker(msgs, 3)
    assert tool == "write_file"


def test_circuit_breaker_non_build_tool_also_detected(plugin):
    """The circuit breaker detects ANY tool repetition, not just build tools
    — this is correct because runaway recursion can happen with any tool."""
    same = '{"query": "SELECT 1"}'
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("a", "execute_code", same),
                        _pc("b", "execute_code", same),
                        _pc("c", "execute_code", same)]},
    ]
    tool, args = plugin._detect_circuit_breaker(msgs, 3)
    assert tool == "execute_code"


# -----------------------------------------------------------------------------
# #8: proof-of-fix regex edge cases
# -----------------------------------------------------------------------------

def test_proof_of_fix_regex_tolerates_extra_whitespace(plugin):
    """The proof block parser should tolerate extra whitespace/indentation."""
    text = """Review pass 2: 0 blockers.

  Proof-of-fix   fix-1:
    test:    test_a
    red-before:   fails
    green-after:  passes
    revert-verified:   yes
"""
    count = plugin._count_verified_proofs(text, [])
    assert count == 1


def test_proof_of_fix_regex_case_insensitive(plugin):
    """revert-verified: YES (uppercase) should count."""
    text = """Proof-of-fix fix-1:
  test: test_a
  red-before: fail
  green-after: pass
  revert-verified: YES
"""
    count = plugin._count_verified_proofs(text, [])
    assert count == 1


def test_proof_of_fix_regex_missing_revert_field_not_counted(plugin):
    """A proof block without revert-verified field → not counted (ambiguous)."""
    text = """Proof-of-fix fix-1:
  test: test_a
  red-before: fail
  green-after: pass
"""
    count = plugin._count_verified_proofs(text, [])
    assert count == 0


def test_proof_of_fix_in_message_history_counted(plugin):
    """Proof blocks in earlier assistant messages are counted, not just
    the response_text."""
    proof = """Proof-of-fix fix-1:
  test: test_a
  red-before: fail
  green-after: pass
  revert-verified: yes
"""
    msgs = [
        {"role": "assistant", "content": f"Review pass 1: 1 blocker. Fixed.\n{proof}"},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    count = plugin._count_verified_proofs("Done.", msgs)
    assert count == 1


# -----------------------------------------------------------------------------
# #8: proof-of-fix gate — dirty pass counting edge cases
# -----------------------------------------------------------------------------

def test_count_dirty_passes_with_only_clean_passes(plugin):
    """All clean passes → 0 dirty passes → proof not required."""
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    assert plugin._count_dirty_passes(msgs, "") == 0


def test_count_dirty_passes_with_multiple_dirty(plugin):
    """Multiple dirty passes → count matches."""
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 2 blockers."},
        {"role": "assistant", "content": "Review pass 2: 1 blocker."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 4: 0 blockers, 0 majors."},
    ]
    assert plugin._count_dirty_passes(msgs, "") == 2


# -----------------------------------------------------------------------------
# #2: step cap — interaction with hard cap
# -----------------------------------------------------------------------------

def test_step_cap_takes_precedence_over_hard_cap(plugin, monkeypatch, state_file):
    """If both MAX_TURNS and MAX_PASSES are configured, the step cap
    (checked first) should take precedence when it trips."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_TURNS", "2")
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "3")
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker."},
        {"role": "assistant", "content": "Review pass 2: 1 blocker."},
        {"role": "assistant", "content": "Review pass 3: 1 blocker."},
    ]
    decision = plugin._pre_exit_verify("Review pass 3: still 1 blocker.", messages=msgs)
    # Step cap (2 tool calls) trips before hard cap (3 passes)
    assert decision["approved"] is True
    assert "step cap" in decision["reason"].lower()
    assert "hard cap" not in decision["reason"].lower()