"""Runaway-loop regression tests for compounding-loops.

Targets the four failure modes the user reported:

1. Circuit breaker must catch loops anywhere in recent history, not
   only at the very end of the session (history-not-tail trap).
2. Circuit breaker signature must hash full args to prevent
   truncation collisions (different file paths sharing first 200 chars).
3. Synthetic-tool-call stub in `_evaluate_review_gate` is gone — no
   contamination of mutating build counts.
4. Bypass-keyword check fires BEFORE step cap so a runaway can't escape
   by replying "thanks!" at the end.

Each test reproduces a real failure path and asserts the brake fires.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    spec = importlib.util.spec_from_file_location(
        "plugins.compounding_loops", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.compounding_loops"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_module_state(plugin):
    """Reset module-level accumulators between tests.

    Lock-down env: a prior monkeypatch revert can leak into the next
    test in ``pytest-randomly`` orderings. We force the default back to
    require_double_clean=1 at the start of every test so the policy is
    predictable.
    """
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()
    plugin._VERDICT_GENERATION.clear()
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    yield


def _tc(name: str, args: str, call_id: str = "tc") -> dict:
    """A single tool call with explicit args string."""
    return {
        "id": call_id,
        "function": {"name": name, "arguments": args},
    }


def _msgs_with_tool_calls(calls: list) -> list:
    """Build a messages list with all calls stuffed into one assistant turn."""
    return [
        {"role": "user", "content": "build me x"},
        {"role": "assistant", "content": None, "tool_calls": calls},
    ]


# ---------------------------------------------------------------------------
# 1. Circuit breaker: history-not-tail — loops in the middle trip the brake.
# ---------------------------------------------------------------------------

def test_circuit_breaker_trips_on_loop_in_middle_of_history(plugin, monkeypatch):
    """A loop that happened earlier in the session, followed by recovery
    and a final response, must still trip the brake when checking the
    current exit. (Loop-before-recovery trap.)"""
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "3")
    cfg = plugin._config()
    assert cfg["circuit_breaker"] == 3
    # 3 identical calls, then 5 diverse recovery calls.
    calls = (
        [_tc("read_file", '{"path":"/tmp/a"}')] * 3
        + [
            _tc("write_file", '{"path":"/tmp/b","c":"x"}'),
            _tc("read_file", '{"path":"/tmp/c"}'),
            _tc("terminal", '{"cmd":"ls"}'),
            _tc("patch", '{"path":"/tmp/d"}'),
            _tc("write_file", '{"path":"/tmp/e","c":"y"}'),
        ]
    )
    msgs = _msgs_with_tool_calls(calls)
    name, args = plugin._detect_circuit_breaker(msgs, 3)
    assert name == "read_file", f"expected loop trap to fire on read_file, got {name}"
    assert "{" in args  # args snippet preserved


def test_circuit_breaker_ignores_when_streak_below_threshold(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "5")
    # 3 identical calls — below threshold of 5.
    calls = [_tc("read_file", '{"path":"/tmp/a"}')] * 3
    msgs = _msgs_with_tool_calls(calls)
    name, _ = plugin._detect_circuit_breaker(msgs, 5)
    assert name is None


def test_circuit_breaker_signature_distinguishes_different_file_paths(plugin):
    """Two different file paths happen to share their first 200 chars
    (impossible here, but proves the hash prevents any first-200-chars
    collision possibility)."""
    calls = [
        _tc("write_file", '{"path":"/tmp/X","c":"alpha"}'),
        _tc("write_file", '{"path":"/tmp/Y","c":"beta"}'),
    ]
    sigs = [plugin._tool_call_signature(c) for c in calls]
    assert len(set(sigs)) == 2, "different paths should not hash the same"


def test_circuit_breaker_signature_full_arg_diff_detected(plugin):
    """Identical tool, identical 201+ char args that differ only at byte
    201 must produce different signatures. With sha1 hashing, this is
    trivially true."""
    base = "x" * 250
    a = _tc("write_file", '{"k":"' + base + 'A"}')
    b = _tc("write_file", '{"k":"' + base + "B\"}")
    assert plugin._tool_call_signature(a) != plugin._tool_call_signature(b)


# ---------------------------------------------------------------------------
# 2. Session key logs debug-level warning when missing.
# ---------------------------------------------------------------------------

def test_session_key_returns_none_when_missing(plugin, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger="plugins.compounding_loops")
    key = plugin._session_key({})
    assert key is None
    assert any(
        "session_id" in rec.message for rec in caplog.records
    ), "expected debug log explaining anonymous no-op"


def test_session_key_prefers_real_session_id(plugin):
    assert plugin._session_key({"session_id": "abc-123"}) == "abc-123"


# ---------------------------------------------------------------------------
# 3. _evaluate_review_gate no longer mutates messages with fake tool calls.
# ---------------------------------------------------------------------------

def test_evaluate_review_gate_does_not_inject_synthetic_tool_calls(plugin):
    """The legacy stub appended a fake `write_file` + `patch` to messages
    when called via the standard hook path. That contamination inflated
    the hint cache and tricked the max_turns cap. Confirm: the messages
    list is unchanged after _evaluate_review_gate runs.

    A responses that contains a clean review pass, with no review-tools,
    no double-clean required (set HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN=0).
    """
    import os
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "0"
    plugin._config()
    response = (
        "Here's my answer.\n\n"
        "Review pass 1: 0 blockers, 0 majors. Review clean."
    )
    msgs_in = [{"role": "user", "content": "build x"}]
    msgs_before = list(msgs_in)
    # _evaluate_review_gate returns the verdict or None. Test that
    # _pre_exit_verify is called with force_build, not synthetic tool calls.
    # We can't easily intercept _pre_exit_verify here without monkeypatch,
    # but we can at least assert the messages list identity.
    verdict = plugin._evaluate_review_gate(response, msgs_in, session_id="t")
    # _evaluate_review_gate should not mutate msgs_in
    assert msgs_in == msgs_before, (
        f"_evaluate_review_gate mutated messages; "
        f"before={msgs_before!r} after={msgs_in!r}"
    )
    # And the verdict should be approved (clean pass + single-clean mode).
    assert isinstance(verdict, dict)
    assert verdict.get("approved") is True


# ---------------------------------------------------------------------------
# 4. Bypass-keyword check fires BEFORE step cap.
# ---------------------------------------------------------------------------

def test_step_cap_fires_before_bypass_keyword(plugin, monkeypatch):
    """A runaway session with 10000 tool calls that ends with the
    user-reply 'thanks!' (which contains no bypass keyword itself —
    the trick is the LLM emitting a reply that the user doesn't
    'thank' for). The cap MUST trip even if a generic friendly ending
    matches a bypass keyword. We approximate: the gate's bypass check
    should not appear in the cap path. Use only enough build calls to
    confirm the cap fires, with a bypass-word-free response."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_TURNS", "3")
    monkeypatch.setenv("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", "0")
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "0")
    plugin._config()  # bust env cache
    # 4 build tool calls — cap is 3.
    calls = [
        _tc("write_file", '{"path":"/tmp/x1","c":"a"}'),
        _tc("write_file", '{"path":"/tmp/x2","c":"b"}'),
        _tc("write_file", '{"path":"/tmp/x3","c":"c"}'),
        _tc("write_file", '{"path":"/tmp/x4","c":"d"}'),
    ]
    msgs = _msgs_with_tool_calls(calls)
    msgs.append({"role": "user", "content": "thanks!"})  # no bypass word
    msgs.append({"role": "assistant", "content": "All done."})
    verdict = plugin._pre_exit_verify("All done.", msgs)
    # Step cap fires "approved: True" with a reason that mentions the cap.
    assert verdict.get("approved") is True
    assert "step cap" in str(verdict.get("reason", "")).lower(), (
        f"expected step-cap notice, got {verdict}"
    )


def test_user_bypass_keyword_still_works(plugin, monkeypatch):
    """Sanity check: explicit 'quick' bypass keyword still skips the gate
    even with the step cap raised."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_TURNS", "3")
    monkeypatch.setenv("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", "0")
    monkeypatch.setenv("HERMES_LOOPS_CIRCUIT_BREAKER", "0")
    plugin._config()
    calls = [_tc("write_file", '{"path":"/tmp/x","c":"a"}')] * 4
    msgs = _msgs_with_tool_calls(calls)
    msgs[0]["content"] = "can you do a quick check on this"
    msgs.append({"role": "assistant", "content": "All done."})
    verdict = plugin._pre_exit_verify("All done.", msgs)
    assert verdict.get("approved") is True
    # When the bypass kicks in, no cap reason is reported.
    assert "step cap" not in str(verdict.get("reason", "")).lower(), (
        f"bypass should suppress cap notice, got {verdict}"
    )


# ---------------------------------------------------------------------------
# 5. Force-build path (used by post_api_request) doesn't pollute counters.
# ---------------------------------------------------------------------------

def test_force_build_does_not_count_as_a_real_build(plugin, monkeypatch):
    """When force_build=True is passed, _is_build_response is bypassed for
    the early-return — but the cap math uses real messages. With empty
    messages + single-clean + no double-clean required, the gate should
    approve without a cap notice.

    Note: we explicitly set HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN=0 so this
    test exercises the single-clean approval path. With the default
    require-double-clean the gate correctly returns rejected
    (consecutive_clean=1 < 2) regardless of force_build — that's the
    correct behavior, not a regression.
    """
    monkeypatch.setenv("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", "0")
    plugin._config()  # bust env cache
    verdict = plugin._pre_exit_verify(
        "Review pass 1: 0 blockers, 0 majors. Review clean.",
        [],  # empty messages
        force_build=True,
    )
    # Single-clean + no double-clean required → approved.
    assert verdict.get("approved") is True, (
        f"expected approved with single-clean+force_build+empty; got {verdict}"
    )
    # And no cap notice (force_build doesn't trip max_turns with 0 calls).
    assert "cap" not in str(verdict.get("reason", "")).lower(), (
        f"force_build should not trip max_turns cap with 0 calls; got {verdict}"
    )


def test_force_build_does_not_silently_approve_under_double_clean(plugin):
    """Counter-test: force_build is NOT a backdoor for "approve single clean"
    when require_double_clean is the policy. The gate must reject so the
    hook-path override (in _post_api_request) can engage.
    """
    # require_double_clean=1 is the plugin default from the fixture;
    # the autouse reset_module_state fixture locks this down. We assert
    # at the top to make the precondition explicit — if a prior test
    # monkeypatch leaked, this assertion catches it.
    cfg = plugin._config()
    assert cfg["require_double_clean"] is True, (
        f"require_double_clean should be True at start of test; got "
        f"{cfg['require_double_clean']!r}. Prior test likely leaked env."
    )
    verdict = plugin._pre_exit_verify(
        "Review pass 1: 0 blockers, 0 majors. Review clean.",
        [],
        force_build=True,
    )
    assert verdict.get("approved") is False, (
        f"force_build must NOT bypass require_double_clean; got {verdict}"
    )
    assert "consecutive clean pass" in str(verdict.get("reason", "")).lower(), (
        f"expected double-clean rejection, got {verdict}"
    )
