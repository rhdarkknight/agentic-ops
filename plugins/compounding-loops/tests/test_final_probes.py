"""Final adversarial probes — consecutive-clean ordering edge cases."""

from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ["HERMES_LOOPS_MAX_PASSES"] = "9"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    spec = importlib.util.spec_from_file_location(
        "plugins.cl_final", _PLUGIN_DIR / "__init__.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["plugins.cl_final"] = m
    spec.loader.exec_module(m)
    return m


def _pc(cid, name):
    return {"id": cid, "function": {"name": name, "arguments": "{}"}}


def _build(user, **kw):
    msgs = [{"role": "user", "content": user}]
    if kw.get("tool_calls"):
        msgs.append({"role": "assistant", "content": None, "tool_calls": kw["tool_calls"]})
    if kw.get("assistant_text"):
        msgs.append({"role": "assistant", "content": kw["assistant_text"]})
    return msgs


def test_response_dirty_after_clean_messages_resets_count(plugin):
    """response_text with a dirty pass after clean passes in messages →
    consecutive count must be 0 (the newest review found issues)."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    final = "On reflection, review pass 3: 1 blocker. Need to fix."
    assert plugin._count_consecutive_clean_passes(msgs, final) == 0
    decision = plugin._pre_exit_verify(final, messages=msgs)
    assert decision["approved"] is False
    assert "blocker" in decision["reason"].lower()


def test_response_already_in_messages_not_double_counted(plugin):
    """If the dispatcher already appended response_text to messages,
    the de-dup keeps it in its chronological position and does not
    re-add it at the end."""
    final = "Review pass 2: 0 blockers, 0 majors. Done."
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": final},  # already appended
    ]
    # response_text == final is in seen → not re-added. Sequence:
    # [clean(1), clean(2)] → consecutive = 2 → approve.
    assert plugin._count_consecutive_clean_passes(msgs, final) == 2
    assert plugin._pre_exit_verify(final, messages=msgs) == {"approved": True}


def test_response_with_pass_after_msg_without_pass_extends_streak(plugin):
    """messages have pass 1 clean; response_text has pass 2 clean →
    consecutive = 2 → approve."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
    ]
    final = "Review pass 2: 0 blockers, 0 majors. Converged."
    assert plugin._count_consecutive_clean_passes(msgs, final) == 2
    assert plugin._pre_exit_verify(final, messages=msgs) == {"approved": True}


def test_highest_pass_includes_response_text_via_messages_only(plugin):
    """_highest_pass_seen only scans messages, not response_text. If the
    only pass marker is in response_text, highest_pass from messages is
    0. This is fine because _find_latest_review_evidence checks
    response_text first; but verify the gate still engages correctly
    when the cap is configured tightly."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
    ]
    final = "Review pass 1: 0 blockers, 0 majors. Review clean."
    # highest_pass from messages == 0 (no pass in messages). The cap
    # check (0 >= max_passes) is False → gate proceeds to review check.
    # review found in response_text → clean → consecutive check.
    # With double-clean on, consecutive == 1 (only pass 1 in response)
    # → rejected, asks for one more clean pass. Correct behavior.
    decision = plugin._pre_exit_verify(final, messages=msgs)
    assert decision["approved"] is False
    assert "consecutive" in decision["reason"].lower() or "1/2" in decision["reason"]


def test_non_mutating_then_mutating_engages_gate(plugin):
    """A read-only terminal followed by a write_file should engage the
    gate (one mutating call is enough; total build tools = 2)."""
    msgs = _build(
        "Wire up the endpoint",
        tool_calls=[_pc("t1", "terminal"), _pc("w1", "write_file")],
        assistant_text="Done.",
    )
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is False


def test_single_write_below_threshold_with_no_keyword_not_a_build(plugin):
    """One write_file (mutating) but total=1 < min_build_tools=2 and no
    build keyword → not a build → approve."""
    msgs = _build(
        "rename this variable",
        tool_calls=[_pc("w1", "write_file")],
        assistant_text="Done.",
    )
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# Context-compaction fallback (session_hints) — P3 risk A
# -----------------------------------------------------------------------------

def test_hints_recover_build_after_compaction_erased_tool_calls(plugin):
    """Compaction scenario: the build tool calls have been summarized out
    of the message content, so _is_build_response would return False and
    the gate would silently approve. With session_hints remembering the
    build_tool_count, the gate must still engage and demand re-evidence."""
    # First, prime the hints with a real build (2 mutating calls).
    prime_msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
    ]
    hints = {}
    plugin._pre_exit_verify("Review pass 1: 0 blockers, 0 majors.", messages=prime_msgs, session_hints=hints)
    assert hints["mutating_build_count"] >= 2
    assert hints["has_review"] is True

    # Now compaction rewrites the history: tool calls gone, review prose
    # gone, only the user message and a final "Done" remain.
    compacted_msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": "Done."},  # summary, no tool calls, no review
    ]
    decision = plugin._pre_exit_verify("Done.", messages=compacted_msgs, session_hints=hints)
    # Without hints this would be approved (not a build). With hints it
    # must engage and demand re-evidence.
    assert decision["approved"] is False
    assert "review" in decision["reason"].lower() or "compaction" in decision["reason"].lower()


def test_hints_recover_review_evidence_after_compaction(plugin):
    """Compaction scenario: build tool calls are still in messages, but
    the review pass markers were summarized out. Without hints the gate
    would reject with "no review evidence". With hints remembering a
    review happened, it must reject with the compaction-specific reason
    (demanding re-evidence, not silently approving)."""
    prime_msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
    ]
    hints = {}
    plugin._pre_exit_verify("Review pass 1: 0 blockers, 0 majors.", messages=prime_msgs, session_hints=hints)
    assert hints["has_review"] is True

    # Compaction: build calls remain (so is_build=True), review prose gone.
    compacted_msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Done."},  # no review markers
    ]
    decision = plugin._pre_exit_verify("Done.", messages=compacted_msgs, session_hints=hints)
    assert decision["approved"] is False
    assert "compaction" in decision["reason"].lower()


def test_hints_highest_pass_survives_compaction(plugin, monkeypatch):
    """Compaction erased the message with the highest pass marker.
    Without hints, highest_pass would under-count and the hard cap
    wouldn't trip when it should. With hints, the cap fires correctly."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "3")
    prime_msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker."},
        {"role": "assistant", "content": "Review pass 2: 1 blocker."},
        {"role": "assistant", "content": "Review pass 3: 1 blocker."},
    ]
    hints = {}
    plugin._pre_exit_verify("Review pass 3: 1 blocker.", messages=prime_msgs, session_hints=hints)
    assert hints["highest_review_pass"] == 3
    import os; os.environ["HERMES_LOOPS_STUCK_CAP"] = "3"

    # Compaction drops the pass-2 and pass-3 messages.
    compacted = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker."},
        {"role": "assistant", "content": "Done."},
    ]
    decision = plugin._pre_exit_verify("Done.", messages=compacted, session_hints=hints)
    # highest_pass from messages alone is 1, but hints remember 3 → cap trips.
    assert decision["approved"] is True
    assert "cap" in decision["reason"].lower()
    assert "3" in decision["reason"] and "limit 3" in decision["reason"]


def test_hints_never_silently_approve_without_prose(plugin):
    """Hints must NEVER be sufficient alone for an approval — only as a
    signal to demand re-evidence. A build with hints but no current prose
    evidence must still be rejected, not approved."""
    hints = {"mutating_build_count": 5, "has_review": True, "highest_review_pass": 2}
    # Messages have NO build tool calls and NO review evidence — only the
    # hint cache claims a build/review happened.
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": "Done."},
    ]
    decision = plugin._pre_exit_verify("Done.", messages=msgs, session_hints=hints)
    assert decision["approved"] is False


def test_hints_monotonic_no_regression_on_compaction(plugin):
    """Hints must climb monotonically — a compaction that drops messages
    must never cause a hint to regress to a lower value."""
    hints = {"mutating_build_count": 5, "highest_review_pass": 4, "has_review": True}
    # A message list with fewer build calls and no reviews (post-compaction).
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file")]},
    ]
    plugin._update_hints(msgs, "Done.", hints)
    assert hints["mutating_build_count"] == 5  # not regressed to 1
    assert hints["highest_review_pass"] == 4  # not regressed
    assert hints["has_review"] is True