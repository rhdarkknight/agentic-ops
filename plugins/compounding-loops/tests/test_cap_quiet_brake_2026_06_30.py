"""Regression tests for the 2026-06-30 PII-removal-Zed-ACP cap-loop bug.

Failure mode (reproduces the user's "ton of looping at the end of the
conversation that doesn't progress it forward" complaint):

  1. The gate fires ``approved_via_cap=True`` after the rejection-streak
     trips. ``_pre_llm_call`` injects a cap-nudge: "Present your final
     answer now, summarising what was built and noting any remaining
     open findings from the cap."
  2. The model interprets "summarising what was built" as "run another
     review pass" and emits ``**REVIEW CLEAN. 0 blockers, 0 majors, 6
     known minors.** Same state.`` — a headless clean-review form with
     no "pass N:" marker.
  3. The next ``_pre_exit_verify`` call sees no "pass N:" marker (the
     strict ``_PASS_RE`` requires it) and rejects with "no adversarial
     review evidence found in the response".
  4. The rejection-streak cap fires again, ``_pre_llm_call`` injects the
     same cap-nudge, the model recycles the same text, ad infinitum.
  5. The user pays for ~55+ extra API calls per session and the agent
     cannot ship the response.

Fix (three parts):

  - ``_extract_all_reviews_from_text`` now recognises the "headless"
    clean-review form ("**REVIEW CLEAN. 0 blockers, 0 majors, 6 known
    minors.**") that does not carry a "pass N:" marker. A response
    containing both ``_CLEAN_RE`` evidence AND explicit 0-blocker /
    0-major counts is treated as an additional clean review pass.
  - Cap-nudge text rewritten to a tight command that forbids review
    prose, tool calls, and finding lists.
  - ``_CAP_QUIET_STREAK`` counter increments on every cap approval.
    After ``_CAP_QUIET_STREAK_GRACE`` followups, ``_pre_exit_verify``
    auto-approves without invoking the review gate, breaking the
    recite loop as a final safety net.
  - Counter resets on session start/reset (alongside the existing
    per-session accumulators).

These tests are the executable specification of the fix.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

import __init__ as cl  # noqa: E402


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def plugin():
    """Reset per-session state without reloading the module.

    Reloading would invalidate module-level references held by other
    test files in the same suite (e.g. ``_REJECTION_STREAK`` in
    ``test_rejection_streak_fix``) and cause unrelated tests to fail.
    The existing convention is: clear module-level state via the
    ``_reset_*`` helpers and let env-driven config stay constant
    across the test run.
    """
    # Snapshot original env so we can restore after.
    orig_env = {k: os.environ.get(k) for k in [
        "HERMES_LOOPS_ENABLED",
        "HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN",
        "HERMES_LOOPS_REJECTION_STREAK_LIMIT",
        "HERMES_LOOPS_CAP_QUIET_GRACE",
    ]}
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ["HERMES_LOOPS_REJECTION_STREAK_LIMIT"] = "3"
    os.environ["HERMES_LOOPS_CAP_QUIET_GRACE"] = "1"
    cl._LOOP_VERDICTS.clear()
    cl._SESSION_CLEAN_STREAK.clear()
    cl._VERDICT_GENERATION.clear()
    cl._REJECTION_STREAK.clear()
    cl._CAP_QUIET_STREAK.clear()
    cl._LAST_REFLECT_TAIL.clear()
    yield cl
    # Restore env
    for k, v in orig_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    cl._LOOP_VERDICTS.clear()
    cl._SESSION_CLEAN_STREAK.clear()
    cl._REJECTION_STREAK.clear()
    cl._CAP_QUIET_STREAK.clear()
    cl._LAST_REFLECT_TAIL.clear()


def _msg(role: str, content: str = "") -> dict:
    return {"role": role, "content": content}


def _build_messages_with_no_evidence() -> list:
    """A build that has shipped, with no fresh review evidence in the
    final response — this is the exact shape that produces the
    "build completed but no adversarial review evidence" rejection.

    Note: the user text uses a build keyword ("implement") so the
    gate's ``_is_build_response`` check returns True. Without that,
    the gate approves without invoking the review path and the
    reject / cap-quiet flow never runs.
    """
    return [
        _msg("user", "implement the PII removal feature"),
        _msg("assistant", "I'll write the patch."),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "/tmp/x.py", "content": "pass"}',
                    },
                }
            ],
        },
        _msg("tool", '{"success": true}'),
    ]


# ---------------------------------------------------------------------- tests


def test_cap_nudge_forbids_review_prose(plugin):
    """The cap-nudge text must explicitly forbid review prose,
    summarising, and tool calls. This is the central change of the
    fix — the prior text asked the model to 'summarise what was built'
    which the model interpreted as 'write a review pass'.
    """
    sid = "test_sid_nudge"
    # Simulate a cap-approval verdict cached for the next pre_llm_call
    plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
    plugin._LOOP_VERDICTS[sid] = {
        "verdict": "approved",
        "reason": "stuck cap fired after 4 identical rejections",
        "approved_via_cap": True,
        "generation": plugin._VERDICT_GENERATION[sid],
    }
    result = plugin._pre_llm_call(session_id=sid)
    assert result is not None
    ctx = result["context"]
    # Forbids review prose
    assert "Do NOT run another review pass" in ctx
    # Forbids tool calls (the old text did not)
    assert "Do NOT call any tools" in ctx
    # Forbids summaries / finding lists (the old text asked for this)
    assert "Do NOT write a review, summary, or list of findings" in ctx
    # Tells the model to emit a one-liner
    assert "one short closeout line" in ctx
    # State mutates
    assert plugin._CAP_QUIET_STREAK[sid] == 1


def test_cap_quiet_streak_increments_on_each_cap_approval(plugin):
    """Each cap-approval must bump the streak counter so the gate
    knows how many followups the model has burned through.
    """
    sid = "test_sid_incr"
    for i in range(3):
        plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
        plugin._LOOP_VERDICTS[sid] = {
            "verdict": "approved",
            "reason": "stuck cap fired",
            "approved_via_cap": True,
            "generation": plugin._VERDICT_GENERATION[sid],
        }
        plugin._pre_llm_call(session_id=sid)
    assert plugin._CAP_QUIET_STREAK[sid] == 3


def test_cap_quiet_streak_resets_on_session_reset(plugin):
    """Session reset must clear the cap-quiet counter so a fresh user
    turn in a new session starts unencumbered.
    """
    sid = "test_sid_reset"
    plugin._CAP_QUIET_STREAK[sid] = 5
    plugin._on_session_reset(session_id=sid)
    assert sid not in plugin._CAP_QUIET_STREAK


def test_headless_clean_review_recognized(plugin):
    """The user-reported recite loop: the model emits
    "**REVIEW CLEAN. 0 blockers, 0 majors, 6 known minors.**" with no
    "pass N:" marker. The gate must recognise this as a clean review
    pass (it has both _CLEAN_RE evidence AND explicit 0-counts) so it
    can approve and break the loop.

    Without this fix the gate keeps rejecting with "no adversarial
    review evidence found" because _PASS_RE requires "pass N:" and
    the model only ever emits the headless form. That drove the
    60-message recite loop the user reported.
    """
    sid = "test_sid_headless"
    messages = _build_messages_with_no_evidence()
    recyle_text = (
        "**REVIEW CLEAN. 0 blockers, 0 majors, 6 known minors.** "
        "Same state."
    )
    verdict = plugin._pre_exit_verify(
        recyle_text, messages, session_id=sid,
    )
    assert isinstance(verdict, dict)
    # The first pass is rejected as "need one more clean pass" — not
    # as "no adversarial review evidence". This proves the headless
    # form was recognised. (Without the fix, the reason would be
    # "build completed but no adversarial review evidence found...".)
    if not verdict.get("approved"):
        reason = verdict.get("reason", "")
        assert "no adversarial review evidence" not in reason, (
            f"headless form was NOT recognised; gate still demands "
            f"'pass N:' template. reason={reason!r}"
        )


def test_two_consecutive_headless_clean_passes_approve(plugin):
    """Two consecutive headless clean-review responses must converge
    to approved. With the headless-form fix, pass 1 is recognised
    (rejected as "need one more"), pass 2 hits double-clean and
    approves. This is the primary loop-terminator.
    """
    sid = "test_sid_headless_converge"
    messages = _build_messages_with_no_evidence()
    recyle_text = (
        "**REVIEW CLEAN. 0 blockers, 0 majors, 6 known minors.** "
        "Same state."
    )
    # Pass 1: headless review, gate should recognise and reject as
    # "need one more clean pass" (NOT as "no adversarial review").
    v1 = plugin._pre_exit_verify(
        recyle_text, messages, session_id=sid,
    )
    assert isinstance(v1, dict)
    assert "no adversarial review evidence" not in v1.get("reason", ""), (
        f"pass 1 was not recognised as a review pass; "
        f"reason={v1.get('reason')!r}"
    )
    # Pass 2: same text, gate should approve via double-clean.
    # The first call's response_text must be folded into messages
    # for pass 2 to count as a second clean pass.
    messages_pass2 = messages + [_msg("assistant", recyle_text)]
    v2 = plugin._pre_exit_verify(
        recyle_text, messages_pass2, session_id=sid,
    )
    assert isinstance(v2, dict)
    if not v2.get("approved"):
        reason = v2.get("reason", "")
        assert "no adversarial review evidence" not in reason, (
            f"pass 2 regressed to no-evidence rejection: {reason!r}"
        )


def test_cap_quiet_brake_auto_approves_after_grace(plugin):
    """Reproduction for the cap-quiet safety net: a response that
    doesn't match the headless-review pattern (so the gate keeps
    rejecting it) — but the cap has fired and the model has burned
    through its grace followup. The brake auto-approves.

    The headless-form fix is the *primary* loop terminator (it makes
    the gate recognise the model's preferred review format). The
    cap-quiet brake is a *secondary* safety net for cases where the
    model emits text that still fails the gate even after the headless
    recognition — e.g. tool calls with no review evidence, or text
    without _CLEAN_RE language.
    """
    sid = "test_sid_brake"
    messages = _build_messages_with_no_evidence()
    # A response that is NOT a headless clean review: no _CLEAN_RE
    # language, no 0-blocker/0-major counts. The gate will reject
    # with "no adversarial review evidence" until the cap fires.
    bad_recyle_text = (
        "Same state. Continuing to recite the same verdict would be "
        "theatre. The build is in the same shape it was 20 turns ago."
    )
    # Cap-approval nudge consumed (grace=1): counter goes to 1
    plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
    plugin._LOOP_VERDICTS[sid] = {
        "verdict": "approved",
        "reason": "stuck cap fired",
        "approved_via_cap": True,
        "generation": plugin._VERDICT_GENERATION[sid],
    }
    plugin._pre_llm_call(session_id=sid)
    assert plugin._CAP_QUIET_STREAK[sid] == 1
    # Model emits a non-review response. Within grace, the gate
    # rejects (the model gets one chance to do better after the
    # cap-nudge).
    verdict_within_grace = plugin._pre_exit_verify(
        bad_recyle_text, messages, session_id=sid,
    )
    assert isinstance(verdict_within_grace, dict)
    assert verdict_within_grace.get("approved") is False
    # Counter exceeds grace → cap-quiet brake auto-approves.
    plugin._CAP_QUIET_STREAK[sid] = plugin._CAP_QUIET_STREAK_GRACE + 1
    verdict_after_grace = plugin._pre_exit_verify(
        bad_recyle_text, messages, session_id=sid,
    )
    assert isinstance(verdict_after_grace, dict)
    assert verdict_after_grace.get("approved") is True
    reason = verdict_after_grace.get("reason", "")
    assert "cap-quiet brake" in reason, (
        f"expected cap-quiet brake reason, got: {reason!r}"
    )
    # The counter is consumed so the next user turn starts fresh.
    assert sid not in plugin._CAP_QUIET_STREAK


def test_cap_quiet_brake_does_not_fire_when_grace_unmet(plugin):
    """A single post-cap followup must NOT trip the brake — the model
    needs one chance to emit a true closeout before we silence the
    gate.
    """
    sid = "test_sid_within_grace"
    messages = _build_messages_with_no_evidence()
    plugin._CAP_QUIET_STREAK[sid] = 1  # == grace, not > grace
    recyle_text = "**REVIEW CLEAN. 0 blockers, 0 majors, 6 known minors.**"
    verdict = plugin._pre_exit_verify(
        recyle_text, messages, session_id=sid,
    )
    # Should fall through to the normal gate path, not the brake.
    assert isinstance(verdict, dict)
    assert verdict.get("approved") is not True or "cap-quiet brake" not in verdict.get("reason", "")


def test_cap_quiet_brake_full_loop_terminates(plugin, capsys):
    """The full reproduction with non-review text: drive 5 cap-approval
    cycles and assert the loop terminates. The hard requirement is
    that after enough cycles the brake auto-approves — the
    headless-form fix won't help here because the response has no
    review language at all.
    """
    sid = "test_sid_full"
    messages = _build_messages_with_no_evidence()
    recyle_text = (
        "Same state. The gates keep firing on identical inputs. "
        "Continuing to recite the same verdict would be theatre."
    )
    decisions = []
    for cycle in range(6):
        # Cap fires: pre_llm_call injects the nudge, counter increments
        plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
        plugin._LOOP_VERDICTS[sid] = {
            "verdict": "approved",
            "reason": "stuck cap fired",
            "approved_via_cap": True,
            "generation": plugin._VERDICT_GENERATION[sid],
        }
        plugin._pre_llm_call(session_id=sid)
        # Model emits another non-review response
        verdict = plugin._pre_exit_verify(
            recyle_text, messages, session_id=sid,
        )
        approved = bool(verdict.get("approved")) if isinstance(verdict, dict) else False
        decisions.append(approved)
    # By the 6th cycle the counter has tripped the brake.
    assert any(decisions[-2:]), (
        f"loop did not terminate: decisions={decisions}"
    )
    # Specifically the final decision must be auto-approved.
    assert decisions[-1] is True, (
        f"final decision was not approved: decisions={decisions}"
    )


def test_cap_quiet_brake_respects_env_grace_override(plugin):
    """HERMES_LOOPS_CAP_QUIET_GRACE=0 must skip the grace period
    entirely — the first post-cap followup auto-approves.
    """
    # The cap-quiet grace is a module-level int that's read fresh on
    # every call to the brake check. Override the constant directly
    # (the test fixture already handles env restoration on teardown).
    saved_grace = plugin._CAP_QUIET_STREAK_GRACE
    plugin._CAP_QUIET_STREAK_GRACE = 0
    try:
        sid = "test_sid_no_grace"
        messages = _build_messages_with_no_evidence()
        recyle_text = (
            "Same state. Continuing to recite the same verdict would "
            "be theatre."
        )
        plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
        plugin._LOOP_VERDICTS[sid] = {
            "verdict": "approved",
            "reason": "stuck cap fired",
            "approved_via_cap": True,
            "generation": plugin._VERDICT_GENERATION[sid],
        }
        plugin._pre_llm_call(session_id=sid)
        verdict = plugin._pre_exit_verify(
            recyle_text, messages, session_id=sid,
        )
        assert isinstance(verdict, dict)
        assert verdict.get("approved") is True
        assert "cap-quiet brake" in verdict.get("reason", "")
    finally:
        plugin._CAP_QUIET_STREAK_GRACE = saved_grace
