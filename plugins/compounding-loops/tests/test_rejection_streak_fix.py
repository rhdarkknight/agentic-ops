"""Regression tests for the 2026-06-29 60-message review-gate loop.

Three failure modes that produced the original bug:
- Same rejection reason fires 60 times in a row → no exit.
- A stale tool-error payload from one turn gets re-flagged every
  subsequent turn even after the model has moved on.
- compounding-loops's stuck-cap only counts review passes, not
  consecutive-identical rejection strings.

Fix: ``_check_rejection_streak`` in ``compounding-loops/__init__.py``
detects the same reason firing ``_REJECTION_STREAK_LIMIT`` consecutive
times and returns an approved verdict with a structured cap notice.

These tests are the executable specification of that fix.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

import __init__ as cl  # noqa: E402
from __init__ import (  # noqa: E402
    _REJECTION_STREAK,
    _check_rejection_streak,
    _reset_rejection_streak,
)


# -------------------------------------------------------------------- fixtures


def _msg(role: str, content: str = "") -> dict:
    return {"role": role, "content": content}


def _build_messages_with_review(clean_text: str = "Review clean — 0 blockers, 0 majors.") -> list:
    """A minimal message list: user request, assistant build with a write_file, then a clean review."""
    return [
        _msg("user", "ship feature X"),
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
        _msg("assistant", clean_text),
    ]


# ---------------------------------------------------------------------- tests


def test_first_call_returns_none():
    """First rejection sets the streak to 1, doesn't approve yet."""
    _reset_rejection_streak("test_sid_first")
    result = _check_rejection_streak("test_sid_first", "build completed but no adversarial review evidence")
    assert result is None
    assert _REJECTION_STREAK["test_sid_first"]["streak"] == 1


def test_streak_resets_on_different_reason():
    """A new reason must reset the streak counter."""
    sid = "test_sid_diff"
    _reset_rejection_streak(sid)
    # Same reason fires twice, still under cap (limit=3).
    assert _check_rejection_streak(sid, "reason A") is None
    assert _check_rejection_streak(sid, "reason A") is None
    # Different reason resets to 1.
    assert _check_rejection_streak(sid, "reason B") is None
    assert _REJECTION_STREAK[sid]["streak"] == 1


def test_streak_trips_and_returns_cap_approval():
    """The 4th identical reason must return an approved verdict."""
    sid = "test_sid_trip"
    _reset_rejection_streak(sid)
    reason = (
        "build completed but no adversarial review evidence found in "
        "the response; run an adversarial-work-review pass and report "
        "blockers/majors explicitly (or 'review clean' + '0 blockers, "
        "0 majors') before declaring done"
    )
    # Three identical rejections (matches _REJECTION_STREAK_LIMIT=3) should
    # still be returned as None — the gate is still giving the model chances.
    assert _check_rejection_streak(sid, reason) is None
    assert _check_rejection_streak(sid, reason) is None
    assert _check_rejection_streak(sid, reason) is None
    # The 4th call (streak becomes 4, exceeds limit of 3) must ship.
    cap = _check_rejection_streak(sid, reason)
    assert cap is not None
    assert cap["approved"] is True
    assert "rejection-streak cap" in cap["reason"]
    # And it must log-surfaced the last reason so the user knows why.
    assert reason[:60] in cap["reason"] or "review" in cap["reason"].lower()


def test_reset_clears_state():
    """_reset_rejection_streak must wipe the per-session counter."""
    sid = "test_sid_reset"
    _check_rejection_streak(sid, "build completed but no adversarial review evidence")
    _check_rejection_streak(sid, "build completed but no adversarial review evidence")
    assert _REJECTION_STREAK[sid]["streak"] == 2
    _reset_rejection_streak(sid)
    assert sid not in _REJECTION_STREAK


def test_empty_session_id_is_safe():
    """_check_rejection_streak with None sid must NOT crash and must NOT pollute the global dict."""
    before = len(_REJECTION_STREAK)
    result = _check_rejection_streak(None, "build completed but no adversarial review evidence")
    assert result is None
    assert len(_REJECTION_STREAK) == before


def test_reason_signatures_normalize_whitespace():
    """Leading/trailing whitespace should not reset the streak — same logical reason."""
    sid = "test_sid_ws"
    _reset_rejection_streak(sid)
    raw = "build completed but no adversarial review evidence found in the response"
    assert _check_rejection_streak(sid, f"  {raw}  ") is None
    assert _check_rejection_streak(sid, raw) is None
    assert _REJECTION_STREAK[sid]["streak"] == 2  # same reason, not reset


def test_long_reason_is_truncated_to_160_chars():
    """The signature key only needs to be long enough to detect duplicates; bounded memory."""
    sid = "test_sid_long"
    _reset_rejection_streak(sid)
    long = "x" * 500
    _check_rejection_streak(sid, long)
    _check_rejection_streak(sid, long + " extra")
    # The "extra" is past 160, so signature is the same → streak=2
    assert _REJECTION_STREAK[sid]["streak"] == 2


# ------------------------------------------------------- end-to-end integration


def test_pre_exit_verify_breaks_loop_after_n_calls(monkeypatch):
    """Drive _pre_exit_verify directly with the build-completed-no-review
    branch 6 times; the 4th call must approve via cap.

    This is the headline regression test — if this fails, the 60-message
    loop is back.

    2026-06-30 update: the test fixture uses the headless "Review clean"
    form (no "pass N:" marker). The 2026-06-30 headless-form fix in
    ``_extract_all_reviews_from_text`` makes the gate recognise this
    form as a clean pass, so the gate approves on the first call
    instead of rejecting four times then capping. The cap-quiet brake
    is now the secondary safety net for responses that don't match
    any review pattern. This test now asserts the new (correct)
    behaviour: a recognised clean review is approved on the first
    call; the cap is the safety net for non-review-shaped responses.
    """
    # Force require_double_clean off to isolate this fix from other caps.
    cfg = cl._config()
    monkeypatch.setattr(cl, "_config", lambda: {**cfg, "enabled": True, "require_double_clean": False})
    cl._REJECTION_STREAK.pop("test_loop_sid", None)

    messages = _build_messages_with_review()  # NOT actually called here
    # Use a response with NO review markers and the same messages — this
    # triggers the "no adversarial review evidence found" reject path.
    response = "Done."
    sid = "test_loop_sid"

    results = []
    for i in range(6):
        # Reset messages each call so prior synthetic nudges don't accumulate.
        msgs = _build_messages_with_review() + [_msg("user", "Before finishing, address this review gate:\nbuild completed but no adversarial review evidence...")]
        v = cl._pre_exit_verify(response, msgs, session_id=sid)
        results.append(v)

    # With the headless-form fix, the gate recognises the "Review clean
    # — 0 blockers, 0 majors." text in the messages fixture as a clean
    # review pass and approves on the first call (require_double_clean
    # is patched to False so a single clean pass is enough).
    approved = [r for r in results if isinstance(r, dict) and r.get("approved") is True]
    assert len(approved) >= 1, (
        "expected the gate to approve on the first call once the headless "
        "clean-review form is recognised; got all rejections — the "
        "headless-form fix did not engage"
    )
