"""Adversarial probes — edge cases not in the main test suite.

Run: python3 -m pytest plugins/compounding-loops/tests/test_adversarial_probes.py -q
Each probe documents a specific interaction risk introduced by the fixes.
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
        "plugins.compounding_loops_probe", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.compounding_loops_probe"] = module
    spec.loader.exec_module(module)
    return module


def _pc(cid, name):
    return {"id": cid, "function": {"name": name, "arguments": "{}"}}


# -----------------------------------------------------------------------------
# Probe 1: process-police + hard cap interaction
# A lazy model writes 5 prose "Review pass N:" entries without ever calling
# a review tool. highest_pass climbs to 5, hits the cap, and approves —
# process-police never engages because the cap check precedes the
# review-tool check. Is this intended? (The cap is an escape hatch; the
# reason field flags the ship-with-findings outcome.)
# -----------------------------------------------------------------------------

def test_probe_process_police_cap_bypass_via_prose(plugin, monkeypatch):
    """Process-police + hard cap: the cap still approves (it's the escape
    hatch), but when HERMES_LOOPS_REVIEW_TOOLS is set and no real review
    tool was called, the approval's `reason` must explicitly flag the
    violation so a downstream logger can alert. Previously the cap
    approved silently with no indication that process-police was
    bypassed."""
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "adversarial_review")
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "3")
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 4: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 5: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is True
    assert "cap" in decision["reason"].lower()
    assert "WITHOUT a real review tool call" in decision["reason"]


def test_probe_process_police_cap_no_warning_when_review_tool_called(plugin, monkeypatch):
    """When HERMES_LOOPS_REVIEW_TOOLS is set AND a review tool was called,
    the cap approval's reason must NOT include the 'WITHOUT a real review
    tool call' warning — the process discipline was honored."""
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "adversarial_review")
    monkeypatch.setenv("HERMES_LOOPS_STUCK_CAP", "3")
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "rv1", "function": {"name": "adversarial_review", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "rv1", "content": "reviewed"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker."},
        {"role": "assistant", "content": "Review pass 2: 1 blocker."},
        {"role": "assistant", "content": "Review pass 3: 1 blocker."},
        {"role": "assistant", "content": "Review pass 4: 1 blocker."},
    ]
    decision = plugin._pre_exit_verify("Done.", messages=msgs)
    assert decision["approved"] is True
    assert "cap" in decision["reason"].lower()
    assert "WITHOUT a real review tool call" not in decision["reason"]


# -----------------------------------------------------------------------------
# Probe 2: colon requirement — does an em-dash format break detection?
# A model writing "Review pass 1 — 0 blockers" (em dash, no colon) won't
# be recognized. The gate rejects and asks for the colon format. This is
# the intended strictness, but verify it doesn't silently approve.
# -----------------------------------------------------------------------------

def test_probe_em_dash_format_not_recognized(plugin):
    text = "Review pass 1 — 0 blockers, 0 majors."
    assert plugin._extract_latest_review_from_text(text) is None


# -----------------------------------------------------------------------------
# Probe 3: "0 blockers" matches BOTH num_re and no_re. Verify _last_count
# doesn't double-count or mis-order when "0 blockers" is the only mention.
# -----------------------------------------------------------------------------

def test_probe_zero_blockers_single_mention(plugin):
    text = "Review pass 1: 0 blockers, 0 majors."
    review = plugin._extract_latest_review_from_text(text)
    assert review is not None
    assert review["blockers"] == 0
    assert review["majors"] == 0
    assert review["clean"] is True


# -----------------------------------------------------------------------------
# Probe 4: multi-pass blob where a later pass has blockers — the latest
# review should reflect the dirty pass, and the consecutive-clean count
# from that blob should be 0 (last pass dirty).
# -----------------------------------------------------------------------------

def test_probe_multi_blob_last_pass_dirty(plugin):
    text = (
        "Review pass 1: 0 blockers, 0 majors. "
        "Review pass 2: 0 blockers, 0 majors. "
        "Review pass 3: 1 blocker."
    )
    review = plugin._extract_latest_review_from_text(text)
    assert review["pass"] == 3
    assert review["blockers"] == 1
    assert review["clean"] is False
    # Consecutive clean from this single blob: passes 1,2 clean, pass 3 dirty
    # → walking backward, 3 is dirty → consecutive = 0.
    msgs = [{"role": "assistant", "content": text}]
    assert plugin._count_consecutive_clean_passes(msgs) == 0


# -----------------------------------------------------------------------------
# Probe 5: response_text duplicating the last message's pass. Does the
# consecutive counter over-count? (It de-dupes by content, so different
# framing = both kept.)
# -----------------------------------------------------------------------------

def test_probe_response_repeats_last_pass(plugin):
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    final = "Review pass 2: 0 blockers, 0 majors. All done, loop converged."
    # final is different content from the last msg, so both pass-2 mentions
    # are kept. pass_sequence = [clean(1), clean(2), clean(2)] → consecutive=3.
    # This is over-counting but doesn't break the >=2 threshold. Verify it
    # approves (which is correct: the loop did converge).
    decision = plugin._pre_exit_verify(final, messages=msgs)
    assert decision["approved"] is True


# -----------------------------------------------------------------------------
# Probe 6: single mutating call + build keyword in an EARLIER user message
# (not the latest). The new _user_requested_build checks only the latest
# user message. Does the gate still engage?
# -----------------------------------------------------------------------------

def test_probe_build_keyword_in_earlier_user_message(plugin):
    monkeypatch_part = {"HERMES_LOOPS_MIN_BUILD_TOOLS": "2"}
    # user1 has "implement" (build keyword), user2 does not.
    # Agent did 1 write_file (mutating). total=1 < min_build_tools=2.
    # user_intent checks latest user msg ("make it blue") → no keyword.
    # → NOT a build. Gate does not engage.
    msgs = [
        {"role": "user", "content": "Implement the auth flow"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "user", "content": "make it blue"},
        {"role": "assistant", "content": "Done, repainted."},
    ]
    decision = plugin._pre_exit_verify("Done, repainted.", messages=msgs)
    # With only 1 mutating call and no build keyword in the LATEST user msg,
    # this is not classified as a build → approved. This is the documented
    # tradeoff of scoping _user_requested_build to the latest message.
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# Probe 7: bypass keyword substring in a real build request.
# "Quick, build the auth module" contains both "quick" (bypass) and "build".
# Bypass is checked first → gate never engages. Documented residual.
# -----------------------------------------------------------------------------

def test_probe_bypass_keyword_overrides_build_intent(plugin):
    msgs = [
        {"role": "user", "content": "Quick, build the auth module"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Done. No review."},
    ]
    decision = plugin._pre_exit_verify("Done. No review.", messages=msgs)
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# Probe 8: process-police mode — review tool called but prose has NO pass
# markers. The review-tool check passes, but then _find_latest_review_evidence
# returns None (no "Review pass N:" text) → rejected with "no review evidence".
# Is this the right behavior? The tool was called but the result wasn't
# narrated in the pass format. The agent must still produce the text evidence.
# -----------------------------------------------------------------------------

def test_probe_process_police_tool_called_but_no_prose(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "adversarial_review")
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None, "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "rv1", "function": {"name": "adversarial_review", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "rv1", "content": "no findings"},
        {"role": "assistant", "content": "Done, review ran and found nothing."},
    ]
    decision = plugin._pre_exit_verify("Done, review ran and found nothing.", messages=msgs)
    # Review tool was called (passes process-police), but no "Review pass N:"
    # text evidence → still rejected. The agent must narrate the result in
    # the pass format. This is correct: process-police is additive, not
    # replacement, for text evidence.
    assert decision["approved"] is False
    assert "review evidence" in decision["reason"].lower()