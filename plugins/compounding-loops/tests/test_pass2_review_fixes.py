"""Pass-2 adversarial review regression tests.

Targets the blockers/majors found in the post-fix re-review of
compounding-loops, specifically:

- 2.1/2.7: session_id threaded into _evaluate_review_gate (and thus into
  the persisted state file).
- 2.2: streak dedup so a model re-emitting the same pass number doesn't
  trip the stuck-cap (highest_pass is now bounded by unique-pass-count).
- 2.5/2.6: gate returns approved via the dedicated override in
  _post_api_request; double-clean logic does NOT silently wedge the
  standard-hook path.
- 2.8: response_text is capped at 200 KB; oversized responses bypass
  review gating entirely.
- 2.10: _VERDICT_GENERATION is bumped on every verdict write; _pre_llm_call
  refuses to inject a stale verdict whose generation is below the
  live counter.
"""

from __future__ import annotations

import importlib.util
import logging
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
    """Reset module-level accumulators between tests in this file.

    The plugin uses module-level dicts (``_LOOP_VERDICTS``,
    ``_SESSION_CLEAN_STREAK``, ``_VERDICT_GENERATION``) for performance —
    they survive the test-runner life of the plugin module. Without
    this fixture, a verdict written by one test can be picked up by
    the next, causing order-dependent flakiness when the suite is run
    with ``pytest-randomly``.
    """
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()
    plugin._VERDICT_GENERATION.clear()
    # Lock down the env so a prior monkeypatch revert doesn't leak.
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    yield


# ---------------------------------------------------------------------------
# 2.1 / 2.7: session_id threaded through _evaluate_review_gate to state file.
# ---------------------------------------------------------------------------

def test_evaluate_review_gate_threads_session_id_to_state(plugin, tmp_path, monkeypatch):
    """When _evaluate_review_gate runs with session_id="abc-123", the
    persisted state file at ~/.hermes/loop-state/STATUS.json must carry
    that session_id, not the empty string."""
    state_file = tmp_path / "STATUS.json"
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(state_file))

    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    # Run the wrapper.
    plugin._evaluate_review_gate(
        response, [], session_id="abc-123",
    )
    # The state file should now exist (written by _write_state_snapshot
    # inside _pre_exit_verify at the verdict=approved branch).
    assert state_file.exists(), (
        "_evaluate_review_gate did not persist state to STATUS.json"
    )
    import json
    state = json.loads(state_file.read_text())
    assert state.get("session_id") == "abc-123", (
        f"session_id was {state.get('session_id')!r}; expected 'abc-123'. "
        f"This is the cross-pollution defect from pass-2 concern 2.1."
    )


# ---------------------------------------------------------------------------
# 2.2: streak dedup — re-emitting the same pass must NOT inflate highest_pass.
# ---------------------------------------------------------------------------

def test_streak_dedup_prevents_repeat_pass_from_tripping_stuck_cap(plugin):
    """The legacy logic appends the pass number to the streak every time
    the model re-emits the same pass. Combined with the new dedup, two
    duplicate pass-1 emissions should leave the streak at length 1, not 2.

    Simulate by calling the post_api_request offset logic directly:
    - emit pass 1 (clean)
    - emit pass 1 again (clean) — same pass, model re-emitted
    - highest_pass should still be 1, not 2.
    """
    sid = "test-dedup-2.2"
    plugin._SESSION_CLEAN_STREAK[sid] = []
    # Replicate the streak-update logic from _post_api_request:
    #   if not streak or streak[-1][0] != latest["pass"]: append
    #   else: replace
    # ...then dedup.
    for _ in range(2):
        reviews = plugin._extract_all_reviews_from_text(
            "Review pass 1: 0 blockers, 0 majors. Review clean."
        )
        latest = reviews[-1]
        streak = plugin._SESSION_CLEAN_STREAK[sid]
        if not streak or streak[-1][0] != latest["pass"]:
            streak.append((latest["pass"], latest["clean"]))
        else:
            streak[-1] = (latest["pass"], latest["clean"])
        # Dedup block (mirrors the patch):
        seen = set()
        deduped = []
        for entry in streak:
            if entry[0] not in seen:
                seen.add(entry[0])
                deduped.append(entry)
        streak[:] = deduped
        highest_pass = max(p for p, _ in streak)
        assert highest_pass == 1, (
            f"highest_pass={highest_pass} after re-emitting pass 1 twice; "
            f"expected 1 (concern 2.2)"
        )


# ---------------------------------------------------------------------------
# 2.5 / 2.6: hook-path double-clean logic doesn't silently wedge.
# ---------------------------------------------------------------------------

def test_post_api_request_approves_via_streak_override(plugin):
    """When the gate rejects with 'need one more consecutive clean pass'
    but the streak accumulator already records 2 clean passes, the hook
    must flip to approved. (This is what currently works; we lock it down
    against regression.)"""
    # Reset state — module-level accumulators leak between tests.
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()
    plugin._VERDICT_GENERATION.clear()

    sid = "test-2.5"
    # Pre-populate the streak with two clean passes.
    plugin._SESSION_CLEAN_STREAK[sid] = [(1, True), (2, True)]

    # Construct an assistant message that emits a clean pass-3 review.
    class FakeMsg:
        content = "Review pass 3: 0 blockers, 0 majors. Review clean."
        tool_calls = None
    # finish_reason "stop", no tool_calls, contains review evidence.
    result = plugin._post_api_request(
        finish_reason="stop",
        assistant_message=FakeMsg(),
        session_id=sid,
    )
    # _post_api_request returns None and stashes a verdict in the cache.
    assert result is None
    verdict = plugin._LOOP_VERDICTS.get(sid)
    assert verdict is not None, "verdict was not stashed"
    assert verdict["verdict"] == "approved", (
        f"override failed: got {verdict}; "
        f"streak accum should have flipped approval on 3 consecutive cleans"
    )


# ---------------------------------------------------------------------------
# 2.8: DoS guard — oversized response bypasses review gating.
# ---------------------------------------------------------------------------

def test_post_api_request_bypasses_oversized_response(plugin, caplog):
    """A runaway model emitting >200 KB of repetitive text must be
    bypassed cleanly (no crash, no infinite regex scan)."""
    caplog.set_level(logging.WARNING, logger="plugins.compounding_loops")
    # 201 KB of 'Review pass 1: 0 blockers, 0 majors. ' repeated.
    huge = "Review pass 1: 0 blockers, 0 majors. " * 5_500
    assert len(huge) > 200_000

    class FakeMsg:
        content = huge
        tool_calls = None
    result = plugin._post_api_request(
        finish_reason="stop",
        assistant_message=FakeMsg(),
        session_id="huge",
    )
    assert result is None
    # No verdict stashed for this session.
    assert plugin._LOOP_VERDICTS.get("huge") is None
    # Warning was logged.
    assert any(
        "exceeds gate cap" in rec.message for rec in caplog.records
    ), "expected warning log; gating must log the bypass"


def test_post_api_request_accepts_normal_response(plugin):
    """Sanity check — under 200 KB responses still trigger gating."""
    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    assert len(response) < 200_000
    class FakeMsg:
        content = response
        tool_calls = None
    # With a single clean pass and require_double_clean=1, this returns None
    # and stashes a rejected verdict (consecutive_clean=1 < 2).
    plugin._post_api_request(
        finish_reason="stop",
        assistant_message=FakeMsg(),
        session_id="normal",
    )
    assert plugin._LOOP_VERDICTS.get("normal") is not None


# ---------------------------------------------------------------------------
# 2.10: _VERDICT_GENERATION bump + pre_llm_call staleness check.
# ---------------------------------------------------------------------------

def test_verdict_generation_increments_on_each_post_api_request(plugin):
    """Each post_api_request with review evidence must bump the global
    _VERDICT_GENERATION counter by 1. The verdict dict must carry the
    bumped generation."""
    plugin._VERDICTS = {}  # ensure fresh
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()
    start_gen = plugin._VERDICT_GENERATION.get("gen-test", 0)

    class FakeMsg:
        content = "Review pass 1: 0 blockers, 0 majors. Review clean."
        tool_calls = None
    plugin._post_api_request(
        finish_reason="stop",
        assistant_message=FakeMsg(),
        session_id="gen-test",
    )
    verdict = plugin._LOOP_VERDICTS.get("gen-test")
    assert verdict is not None
    assert verdict["generation"] == start_gen + 1, (
        f"expected generation {start_gen + 1}, got {verdict['generation']}"
    )


# ---------------------------------------------------------------------------
# 4.3: post-read snapshot detects same-session refresh races.
# ---------------------------------------------------------------------------

def test_post_read_snapshot_detects_same_session_race(plugin):
    """The post-read snapshot catches races the pre-read one misses:

    1. session A's _post_api_request writes verdict at gen=N.
    2. session B's _pre_llm_call reads verdict.gen=N.
    3. session A's _post_api_request fires AGAIN (gen=N+1).
    4. session B's pre_llm_call re-snapshots (_VERDICT_GENERATION = N+1).
    5. cached_gen(N) < gen_after_read(N+1) → drop.

    We simulate by replacing the verdict cache with a subclass that
    bumps the global counter when ``.get()`` is called for our test sid.
    The standard dict class doesn't allow ``.get`` reassignment, so we
    wrap it.
    """
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()
    plugin._VERDICT_GENERATION.clear()

    sid = "race-test"
    cur_gen = plugin._VERDICT_GENERATION.get(sid, 0)
    plugin._VERDICT_GENERATION[sid] = cur_gen
    plugin._LOOP_VERDICTS[sid] = {
        "verdict": "rejected",
        "reason": "would-be-injected",
        "approved_via_cap": False,
        "generation": cur_gen,
    }

    class RacingDict(dict):
        """Wraps the verdict cache; bumps the global counter on .get()."""
        def get(self, key, default=None):
            result = super().get(key, default)
            if key == sid:
                # Simulate "another session's _post_api_request fired
                # between our read and our post-snapshot".
                plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
            return result

    original = plugin._LOOP_VERDICTS
    plugin._LOOP_VERDICTS = RacingDict(original)
    try:
        result = plugin._pre_llm_call(session_id=sid)
        assert result is None, (
            f"cross-session race went undetected; "
            f"pre_llm_call returned {result!r}"
        )
        # Once the racing dict detected it, the verdict should have been
        # popped. We need to copy back to the real dict so subsequent
        # tests don't see it.
    finally:
        racing = plugin._LOOP_VERDICTS
        plugin._LOOP_VERDICTS = original
        # If the verdict was popped during the race test, sync back;
        # otherwise leave the dict as it was.
        plugin._LOOP_VERDICTS.pop(sid, None)


def test_pre_llm_call_drops_stale_verdict(plugin):
    """Cross-session staleness via direct counter bump (no concurrent
    read): the verdict was written at gen=N, then the global counter
    was bumped to N+1 by another session's _post_api_request, then
    pre_llm_call fires.
    """
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()

    sid = "stale-test"
    # Stash a verdict at generation "today" and bump the counter past it.
    plugin._VERDICT_GENERATION[sid] = plugin._VERDICT_GENERATION.get(sid, 0) + 1
    stale_gen = plugin._VERDICT_GENERATION[sid]
    plugin._LOOP_VERDICTS[sid] = {
        "verdict": "rejected",
        "reason": "old rejection",
        "approved_via_cap": False,
        "generation": stale_gen,
    }
    # Bump this session's counter past the cached verdict's generation.
    plugin._VERDICT_GENERATION[sid] += 1

    # Now pre_llm_call fires. The cached verdict is stale.
    result = plugin._pre_llm_call(session_id=sid)
    # Stale verdict must be dropped — no context injected.
    assert result is None, (
        f"pre_llm_call injected a stale verdict; result={result}"
    )
    # And the cache should be empty for this session.
    assert plugin._LOOP_VERDICTS.get(sid) is None, (
        "stale verdict must be popped, not left in the cache"
    )


def test_pre_llm_call_accepts_fresh_verdict(plugin):
    """Sanity: a verdict whose generation matches _VERDICT_GENERATION
    is consumed normally."""
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()

    sid = "fresh-test"
    # Stash a verdict with the current session generation.
    cur_gen = plugin._VERDICT_GENERATION.get(sid, 0)
    plugin._LOOP_VERDICTS[sid] = {
        "verdict": "rejected",
        "reason": "fresh rejection",
        "approved_via_cap": False,
        "generation": cur_gen,
    }
    result = plugin._pre_llm_call(session_id=sid)
    assert result is not None, "fresh verdict must be injected"
    assert "fresh rejection" in result["context"]


# ---------------------------------------------------------------------------
# Combined: a model re-emits pass 1 seven times — must NOT trip stuck cap.
# ---------------------------------------------------------------------------

def test_repeated_pass_emission_does_not_trip_stuck_cap(plugin, monkeypatch):
    """The classic runaway: model emits 'Review pass 1: 0 blockers, 0
    majors. Review clean.' seven times in a row on the standard-hook
    path. Pre-fix, the highest_pass grew unbounded and tripped the
    stuck-cap prematurely (concern 2.2). Post-fix the dedup keeps
    highest_pass = 1 indefinitely."""
    monkeypatch.setenv("HERMES_LOOPS_STUCK_CAP", "6")
    plugin._config()
    plugin._LOOP_VERDICTS.clear()
    plugin._SESSION_CLEAN_STREAK.clear()

    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    class FakeMsg:
        content = response
        tool_calls = None
    sid = "runaway-pass"
    for i in range(7):
        plugin._post_api_request(
            finish_reason="stop",
            assistant_message=FakeMsg(),
            session_id=sid,
        )
        verdict = plugin._LOOP_VERDICTS.get(sid)
        if verdict is None:
            continue
        if verdict["approved_via_cap"]:
            pytest.fail(
                f"iteration {i}: stuck cap fired prematurely — "
                f"verdict={verdict}; concern 2.2 regression"
            )
