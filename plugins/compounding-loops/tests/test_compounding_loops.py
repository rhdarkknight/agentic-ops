"""Tests for the compounding-loops plugin decisions.

Mirrors the test harness-conductor pattern: import the plugin module
directly, exercise the decision functions, assert the return shape.

Tests are written to run from anywhere — they resolve the plugin dir
from their own location so the suite works both inside the agent tree
and inside the standalone plugin dir (~/.hermes/plugins/compounding-loops).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


# Resolve the plugin directory relative to this test file. Tests live at
#   <plugin_dir>/tests/test_compounding_loops.py
# so parent.parent is the plugin dir. This works regardless of whether
# the plugin is shipped standalone under ~/.hermes/plugins/ or copied
# into hermes-agent/plugins/.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def plugin():
    """Load the plugin module fresh; force the gated env vars to on."""
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    # Process-police mode is opt-in; default off for the historical
    # text-evidence behaviour.
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    spec = importlib.util.spec_from_file_location(
        "plugins.compounding_loops", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.compounding_loops"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def disable_double_clean(monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", "0")


@pytest.fixture()
def disable_plugin(monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_ENABLED", "0")


@pytest.fixture()
def process_police(monkeypatch):
    """Opt into HERMES_LOOPS_REVIEW_TOOLS so the gate requires a real
    review tool call, not just prose."""
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "adversarial_review,review")


# Convenience aliases
_pre_exit_verify = lambda plugin, *a, **k: plugin._pre_exit_verify(*a, **k)
_post_tool_batch_reflect = lambda plugin, *a, **k: plugin._post_tool_batch_reflect(*a, **k)
_extract_latest_review = lambda plugin, *a, **k: plugin._extract_latest_review_from_text(*a, **k)
_extract_all_reviews = lambda plugin, *a, **k: plugin._extract_all_reviews_from_text(*a, **k)
_count_consecutive_clean_passes = lambda plugin, *a, **k: plugin._count_consecutive_clean_passes(*a, **k)
_highest_pass_seen = lambda plugin, *a, **k: plugin._highest_pass_seen(*a, **k)


def _build_messages(
    user: str,
    assistant_text: str = "",
    tool_calls: list | None = None,
    tool_results: list | None = None,
) -> list:
    msgs = [{"role": "user", "content": user}]
    if tool_calls:
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
    if tool_results:
        for tc, result in zip(tool_calls or [], tool_results):
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.get("id", "tc"),
                "content": result,
            })
    if assistant_text:
        msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


def _patch_call(call_id: str, name: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


# -----------------------------------------------------------------------------
# pre_exit_verify — base cases
# -----------------------------------------------------------------------------

def test_pre_exit_approve_when_disabled(plugin, disable_plugin):
    decision = _pre_exit_verify(
        plugin,
        "Done.",
        messages=_build_messages(
            "Build me a CLI tool",
            assistant_text="Done. Wrote the file.",
            tool_calls=[_patch_call("tc1", "write_file")],
        ),
    )
    assert decision == {"approved": True}


def test_pre_exit_approve_empty_response(plugin):
    """Empty response is left to the harness-conductor plugin."""
    decision = _pre_exit_verify(plugin, "", messages=[])
    assert decision == {"approved": True}


def test_pre_exit_approve_non_build(plugin):
    decision = _pre_exit_verify(
        plugin,
        "Sure — the answer is 42.",
        messages=_build_messages("What's 6 times 7?"),
    )
    assert decision == {"approved": True}


def test_pre_exit_approve_bypass_keyword(plugin):
    decision = _pre_exit_verify(
        plugin,
        "Done.",
        messages=_build_messages(
            "Quick — rename this variable",
            tool_calls=[_patch_call("tc1", "write_file")],
        ),
    )
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# pre_exit_verify — build detection
# -----------------------------------------------------------------------------

def test_pre_exit_terminal_only_does_not_count_as_build(plugin):
    """Two read-only terminal calls without any write_file/patch must NOT
    engage the gate — they don't constitute a build."""
    msgs = _build_messages(
        "Show me what's in the repo",
        assistant_text="Here's what I found.",
        tool_calls=[_patch_call("tc1", "terminal"),
                    _patch_call("tc2", "terminal")],
    )
    decision = _pre_exit_verify(plugin, msgs[-1]["content"], messages=msgs)
    assert decision == {"approved": True}


def test_pre_exit_execute_code_only_does_not_count_as_build(plugin):
    """execute_code without any mutating call is not a build."""
    msgs = _build_messages(
        "Summarize the data",
        assistant_text="Done.",
        tool_calls=[_patch_call("tc1", "execute_code"),
                    _patch_call("tc2", "execute_code")],
    )
    decision = _pre_exit_verify(plugin, msgs[-1]["content"], messages=msgs)
    assert decision == {"approved": True}


def test_pre_exit_mixed_terminal_and_write_counts_as_build(plugin):
    """A mutating call + a terminal call should engage the gate."""
    msgs = _build_messages(
        "Wire up the new endpoint and check it",
        assistant_text="Done.",
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "terminal")],
    )
    decision = _pre_exit_verify(plugin, msgs[-1]["content"], messages=msgs)
    assert decision["approved"] is False


# -----------------------------------------------------------------------------
# pre_exit_verify — review enforcement
# -----------------------------------------------------------------------------

def test_pre_exit_rejects_build_without_review(plugin):
    msgs = _build_messages(
        "Implement the new auth flow",
        assistant_text="Done. The auth module is wired up.",
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, msgs[-1]["content"], messages=msgs)
    assert decision["approved"] is False
    assert "review" in decision["reason"].lower()
    assert decision["max_reverify_passes"] >= 1


def test_pre_exit_rejects_when_blockers_found(plugin):
    response = (
        "Review pass 1: 3 blockers, 1 major. "
        "Blockers: missing CSRF, broken auth redirect, password leak."
    )
    msgs = _build_messages(
        "Implement the new auth flow",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    assert decision["approved"] is False
    assert "blocker" in decision["reason"].lower()
    assert "3" in decision["reason"]


def test_pre_exit_rejects_when_majors_found(plugin):
    response = (
        "Review pass 2: 0 blockers, 2 majors. "
        "Majors: missing tests, race condition in cache invalidation."
    )
    msgs = _build_messages(
        "Implement cache invalidation",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    assert decision["approved"] is False
    assert "major" in decision["reason"].lower()


def test_pre_exit_rejects_when_only_one_clean_pass(plugin):
    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    msgs = _build_messages(
        "Implement X",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    assert decision["approved"] is False
    assert "consecutive" in decision["reason"].lower() or "1/2" in decision["reason"]


def test_pre_exit_approves_after_two_clean_passes(plugin):
    """Two consecutive clean passes across messages → loop complete."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc1", "write_file"),
                        _patch_call("tc2", "patch")]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        {"role": "assistant",
         "content": "Review pass 1: 0 blockers, 0 majors. Review clean."},
        {"role": "assistant",
         "content": "Review pass 2: 0 blockers, 0 majors. Review clean."},
    ]
    final = "Done. Loop converged on two consecutive clean passes."
    msgs.append({"role": "assistant", "content": final})
    decision = _pre_exit_verify(plugin, final, messages=msgs)
    assert decision == {"approved": True}


def test_pre_exit_approves_single_clean_pass_when_double_clean_disabled(
    plugin, disable_double_clean,
):
    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    msgs = _build_messages(
        "Implement X",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    assert decision == {"approved": True}


def test_pre_exit_approves_at_hard_cap(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_STUCK_CAP", "3")
    response = "Review pass 5: 2 blockers still open, will iterate."
    msgs = _build_messages(
        "Implement X",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    # Pass 5 is at/over the cap (3) — let it exit, but with a reason.
    assert decision["approved"] is True
    assert "reason" in decision
    assert "cap" in decision["reason"].lower()


# -----------------------------------------------------------------------------
# pre_exit_verify — process-police mode (HERMES_LOOPS_REVIEW_TOOLS)
# -----------------------------------------------------------------------------

def test_pre_exit_process_police_rejects_prose_only_review(plugin, process_police):
    """With review tools configured, prose "Review pass 1: clean" without
    a review tool call is rejected."""
    response = "Review pass 1: 0 blockers, 0 majors. Review clean."
    msgs = _build_messages(
        "Implement X",
        assistant_text=response,
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _pre_exit_verify(plugin, response, messages=msgs)
    assert decision["approved"] is False
    assert "review tool" in decision["reason"].lower()


def test_pre_exit_process_police_accepts_when_review_tool_called(plugin, process_police):
    """A real review tool call plus clean evidence → may converge."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc1", "write_file"),
                        _patch_call("tc2", "patch")]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "rv1",
                         "function": {"name": "adversarial_review",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "rv1", "content": "clean"},
        {"role": "assistant",
         "content": "Review pass 1: 0 blockers, 0 majors. Review clean."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "rv2",
                         "function": {"name": "review",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "rv2", "content": "clean"},
        {"role": "assistant",
         "content": "Review pass 2: 0 blockers, 0 majors. Review clean."},
    ]
    final = "Done. Loop converged."
    msgs.append({"role": "assistant", "content": final})
    decision = _pre_exit_verify(plugin, final, messages=msgs)
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# post_tool_batch_reflect
# -----------------------------------------------------------------------------

def test_reflect_approves_when_disabled(plugin, disable_plugin, request):
    msgs = _build_messages(
        "Implement X",
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision == {"reflect": False}


def test_reflect_approves_non_build(plugin, request):
    msgs = _build_messages("What's the weather?")
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision == {"reflect": False}


def test_reflect_approves_bypass(plugin, request):
    msgs = _build_messages(
        "Trivial — change this string",
        tool_calls=[_patch_call("tc1", "write_file")],
    )
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision == {"reflect": False}


def test_reflect_triggers_on_build_without_review(plugin, request):
    msgs = _build_messages(
        "Implement the new auth flow",
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision["reflect"] is True
    assert "review" in decision["reason"].lower()


def test_reflect_silent_when_review_started(plugin, request):
    msgs = _build_messages(
        "Implement X",
        tool_calls=[_patch_call("tc1", "write_file"),
                    _patch_call("tc2", "patch")],
    )
    msgs.append({
        "role": "assistant",
        "content": "Review pass 1: 0 blockers, 0 majors. Review clean.",
    })
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision == {"reflect": False}


def test_reflect_triggers_when_build_after_review(plugin, request):
    """Mutating build after the last review should nudge a new pass."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc1", "write_file")]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "assistant",
         "content": "Review pass 1: 1 blocker. Fixing."},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc2", "patch")]},
        {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
    ]
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision["reflect"] is True
    assert "review pass 2" in decision["reason"].lower()


def test_reflect_silent_when_terminal_after_review(plugin, request):
    """Read-only terminal after a review should NOT trigger a nudge —
    only mutating calls warrant re-review."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc1", "write_file")]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "assistant",
         "content": "Review pass 1: 0 blockers, 0 majors. Review clean."},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc2", "terminal")]},
        {"role": "tool", "tool_call_id": "tc2", "content": "ls output"},
    ]
    decision = _post_tool_batch_reflect(
        plugin, messages=msgs, session_id=f"test_{request.node.name}"
    )
    assert decision == {"reflect": False}


# -----------------------------------------------------------------------------
# Response parsing helpers
# -----------------------------------------------------------------------------

def test_extract_latest_review_no_evidence(plugin):
    assert _extract_latest_review(plugin, "All done.") is None


def test_extract_latest_review_first_pass(plugin):
    text = "Review pass 1: 3 blockers, 2 majors, 1 minor."
    review = _extract_latest_review(plugin, text)
    assert review == {"pass": 1, "blockers": 3, "majors": 2, "clean": False}


def test_extract_latest_review_clean_pass(plugin):
    text = "Review pass 2: 0 blockers, 0 majors. Review clean."
    review = _extract_latest_review(plugin, text)
    assert review == {"pass": 2, "blockers": 0, "majors": 0, "clean": True}


def test_extract_latest_review_uses_last_mention(plugin):
    """When multiple passes are mentioned, parse the latest one."""
    text = (
        "Review pass 1: 3 blockers. Fixed. "
        "Review pass 2: 0 blockers, 0 majors. Review clean."
    )
    review = _extract_latest_review(plugin, text)
    assert review["pass"] == 2
    assert review["clean"] is True


def test_extract_latest_review_no_blockers_phrase(plugin):
    text = "Pass 3: no blockers, 1 major, 0 minors."
    review = _extract_latest_review(plugin, text)
    assert review["blockers"] == 0
    assert review["majors"] == 1
    assert review["clean"] is False


def test_extract_latest_review_with_hash_prefix(plugin):
    text = "Review pass #4: 0 blockers, 0 majors. Pass complete."
    review = _extract_latest_review(plugin, text)
    assert review["pass"] == 4
    assert review["clean"] is True


def test_extract_latest_review_clean_without_magic_phrase(plugin):
    """P1d: '0 blockers, 0 majors' alone (no 'review clean' phrase) must
    count as clean — previously it required an explicit clean phrase."""
    text = "Review pass 2: 0 blockers, 0 majors."
    review = _extract_latest_review(plugin, text)
    assert review is not None
    assert review["blockers"] == 0
    assert review["majors"] == 0
    assert review["clean"] is True


# -----------------------------------------------------------------------------
# _PASS_RE — tightened to require a colon (P1b)
# -----------------------------------------------------------------------------

def test_pass_re_rejects_non_review_prose(plugin):
    """A bare 'we did 3 passes over the data' must NOT be parsed as
    review evidence — previously it could inflate highest_pass and trip
    the hard cap."""
    text = "We did 3 passes over the dataset to clean it up."
    assert _extract_latest_review(plugin, text) is None


def test_pass_re_rejects_passphrase_and_compiler_pass(plugin):
    """False-positive guards: 'pass 5 of the compiler' and 'passphrase 2'
    must not be parsed as review pass 5 / pass 2."""
    assert _extract_latest_review(plugin, "pass 5 of the compiler") is None
    assert _extract_latest_review(plugin, "the passphrase 2 factor") is None


def test_highest_pass_ignores_non_review_prose(plugin):
    msgs = [
        {"role": "user", "content": "Analyze the data"},
        {"role": "assistant", "content": "We did 5 passes over the data."},
    ]
    assert _highest_pass_seen(plugin, msgs) == 0


def test_highest_pass_seen(plugin):
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 0 blockers."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers."},
    ]
    assert _highest_pass_seen(plugin, msgs) == 3


def test_highest_pass_seen_no_reviews(plugin):
    msgs = [{"role": "assistant", "content": "All done."}]
    assert _highest_pass_seen(plugin, msgs) == 0


# -----------------------------------------------------------------------------
# _last_count — latest-mention-wins (P1c)
# -----------------------------------------------------------------------------

def test_last_count_takes_latest_blocker_mention(plugin):
    """'found 0 blockers, then 3 blockers' should resolve to 3 — the
    latest correction, not the first 'no blockers' phrase."""
    text = "Review pass 1: 0 blockers, then found 3 blockers on closer look."
    review = _extract_latest_review(plugin, text)
    assert review is not None
    assert review["blockers"] == 3
    assert review["clean"] is False


def test_last_count_takes_latest_no_blocker_mention(plugin):
    """'3 blockers' followed by '0 blockers' (correction) → 0."""
    text = "Review pass 1: 3 blockers reported earlier, but actually 0 blockers after re-check."
    review = _extract_latest_review(plugin, text)
    assert review is not None
    assert review["blockers"] == 0
    assert review["clean"] is True


# -----------------------------------------------------------------------------
# _extract_all_reviews_from_text — per-pass parsing (P2a)
# -----------------------------------------------------------------------------

def test_extract_all_reviews_multi_pass_in_one_blob(plugin):
    """Three pass mentions in one assistant message yield three reviews,
    not one. The consecutive-clean walker depends on this."""
    text = (
        "Review pass 1: 2 blockers. Fixed. "
        "Review pass 2: 0 blockers, 0 majors. "
        "Review pass 3: 0 blockers, 0 majors."
    )
    reviews = _extract_all_reviews(plugin, text)
    assert [r["pass"] for r in reviews] == [1, 2, 3]
    assert reviews[0]["clean"] is False
    assert reviews[1]["clean"] is True
    assert reviews[2]["clean"] is True


def test_extract_all_reviews_empty(plugin):
    assert _extract_all_reviews(plugin, "Nothing here.") == []


# -----------------------------------------------------------------------------
# _count_consecutive_clean_passes — per-pass-mention (P2a)
# -----------------------------------------------------------------------------

def test_count_consecutive_clean_passes_empty(plugin):
    assert _count_consecutive_clean_passes(plugin, []) == 0


def test_count_consecutive_clean_passes_two(plugin):
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors. Review clean."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors. Review clean."},
    ]
    assert _count_consecutive_clean_passes(plugin, msgs) == 2


def test_count_consecutive_clean_passes_breaks_on_dirty(plugin):
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors. Review clean."},
        {"role": "assistant", "content": "Review pass 2: 2 blockers. Fixing."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors. Review clean."},
    ]
    assert _count_consecutive_clean_passes(plugin, msgs) == 1


def test_count_consecutive_clean_passes_multi_pass_in_one_msg(plugin):
    """P2a: two clean passes mentioned in a single assistant message
    must contribute two clean entries to the sequence, not one."""
    msgs = [
        {"role": "assistant", "content": (
            "Review pass 1: 2 blockers. Fixed. "
            "Review pass 2: 0 blockers, 0 majors. "
            "Review pass 3: 0 blockers, 0 majors."
        )},
    ]
    assert _count_consecutive_clean_passes(plugin, msgs) == 2


def test_count_consecutive_clean_passes_includes_response_text(plugin):
    """The in-flight final response counts as the latest entry."""
    msgs = [
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
    ]
    final = "Review pass 2: 0 blockers, 0 majors."
    assert _count_consecutive_clean_passes(plugin, msgs, final) == 2


# -----------------------------------------------------------------------------
# _config — safe env parsing (P3)
# -----------------------------------------------------------------------------

def test_config_safe_int_fallback_on_garbage(plugin, monkeypatch):
    """Garbage env values must fall back to defaults, not raise."""
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "abc")
    monkeypatch.setenv("HERMES_LOOPS_MIN_BUILD_TOOLS", "not-a-number")
    cfg = plugin._config()
    assert cfg["max_passes"] == 3
    assert cfg["min_build_tools"] == 2


def test_config_clamps_max_passes(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "999")
    assert plugin._config()["max_passes"] == 9
    monkeypatch.setenv("HERMES_LOOPS_MAX_PASSES", "0")
    assert plugin._config()["max_passes"] == 1


def test_config_review_tools_parses_csv(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "review, audit , adversarial_review")
    rt = plugin._config()["review_tools"]
    assert rt == ("review", "audit", "adversarial_review")


def test_config_review_tools_empty_default(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_REVIEW_TOOLS", "")
    assert plugin._config()["review_tools"] == ()


# -----------------------------------------------------------------------------
# register — hard-fail option (P3)
# -----------------------------------------------------------------------------

def test_register_hard_fail_raises(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_HARD_FAIL", "1")

    class BadCtx:
        pass

    with pytest.raises(RuntimeError, match="register_hook"):
        plugin.register(BadCtx())


def test_register_soft_fail_warns(plugin, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_LOOPS_HARD_FAIL", "0")

    class BadCtx:
        pass

    # Should not raise.
    plugin.register(BadCtx())


def test_register_success_wires_hooks(plugin, monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_HARD_FAIL", "1")

    class GoodCtx:
        def __init__(self):
            self.hooks = []

        def register_hook(self, name, fn):
            self.hooks.append((name, fn))

    ctx = GoodCtx()
    plugin.register(ctx)
    names = [h[0] for h in ctx.hooks]
    assert "pre_exit_verify" in names
    assert "post_tool_batch_reflect" in names

# -----------------------------------------------------------------------------
# Regression: review delegated to a subagent (delegate_task) must satisfy the
# gate. The subagent's report returns as a `role == "tool"` message; before the
# fix the gate only scanned `role == "assistant"` messages, so it never saw the
# review evidence and force-redispatched forever ("the task ends inside an
# unclosed code block" / delegation loop).
# -----------------------------------------------------------------------------

def _delegated_review_messages(user, review_text):
    """Build a message trace where the review was produced by a subagent.

    Shape mirrors how delegate_task returns: an assistant tool_call to
    delegate_task, followed by a `role == "tool"` result carrying the
    subagent's review report.
    """
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": None,
         "tool_calls": [
             _patch_call("tc1", "write_file"),
             _patch_call("tc2", "patch"),
             {"id": "del1", "function": {"name": "delegate_task",
                                         "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        # Subagent review report -- this is what the gate must now see.
        {"role": "tool", "tool_call_id": "del1", "content": review_text},
        {"role": "assistant",
         "content": "Review delegated to subagent. Summary: " + review_text},
    ]


def test_delegated_review_single_clean_pass_approved(plugin):
    """A single delegated clean review (0 blockers, 0 majors) should be
    accepted (double-clean disabled here so one clean pass converges)."""
    import os as _os
    review = "Review pass 1: 0 blockers, 0 majors. Review clean."
    msgs = _delegated_review_messages("Implement the login flow", review)
    _os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "0"
    try:
        decision = _pre_exit_verify(plugin, "Done. " + review, messages=msgs)
    finally:
        _os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    assert decision["approved"] is True, decision
    assert "no adversarial review evidence" not in decision.get("reason", "").lower()


def test_delegated_review_pass_marker_only_in_tool_msg(plugin):
    """If the pass marker lives ONLY in the `role == 'tool'` subagent
    report (not the assistant summary), the gate must still see it."""
    import os as _os
    review = "Review pass 1: 0 blockers, 0 majors. Review clean."
    msgs = _delegated_review_messages("Implement the login flow", review)
    msgs[-1] = {"role": "assistant", "content": "Delegated review complete."}
    _os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "0"
    try:
        decision = _pre_exit_verify(plugin, "Delegated review complete.", messages=msgs)
    finally:
        _os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    assert decision["approved"] is True, decision
    assert "no adversarial review evidence" not in decision.get("reason", "").lower()


def test_raw_terminal_tool_output_does_not_spoof_review(plugin):
    """Tool results NOT produced by a delegation/review tool must NOT be
    treated as review evidence -- shell output with '0 blockers' must not
    bypass the gate."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_patch_call("tc1", "write_file"),
                        {"id": "term1", "function": {"name": "terminal",
                                                     "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "tool", "tool_call_id": "term1",
         "content": "scan done: 0 blockers found, all good"},
        {"role": "assistant", "content": "Implemented X. No review yet."},
    ]
    decision = _pre_exit_verify(plugin, "Implemented X. No review yet.", messages=msgs)
    assert decision["approved"] is False
    assert "no adversarial review evidence" in decision.get("reason", "").lower()


def test_delegated_review_double_clean_converges(plugin):
    """Two consecutive delegated clean reviews converge under double-clean."""
    r1 = "Review pass 1: 0 blockers, 0 majors. Review clean."
    r2 = "Review pass 2: 0 blockers, 0 majors. Review clean."
    msgs = [
        {"role": "user", "content": "Implement the cache"},
        {"role": "assistant", "content": None,
         "tool_calls": [
             _patch_call("tc1", "write_file"),
             _patch_call("tc2", "patch"),
             {"id": "d1", "function": {"name": "delegate_task",
                                       "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        {"role": "tool", "tool_call_id": "d1", "content": r1},
        {"role": "assistant", "content": "Pass 1 delegated: " + r1},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "d2", "function": {"name": "delegate_task",
                                                 "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "d2", "content": r2},
        {"role": "assistant", "content": "Pass 2 delegated: " + r2},
    ]
    decision = _pre_exit_verify(plugin, "Done. " + r1 + " " + r2, messages=msgs)
    assert decision["approved"] is True, decision
    assert "no adversarial review evidence" not in decision.get("reason", "").lower()
