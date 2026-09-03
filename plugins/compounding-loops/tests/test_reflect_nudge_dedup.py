"""Regression tests for the reflect-nudge dedup fix (2026-06-30).

User symptom: Zed ACP session ran 40+ API calls in a row, all `out=20`,
~3.8s apart, until manual interrupt. acp-diag.log showed
"Pre-API-call reflect drain: injected into tool msg at index N" firing
on every single turn. Root cause: _post_tool_batch_reflect returned
reflect=True on every batch as long as _is_build_response(messages) was
true (i.e. once a build had ever happened this session), so the drain
re-injected the nudge → model emitted empty tool call →
post_tool_batch_reflect fired again → loop.

Fix: track the tail of mutating-build tool-call signatures per session
at the moment of the last reflect=True; if the same tail appears again
(no new build activity), return False. Signatures (not counts) so the
dedup survives context compaction erasing old tool messages.
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
    yield module
    os.environ.pop("HERMES_LOOPS_ENABLED", None)
    os.environ.pop("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", None)


def _mutating_assistant_msg(name: str = "write_file", args: str = '{"path": "x"}') -> dict:
    """Build an assistant message with a single mutating tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def _build_msgs(num_mutating_calls: int) -> list:
    """Build a session with N mutating build tool calls (>= MIN_BUILD_TOOLS=3)."""
    msgs = [
        {"role": "user", "content": "build the thing"},
        _mutating_assistant_msg(),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    for i in range(num_mutating_calls):
        msgs.append(_mutating_assistant_msg(args=f'{{"path": "x{i}"}}'))
        msgs.append({"role": "tool", "tool_call_id": f"call_{i + 2}", "content": "ok"})
    return msgs


def test_first_reflect_after_build_returns_true(plugin):
    """The first call after a build batch lands must still nudge."""
    session_id = "dedup_test_session_first"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(4)
    result = plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    assert result == {"reflect": True, "reason": result["reason"], "max_reflect_passes": 1}, (
        f"first reflect after build must return reflect=True; got {result!r}"
    )
    assert "review pass 1" in result["reason"]


def test_second_reflect_without_new_build_returns_false(plugin):
    """The exact symptom: same messages, same session — second call must stay silent."""
    session_id = "dedup_test_session_second"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(4)
    first = plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    assert first["reflect"] is True, "first call must still nudge (sanity)"
    for i in range(9):
        result = plugin._post_tool_batch_reflect(msgs, session_id=session_id)
        assert result == {"reflect": False}, (
            f"call {i + 2}/10 must be silent; got {result!r} "
            f"(this is the runaway-loop bug)"
        )


def test_reflect_resumes_after_new_build(plugin):
    """If the agent actually does new work, we nudge again."""
    session_id = "dedup_test_session_resume"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(4)
    plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    msgs.append(_mutating_assistant_msg(args='{"path": "x4"}'))
    msgs.append({"role": "tool", "tool_call_id": "call_6", "content": "ok"})
    msgs.append(_mutating_assistant_msg(args='{"path": "x5"}'))
    msgs.append({"role": "tool", "tool_call_id": "call_7", "content": "ok"})
    result = plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    assert result["reflect"] is True, (
        f"reflect must re-arm after new mutating build; got {result!r}"
    )


def test_reflect_dedup_survives_compaction(plugin):
    """Compaction erases old tool calls but the agent does real new
    work — the dedup MUST re-arm and nudge. This is the v1-count bug
    the v2-signature fix addresses.
    """
    session_id = "dedup_test_session_compact"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(8)
    plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    # Simulate compaction: drop the first 5 messages (which held the
    # original 3 build calls). Current visible calls = 5 (different
    # signatures, but the count would also be 5 — same as the original
    # last_count of 8 minus a few. The signatures are what matters).
    compacted = msgs[5:]
    # Add 2 brand-new mutating calls after compaction.
    compacted.append(_mutating_assistant_msg(args='{"path": "after_compact_1"}'))
    compacted.append({"role": "tool", "tool_call_id": "c_after_1", "content": "ok"})
    compacted.append(_mutating_assistant_msg(args='{"path": "after_compact_2"}'))
    compacted.append({"role": "tool", "tool_call_id": "c_after_2", "content": "ok"})
    result = plugin._post_tool_batch_reflect(compacted, session_id=session_id)
    assert result["reflect"] is True, (
        f"after compaction + new work, reflect must re-arm; got {result!r}. "
        f"If this fails, the dedup is stuck on stale state from pre-compaction."
    )


def test_session_reset_clears_reflect_tail(plugin):
    """Session reset must drop the per-session dedup tail."""
    session_id = "dedup_test_session_reset"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(4)
    plugin._post_tool_batch_reflect(msgs, session_id=session_id)
    assert session_id in plugin._LAST_REFLECT_TAIL
    plugin._on_session_reset(session_id=session_id)
    assert session_id not in plugin._LAST_REFLECT_TAIL, (
        "on_session_reset must pop the dedup tail or a future session "
        "with the same id will silently never nudge"
    )


def test_runaway_loop_simulation(plugin):
    """End-to-end: 50 consecutive calls with same messages, exactly 1 reflect=True."""
    session_id = "dedup_test_runaway_sim"
    plugin._LAST_REFLECT_TAIL.pop(session_id, None)
    msgs = _build_msgs(5)
    results = [
        plugin._post_tool_batch_reflect(msgs, session_id=session_id)
        for _ in range(50)
    ]
    reflect_true_count = sum(1 for r in results if r.get("reflect") is True)
    reflect_false_count = sum(1 for r in results if r.get("reflect") is False)
    assert reflect_true_count == 1, (
        f"with no new build activity across 50 hook firings, exactly 1 must "
        f"return reflect=True; got {reflect_true_count} (rest should be False)"
    )
    assert reflect_false_count == 49, (
        f"49 of 50 must be silent; got {reflect_false_count} (runaway symptom)"
    )