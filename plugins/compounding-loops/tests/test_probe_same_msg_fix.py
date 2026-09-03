"""Regression test — pass marker and fix in the same assistant message.

_count_build_tool_calls_after_pass must count mutating tool calls in the
same message that contains the review pass marker, because the agent
often reports a review and issues a fix in one turn. Previously the
function `continue`d past the boundary message's tool_calls, under-
counting and leaving the reflect hook silent when it should nudge.
"""

from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    spec = importlib.util.spec_from_file_location(
        "plugins.cl_same_msg", _PLUGIN_DIR / "__init__.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["plugins.cl_same_msg"] = m
    spec.loader.exec_module(m)
    return m


def _pc(cid, name):
    return {"id": cid, "function": {"name": name, "arguments": "{}"}}


def test_count_includes_fix_in_same_message_as_pass(plugin):
    """A patch in the same assistant turn as "Review pass 1:" must be
    counted as a post-review build (count == 1)."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant",
         "content": "Review pass 1: 1 blocker. Applying fix.",
         "tool_calls": [_pc("f1", "patch")]},
        {"role": "tool", "tool_call_id": "f1", "content": "ok"},
    ]
    assert plugin._count_build_tool_calls_after_pass(msgs, 1) == 1


def test_reflect_nudges_when_fix_in_same_msg_as_review(plugin):
    """Consequence: reflect hook must nudge pass 2 when a fix lands in
    the same turn as the review that found the blocker."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant",
         "content": "Review pass 1: 1 blocker. Applying fix.",
         "tool_calls": [_pc("f1", "patch")]},
        {"role": "tool", "tool_call_id": "f1", "content": "ok"},
    ]
    decision = plugin._post_tool_batch_reflect(messages=msgs)
    assert decision["reflect"] is True
    assert "review pass 2" in decision["reason"].lower()


def test_count_excludes_mutating_calls_before_pass_marker(plugin):
    """The boundary message's own tool_calls are scanned, but a build
    call BEFORE the boundary message is not counted (only the
    post-review build matters)."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker."},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("f1", "patch")]},
        {"role": "tool", "tool_call_id": "f1", "content": "ok"},
    ]
    # write_file (w1) is before the pass marker; only patch (f1) counts.
    assert plugin._count_build_tool_calls_after_pass(msgs, 1) == 1


def test_count_does_not_trigger_on_unrelated_pass_number(plugin):
    """A message containing "Review pass 2:" must not set the boundary
    when after_pass == 1."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "assistant", "content": "Review pass 2: 0 blockers."},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("f1", "patch")]},
    ]
    # after_pass=1 is never matched → boundary never set → count == 0.
    assert plugin._count_build_tool_calls_after_pass(msgs, 1) == 0