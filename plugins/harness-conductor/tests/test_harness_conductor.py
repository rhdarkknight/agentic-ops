"""Tests for the harness-conductor plugin decisions.

These tests import the plugin's hook callbacks directly and assert that only
final-response shape problems trigger a bounded retry. Tool errors and large
batches remain observation-only. They do not exercise the full agent loop.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Resolve the plugin directory relative to this test file. Tests live at
#   <plugin_dir>/tests/test_harness_conductor.py
# so parent.parent is the plugin dir. This works whether the plugin is
# shipped standalone under ~/.hermes/plugins/ or inside the agent tree.
HERMES_AGENT = Path(__file__).resolve().parent.parent
if str(HERMES_AGENT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT))


import importlib.util
import pytest


_PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_harness_conductor():
    """Load the plugin module the way the runtime does: from its directory."""
    spec = importlib.util.spec_from_file_location(
        "plugins.harness_conductor", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.harness_conductor"] = module
    spec.loader.exec_module(module)
    return module


_hc = _load_harness_conductor()

_collect_tool_errors = _hc._collect_tool_errors
_looks_high_stakes = _hc._looks_high_stakes
_looks_incomplete = _hc._looks_incomplete
_post_tool_batch_reflect = _hc._post_tool_batch_reflect
_post_tool_call = _hc._post_tool_call
_pre_exit_verify = _hc._pre_exit_verify
_is_raw_dict_error = _hc._is_raw_dict_error


def test_harness_manifest_declares_every_registered_hook():
    registered = []
    _hc.register(SimpleNamespace(register_hook=lambda name, callback: registered.append(name)))

    def declared(section):
        values = []
        in_section = False
        for line in (_PLUGIN_DIR / "plugin.yaml").read_text().splitlines():
            if line == f"{section}:":
                in_section = True
                continue
            if in_section and line.startswith("  - "):
                values.append(line.removeprefix("  - "))
            elif in_section:
                break
        return values

    assert set(declared("hooks")) == set(registered)
    assert set(declared("provides_hooks")) == set(registered)


# -----------------------------------------------------------------------------
# pre_exit_verify
# -----------------------------------------------------------------------------

def test_pre_exit_verify_rejects_empty_response():
    decision = _pre_exit_verify("", messages=[])
    assert decision == {
        "approved": False,
        "reason": "final response is empty",
        "max_reverify_passes": 1,
    }


def test_pre_exit_verify_rejects_whitespace_only_response():
    decision = _pre_exit_verify("   \n\t  ", messages=[])
    assert decision == {
        "approved": False,
        "reason": "final response is empty",
        "max_reverify_passes": 1,
    }


def test_post_api_request_preserves_empty_response_safeguard():
    session_id = "empty-response"
    _hc._HARNESS_VERDICTS.pop(session_id, None)
    _hc._post_api_request(
        session_id=session_id,
        assistant_message=SimpleNamespace(content=""),
        finish_reason="stop",
    )
    assert _hc._HARNESS_VERDICTS[session_id]["approved"] is False
    transformed = _hc._transform_llm_output(session_id=session_id, response_text="")
    assert transformed is not None
    assert "final response is empty" in transformed


def test_pre_exit_verify_approves_prose_ending_in_ellipsis():
    assert _pre_exit_verify("The fix is complete...", messages=[]) == {"approved": True}


def test_pre_exit_verify_approves_prose_ending_in_parenthesis():
    assert _pre_exit_verify("See the documented caveat (", messages=[]) == {"approved": True}


def test_pre_exit_verify_rejects_unclosed_code_block():
    decision = _pre_exit_verify("```python\nprint('hi')", messages=[])
    assert decision["approved"] is False
    assert "code block" in decision["reason"].lower()


def test_pre_exit_verify_approves_closed_code_block():
    decision = _pre_exit_verify("```python\nprint('hi')\n```", messages=[])
    assert decision == {"approved": True}


def test_pre_exit_verify_approves_four_backtick_block_with_literal_triple_backticks():
    response = "````markdown\n```\nliteral fence content\n````"
    assert _pre_exit_verify(response, messages=[]) == {"approved": True}


def test_pre_exit_verify_approves_confidence_hedge():
    decision = _pre_exit_verify("I'm not sure, but maybe try rebooting.", messages=[])
    assert decision == {"approved": True}


def test_pre_exit_verify_approves_complete_response():
    decision = _pre_exit_verify("Done. The file was written to /tmp/out.txt.", messages=[])
    assert decision == {"approved": True}


def test_pre_exit_verify_approves_when_high_stakes_and_tool_error():
    messages = [
        {"role": "user", "content": "Write a python script that sorts a list."},
        {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "execute_code"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "Error: NameError: name 'sort' is not defined"},
    ]
    decision = _pre_exit_verify("Here is the script: `sorted(lst)`", messages=messages)
    assert decision == {"approved": True}


def test_pre_exit_verify_approves_when_low_stakes_and_tool_error():
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "execute_code"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "Error: failed to fetch"},
    ]
    decision = _pre_exit_verify("I could not fetch the weather.", messages=messages)
    assert decision == {"approved": True}


# -----------------------------------------------------------------------------
# post_tool_batch_reflect
# -----------------------------------------------------------------------------

def test_tool_error_does_not_reflect():
    messages = [
        {"role": "user", "content": "Run a script."},
        {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "execute_code"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "Traceback (most recent call last): ..."},
    ]
    decision = _post_tool_batch_reflect(messages=messages)
    assert decision == {"reflect": False}


def test_reflect_approves_clean_small_batch():
    messages = [
        {"role": "user", "content": "Run a script."},
        {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "execute_code"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
    ]
    decision = _post_tool_batch_reflect(messages=messages)
    assert decision == {"reflect": False}


def test_large_batch_does_not_reflect():
    messages = [
        {"role": "user", "content": "Run a script."},
        {"role": "assistant", "tool_calls": [{"id": f"tc{i}", "function": {"name": "execute_code"}} for i in range(4)]},
    ] + [
        {"role": "tool", "tool_call_id": f"tc{i}", "content": "ok"}
        for i in range(4)
    ]
    decision = _post_tool_batch_reflect(messages=messages)
    assert decision == {"reflect": False}


def test_raw_dict_content_error_is_classified_without_requesting_reflection():
    raw_error = {"content": "Error: a"}
    assert _is_raw_dict_error(raw_error) is True
    assert _post_tool_call(session_id="raw-dict", result=raw_error) is None


# -----------------------------------------------------------------------------
# Helper internals
# -----------------------------------------------------------------------------

def test_collect_tool_errors_finds_error_keywords():
    messages = [
        {"role": "tool", "tool_call_id": "tc1", "content": "Error: something broke"},
    ]
    assert _collect_tool_errors(messages) == ["Error: something broke"]


def test_collect_tool_errors_ignores_non_tool_messages():
    messages = [
        {"role": "user", "content": "Error in my request"},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
    ]
    assert _collect_tool_errors(messages) == []


def test_looks_high_stakes_detects_code_tasks():
    assert _looks_high_stakes([{"role": "user", "content": "Write a python function"}]) is True


def test_looks_high_stakes_false_for_casual():
    assert _looks_high_stakes([{"role": "user", "content": "How are you today?"}]) is False


def test_looks_incomplete_returns_none_for_complete():
    assert _looks_incomplete("Complete answer.") is None
