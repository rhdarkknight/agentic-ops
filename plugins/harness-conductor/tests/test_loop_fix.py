"""Focused regressions for bounded harness-conductor review behavior."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))
hc = importlib.import_module("__init__")


def test_raw_dict_content_error_is_not_false_success():
    """Raw hook dicts do not have a data-envelope schema to prove success."""
    assert hc._is_raw_dict_error({"content": "Error: a"}) is True
    assert hc._is_raw_dict_error({"output": "Traceback (most recent call last)"}) is True
    assert hc._is_raw_dict_error({"content": "all clean", "success": True}) is False


def test_tool_errors_never_request_reflection():
    """Tool errors are observed but do not initiate another agent loop."""
    assert hc._post_tool_call(session_id="tool-error", result={"content": "Error: a"}) is None
    assert hc._post_tool_call(session_id="tool-error", result="Traceback: failed") is None


def test_tool_error_and_recovery_do_not_retrigger_batch_or_closeout():
    messages = [
        {"role": "user", "content": "Deploy the fix."},
        {"role": "tool", "content": "Error: first attempt failed"},
        {"role": "tool", "content": "recovered successfully"},
    ]
    assert hc._post_tool_batch_reflect(messages=messages) == {"reflect": False}
    assert hc._pre_exit_verify("Deployment completed.", messages=messages) == {"approved": True}
