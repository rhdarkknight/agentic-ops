"""Regression coverage for the former harness-conductor tool-error loop."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PLUGIN_HC = Path(__file__).resolve().parent.parent / "__init__.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hc = _load("e2e_hc", PLUGIN_HC)


def test_recovered_raw_dict_error_cannot_restart_batch_or_closeout_review():
    """A raw error is classified correctly but never starts a reflection loop."""
    raw_error = {"content": "Error: a"}
    assert hc._is_raw_dict_error(raw_error) is True

    # Individual errors and their successful recovery are observation-only.
    assert hc._post_tool_call(session_id="recovery", result=raw_error) is None
    assert hc._post_tool_call(session_id="recovery", result={"content": "all clean"}) is None

    messages = [
        {"role": "user", "content": "Implement the bounded cleanup."},
        {"role": "assistant", "tool_calls": [{"id": "bad", "function": {"name": "terminal"}}]},
        {"role": "tool", "tool_call_id": "bad", "content": "Error: a"},
        {"role": "assistant", "tool_calls": [{"id": "good", "function": {"name": "terminal"}}]},
        {"role": "tool", "tool_call_id": "good", "content": "all clean"},
    ]

    # Neither the next batch nor the final closeout is re-triggered by the
    # prior error, including a high-stakes request.
    assert hc._post_tool_batch_reflect(messages=messages) == {"reflect": False}
    assert hc._pre_exit_verify("Implemented and verified.", messages=messages) == {"approved": True}
