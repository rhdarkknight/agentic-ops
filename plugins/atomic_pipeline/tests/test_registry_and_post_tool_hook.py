"""Regression coverage for registry dispatch and post-tool result handling."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


_CORE_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "hermes-agent" / "tools" / "registry.py"
_CORE_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "atomic_pipeline_test_core_registry", _CORE_REGISTRY_PATH
)
assert _CORE_REGISTRY_SPEC is not None and _CORE_REGISTRY_SPEC.loader is not None
_CORE_REGISTRY = importlib.util.module_from_spec(_CORE_REGISTRY_SPEC)
sys.modules[_CORE_REGISTRY_SPEC.name] = _CORE_REGISTRY
_CORE_REGISTRY_SPEC.loader.exec_module(_CORE_REGISTRY)
ToolRegistry = _CORE_REGISTRY.ToolRegistry

from .. import register
from ..core.orchestrator import AtomicOrchestrator
from ..hooks.post_tool_hook import on_post_tool_call


class _RegistryContext:
    """Minimal plugin context backed by the real core ToolRegistry."""

    def __init__(self):
        self.registry = ToolRegistry()

    def register_tool(self, **kwargs):
        self.registry.register(**kwargs)

    def register_hook(self, *_args, **_kwargs):
        pass


def test_code_review_dispatch_expands_model_arguments_for_registry_handler():
    """The core dispatcher calls handlers as handler(args, **metadata)."""
    context = _RegistryContext()
    register(context)

    response = context.registry.dispatch(
        "atomic_code_review",
        {
            "pr_diff": "--- a/timeout.py\n+++ b/timeout.py\n+fix timeout\n+assert timeout",
            "issue_description": "Fix the timeout bug",
            "pr_files": ["timeout.py"],
        },
        task_id="review-task-1",
    )

    payload = json.loads(response)
    assert payload["success"] is True
    assert payload["skill"] == "code_review"
    assert payload["data"]["judgment"] == "accept"


@patch.object(AtomicOrchestrator, "get_state", return_value=object())
@patch.object(AtomicOrchestrator, "record_skill_result")
def test_post_tool_hook_records_success_and_failure_json_payloads(record, _state):
    """Both successful and failed atomic JSON results are recorded unchanged."""
    success_payload = {"success": True, "data": {"judgment": "accept"}}
    failure_payload = {"success": False, "error": "review failed"}

    assert on_post_tool_call("atomic_code_review", json.dumps(success_payload)) == {}
    assert on_post_tool_call("atomic_code_review", json.dumps(failure_payload)) == {}

    assert record.call_args_list == [
        (("code_review", success_payload),),
        (("code_review", failure_payload),),
    ]


@patch.object(AtomicOrchestrator, "get_state", return_value=object())
@patch.object(AtomicOrchestrator, "record_skill_result")
def test_post_tool_hook_ignores_malformed_json_payload(record, _state):
    """Malformed tool output is ignored without breaking the hook pipeline."""
    assert on_post_tool_call("atomic_code_review", "not-json") == {}
    record.assert_not_called()
