"""Atomic Task Decomposition Pipeline plugin — register(ctx) entry point."""
import os

from .core.classifier import TaskClassifier
from .core.orchestrator import AtomicOrchestrator
from .tools.atomic_localize import atomic_code_localize
from .tools.atomic_edit import atomic_code_edit
from .tools.atomic_test_gen import atomic_unit_test_gen
from .tools.atomic_reproduce import atomic_issue_reproduce
from .tools.atomic_review import atomic_code_review
from .hooks.pre_llm_hook import on_pre_llm_call
from .hooks.post_tool_hook import on_post_tool_call
from .hooks.session_hooks import on_session_finalize, on_session_reset, on_session_start


LOCALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string", "description": "Issue description text"},
        "repo_context": {"type": "object", "description": "Optional repo context with 'files' key"},
    },
    "required": ["issue"],
}

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to the file to edit"},
        "edit_instruction": {"type": "string", "description": "Natural language description of the change"},
        "run_tests": {"type": "boolean", "description": "Whether to run tests after editing", "default": True},
    },
    "required": ["file_path", "edit_instruction"],
}

TEST_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "function_code": {"type": "string", "description": "Source code of the function to test"},
        "specification": {"type": "string", "description": "Description of expected behavior"},
        "test_file": {"type": "string", "description": "Optional path for generated test file"},
    },
    "required": ["function_code", "specification"],
}

REPRODUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_description": {"type": "string", "description": "Description of the bug/issue"},
        "repo_context": {"type": "object", "description": "Optional repo context"},
    },
    "required": ["issue_description"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "pr_diff": {"type": "string", "description": "Unified diff of the PR changes"},
        "issue_description": {"type": "string", "description": "Original issue/bug description"},
        "pr_files": {"type": "array", "items": {"type": "string"}, "description": "Optional list of files changed"},
    },
    "required": ["pr_diff", "issue_description"],
}


def register(ctx):
    """Plugin entry point — register tools and hooks with Hermes context."""
    if not os.getenv("HERMES_ATOMIC_ENABLED", "1") == "1":
        return

    # Register 5 atomic skill tools
    ctx.register_tool(
        name="atomic_code_localize",
        toolset="atomic_pipeline",
        schema=LOCALIZE_SCHEMA,
        handler=atomic_code_localize,
        description="Atomic skill: identify files relevant to an issue",
        emoji="🎯",
    )

    ctx.register_tool(
        name="atomic_code_edit",
        toolset="atomic_pipeline",
        schema=EDIT_SCHEMA,
        handler=lambda args, **kw: atomic_code_edit(**args, **kw),
        description="Atomic skill: generate a code patch from edit instruction",
        emoji="✏️",
    )

    ctx.register_tool(
        name="atomic_unit_test_gen",
        toolset="atomic_pipeline",
        schema=TEST_GEN_SCHEMA,
        handler=atomic_unit_test_gen,
        description="Atomic skill: generate unit tests with mutation-based fault detection",
        emoji="🧪",
    )

    ctx.register_tool(
        name="atomic_issue_reproduce",
        toolset="atomic_pipeline",
        schema=REPRODUCE_SCHEMA,
        handler=atomic_issue_reproduce,
        description="Atomic skill: create a script that reproduces the reported failure",
        emoji="🐛",
    )

    ctx.register_tool(
        name="atomic_code_review",
        toolset="atomic_pipeline",
        schema=REVIEW_SCHEMA,
        handler=lambda args, **kw: atomic_code_review(**args, **kw),
        description="Atomic skill: judge whether a PR correctly addresses the issue",
        emoji="🔍",
    )

    # Register hooks
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_finalize", on_session_finalize)
