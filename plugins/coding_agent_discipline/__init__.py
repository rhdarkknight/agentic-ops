"""coding-agent-discipline plugin — wires the 10 cross-prompt patterns
into the live Hermes agent loop.

Patterns enforced (via pre_tool_call / post_tool_call / transform_tool_result):

  1. parallel-by-default          — pre_tool_call group-aware block warn
  2. read-before-edit             — pre_tool_call recency check (block)
  3. recency-window               — tracked here (per-task, ContextVar)
  4. todo-1-in-progress           — post_tool_call (warn on violation)
  5. pre-deps-post-imports        — pre_tool_call (block + suggest install)
  6. tool-over-bash               — pre_tool_call (block terminal w/ cat/sed/...)
  7. plan-mode gate               — pre_tool_call (block edits in PLANNING)
  8. linter-as-tool               — post_tool_call (warn, opt-in)
  9. no-edit-fail-tests           — post_tool_call (warn on test file edit)
 10. anti-inventory               — pre_tool_call (block unknown args)

State: per-task RecencyTracker + PlanModeController stored in a ContextVar
on `agent.coding_discipline` (set by the agent init if attribute exists).
For unit tests we expose module-level helpers that read from a per-thread
fallback (see `_get_recency_for_task`).

Hard opt-outs via env (escape hatch):
  * ``HERMES_DISCIPLINE_OFF=1``  — disable the plugin entirely
  * ``HERMES_DISCIPLINE_NUDGE=0`` — turn blocks into soft warnings
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


PLUGIN_OFF = os.getenv("HERMES_DISCIPLINE_OFF") == "1"
NUDGE_MODE = os.getenv("HERMES_DISCIPLINE_NUDGE") == "0"  # default: blocks


# Per-task state. Tests construct their own; live agent attaches via
# `agent.coding_discipline = CodingDisciplineState()`.
class CodingDisciplineState:
    """Mutable per-task state container."""

    def __init__(
        self,
        recency_window: int = 5,
        plan_mode: str = "auto",
        linter_enabled: bool = False,
        parallel_enabled: bool = True,
    ) -> None:
        from tools.recency import RecencyTracker
        from tools.plan_mode import PlanModeController

        self.recency = RecencyTracker(window=recency_window)
        self.plan = PlanModeController()
        self.plan.configure(plan_mode)
        self.linter_enabled = linter_enabled
        self.parallel_enabled = parallel_enabled
        # last warn appended into a tool result, used to dedupe in tests
        self.last_warnings: list[str] = []
        self._lock = threading.Lock()

    def warn(self, msg: str) -> None:
        with self._lock:
            self.last_warnings.append(msg)


# Module-level fallback for contexts where the agent doesn't carry state
# (unit tests, sub-tools). Keyed by task_id, scoped to current process.
_state_by_task: Dict[str, CodingDisciplineState] = {}
_state_lock = threading.Lock()


def get_state(task_id: str = "default") -> CodingDisciplineState:
    with _state_lock:
        st = _state_by_task.get(task_id)
        if st is None:
            st = CodingDisciplineState()
            _state_by_task[task_id] = st
        return st


def set_state(task_id: str, state: CodingDisciplineState) -> None:
    with _state_lock:
        _state_by_task[task_id] = state


def reset_state(task_id: str = "default") -> None:
    with _state_lock:
        _state_by_task.pop(task_id, None)


def _try_get_agent_state(task_id: str) -> Optional[CodingDisciplineState]:
    """Best-effort fetch from the AIAgent instance. Returns None on miss."""
    try:
        from run_agent import AIAgent  # type: ignore
    except Exception:
        return None
    # task_id may map to an agent — we can't iterate agents, so we rely on
    # the caller to have set module-level state. The live wiring in
    # `wire_agent.py` (separate file) will attach this for us.
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block(message: str) -> Dict[str, Any]:
    """Build a pre_tool_call block directive."""
    if NUDGE_MODE:
        # Soft mode: return None to allow, but record the warning
        logger.info("coding-agent-discipline nudge: %s", message)
        get_state().warn(message)
        return None  # type: ignore[return-value]
    return {"action": "block", "message": message}


def _resolve_path(args: Dict[str, Any]) -> Optional[str]:
    p = args.get("path") or args.get("file_path") or args.get("filepath")
    if not isinstance(p, str) or not p.strip():
        return None
    return p


def _extract_content(args: Dict[str, Any]) -> str:
    """Pull 'code being written' out of write_file / patch / skill_manage."""
    parts: list[str] = []
    for key in ("content", "new_string", "patch", "file_content"):
        v = args.get(key)
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rule 6: tool-over-bash (terminal pre-hook)
# ---------------------------------------------------------------------------

def _check_terminal(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from tools.agent_discipline import check_shell_command
    cmd = args.get("command") or args.get("cmd") or args.get("script")
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    # Skip when the caller explicitly flagged it as already-validated
    if args.get("_discipline_validated") is True:
        return None
    result = check_shell_command(cmd)
    if not result.allowed:
        msg = (
            f"coding-agent-discipline: blocked shell command — use "
            f"{result.suggestion} instead of `{cmd.splitlines()[0][:60]}…`. "
            f"Set _discipline_validated=true to override."
        )
        return _block(msg)
    return None


# ---------------------------------------------------------------------------
# Rule 2/3: read-before-edit (recency)
# ---------------------------------------------------------------------------

def _check_recency(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
) -> Optional[Dict[str, Any]]:
    if tool_name not in {"write_file", "patch"}:
        return None
    p = _resolve_path(args)
    if not p:
        return None
    res = state.recency.check_edit(p)
    if not res.allowed:
        msg = (
            f"coding-agent-discipline: read-before-edit — path was last "
            f"read {res.stale_turns} turns ago (window={res.window}). "
            f"Re-read with read_file first, then re-attempt the edit. "
            f"({res.reason})"
        )
        return _block(msg)
    return None


def _record_read(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
) -> None:
    if tool_name != "read_file":
        return
    p = _resolve_path(args)
    if p:
        state.recency.record_read(p)


# ---------------------------------------------------------------------------
# Rule 5: pre-deps-post-imports
# ---------------------------------------------------------------------------

_IMPORT_PKG_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)|import\s+([\w.]+))",
    re.M,
)


def _scan_new_imports(content: str) -> tuple[str, ...]:
    """Return top-level package names referenced but not in stdlib set.

    Light heuristic — uses agent_discipline.scan_imports under the hood
    when available.
    """
    if not content.strip():
        return ()
    try:
        from tools.agent_discipline import scan_imports
        r = scan_imports(content)
        return r.new_packages
    except Exception:
        return ()


def _check_imports(
    tool_name: str,
    args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if tool_name not in {"write_file", "patch"}:
        return None
    content = _extract_content(args)
    new_pkgs = _scan_new_imports(content)
    if not new_pkgs:
        return None
    # Don't block — instruct the model to install first
    pkgs = ", ".join(new_pkgs)
    msg = (
        f"coding-agent-discipline: pre-deps-post-imports — file imports "
        f"new package(s): {pkgs}. Run `pip install {pkgs}` BEFORE the "
        f"next edit on this file to avoid missing-module churn. The write "
        f"still proceeded; this is a reminder."
    )
    # Soft: warn, don't block
    get_state().warn(msg)
    logger.info(msg)
    return None


# ---------------------------------------------------------------------------
# Rule 4: TODO 1-in-progress (post-hook)
# ---------------------------------------------------------------------------

def _check_todo_invariant(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
) -> Optional[str]:
    if tool_name != "todo":
        return None
    # todo tool may take `todos` param to write; both modes return items
    items = args.get("todos")
    if not isinstance(items, list):
        return None
    try:
        from tools.agent_discipline import check_todo_invariant
        r = check_todo_invariant(items)
    except Exception:
        return None
    if r.ok:
        return None
    state.warn(r.message)
    return r.message


# ---------------------------------------------------------------------------
# Rule 7: plan-mode gate
# ---------------------------------------------------------------------------

_EDIT_TOOLS = frozenset({"write_file", "patch", "terminal", "execute_code"})


def _check_plan_mode(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
    user_message: str = "",
) -> Optional[Dict[str, Any]]:
    if state.plan.state.value != "planning":
        return None
    if tool_name not in _EDIT_TOOLS:
        return None
    msg = (
        "coding-agent-discipline: plan-mode active — edit tools are "
        "blocked. Call exit_plan_mode (approved=True, plan=...) to resume."
    )
    return _block(msg)


# ---------------------------------------------------------------------------
# Rule 10: anti-inventory (unknown args)
# ---------------------------------------------------------------------------

def _check_unknown_args(
    tool_name: str,
    args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        from tools.registry import registry
    except Exception:
        return None
    schema = registry.get_schema(tool_name)
    if not isinstance(schema, dict):
        return None
    try:
        from tools.agent_discipline import is_strict_schema, find_unrecognized_args
        if not is_strict_schema(schema):
            return None
        unknown = find_unrecognized_args(schema, args)
    except Exception:
        return None
    if not unknown:
        return None
    msg = (
        f"coding-agent-discipline: tool `{tool_name}` received unrecognized "
        f"argument(s): {', '.join(unknown)}. Strict-schema mode forbids "
        f"fabricated params. Check the tool schema."
    )
    return _block(msg)


# ---------------------------------------------------------------------------
# Rule 8: linter-as-tool (post-hook)
# ---------------------------------------------------------------------------

def _maybe_lint(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
) -> Optional[str]:
    if not state.linter_enabled:
        return None
    if tool_name not in {"write_file", "patch"}:
        return None
    p = _resolve_path(args)
    if not p:
        return None
    try:
        from tools.lint import lint_path
    except Exception:
        return None
    res = lint_path(p, enabled=True, timeout_seconds=15.0)
    if res.skipped or res.error or not res.diagnostics:
        return None
    n = len(res.diagnostics)
    first = res.diagnostics[0]
    snippet = (
        f" [{first.file}:{first.line}:{first.column} {first.code} {first.message[:80]}]"
        if n else ""
    )
    msg = (
        f"coding-agent-discipline: read_lints surfaced {n} diagnostic(s) "
        f"for {p}{snippet}. Fix before next iteration."
    )
    state.warn(msg)
    return msg


# ---------------------------------------------------------------------------
# Rule 9: no-edit-fail-tests (post-hook)
# ---------------------------------------------------------------------------

def _check_test_edit(
    tool_name: str,
    args: Dict[str, Any],
    state: CodingDisciplineState,
) -> Optional[str]:
    if tool_name not in {"write_file", "patch"}:
        return None
    p = _resolve_path(args)
    if not p:
        return None
    try:
        from tools.agent_discipline import detect_test_path_edit
        w = detect_test_path_edit(p)
    except Exception:
        return None
    if not w.is_test_edit:
        return None
    msg = w.hint
    state.warn(msg)
    return msg


# ---------------------------------------------------------------------------
# Hook entry points
# ---------------------------------------------------------------------------

def _on_pre_tool_call(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    **_: Any,
) -> Optional[Dict[str, Any]]:
    if PLUGIN_OFF:
        return None
    if not isinstance(args, dict):
        args = {}
    state = get_state(task_id or "default")

    # 1. anti-inventory — block unknown args
    r = _check_unknown_args(tool_name, args)
    if r:
        return r

    # 2. tool-over-bash — terminal deny-list
    if tool_name in {"terminal", "shell", "shell_exec"}:
        r = _check_terminal(args)
        if r:
            return r

    # 3. read-before-edit
    r = _check_recency(tool_name, args, state)
    if r:
        return r

    # 4. plan-mode gate
    r = _check_plan_mode(tool_name, args, state)
    if r:
        return r

    # 5. pre-deps-post-imports — soft warning, not block
    _check_imports(tool_name, args)

    return None


def _on_post_tool_call(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    **_: Any,
) -> None:
    if PLUGIN_OFF:
        return
    if not isinstance(args, dict):
        args = {}
    state = get_state(task_id or "default")

    # Track reads for the recency check
    _record_read(tool_name, args, state)

    # Warn-only checks (recorded into state.last_warnings for tests)
    _check_todo_invariant(tool_name, args, state)
    _check_test_edit(tool_name, args, state)
    _maybe_lint(tool_name, args, state)


def _on_transform_tool_result(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    **_: Any,
) -> Optional[str]:
    """Append discipline warnings to a tool result that the model will see.

    Only attaches when there are pending warnings; never blocks.
    """
    if PLUGIN_OFF:
        return None
    state = get_state(task_id or "default")
    if not state.last_warnings:
        return None
    # Drain the most recent warning (per-result, not cumulative)
    last = state.last_warnings[-1]
    if not isinstance(result, str):
        try:
            result = json.dumps(result)
        except Exception:
            return None
    # Don't double-decorate
    if "[coding-agent-discipline]" in result:
        return None
    return result + "\n\n[coding-agent-discipline] " + last


# ---------------------------------------------------------------------------
# Tool registration: enter_plan_mode / exit_plan_mode
# ---------------------------------------------------------------------------

ENTER_PLAN_SCHEMA = {
    "name": "enter_plan_mode",
    "description": (
        "Enter plan mode. Edit tools will be blocked until you call "
        "exit_plan_mode. Use for non-trivial changes (3+ files, new "
        "feature, refactor, large diff). Auto-triggered by heuristics "
        "when plan_mode=auto."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why plan mode is needed (logged)",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
}

EXIT_PLAN_SCHEMA = {
    "name": "exit_plan_mode",
    "description": (
        "Exit plan mode. Edit tools re-enabled. If approved=False, the "
        "plan was rejected and the model should re-plan or ask."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "Whether the user approved the plan",
            },
            "plan": {
                "type": "string",
                "description": "The final plan text (when approved=True)",
            },
        },
        "required": ["approved"],
        "additionalProperties": False,
    },
}


def _tool_enter_plan_mode(args: Dict[str, Any], **kw: Any) -> str:
    state = get_state(kw.get("task_id") or "default")
    state.plan.enter(args.get("reason", "manual"))
    return json.dumps({"state": state.plan.state.value})


def _tool_exit_plan_mode(args: Dict[str, Any], **kw: Any) -> str:
    state = get_state(kw.get("task_id") or "default")
    state.plan.exit(
        approved=bool(args.get("approved", False)),
        plan_text=str(args.get("plan", "")),
    )
    return json.dumps({"state": state.plan.state.value})


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    from tools.registry import registry, tool_error

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)

    try:
        registry.register(
            name="enter_plan_mode",
            toolset="coding-agent-discipline",
            schema=ENTER_PLAN_SCHEMA,
            handler=_tool_enter_plan_mode,
            check_fn=lambda: True,
            emoji="📝",
        )
        registry.register(
            name="exit_plan_mode",
            toolset="coding-agent-discipline",
            schema=EXIT_PLAN_SCHEMA,
            handler=_tool_exit_plan_mode,
            check_fn=lambda: True,
            emoji="✅",
        )
    except Exception as e:
        # Don't fail plugin load if registry already has them
        logger.debug("plan-mode tool registration skipped: %s", e)


# ---------------------------------------------------------------------------
# Public batch-plan API (replaces run_agent.py +35 inline dance, 2026-06-11)
# ---------------------------------------------------------------------------

def batch_plan(agent: Any, tool_calls: List[Any], task_id: str) -> "ParallelPlan":
    """Pre-group a tool-call batch by dependency graph + advance recency turn.

    Replaces the ~35-line inline block that previously lived in
    ``run_agent.AIAgent._execute_tool_calls``. Idempotent and safe to call
    even when the plugin is half-broken: the only state-mutating side effects
    are ``agent.coding_discipline`` setattr and ``state.recency.advance_turn``.

    Args:
        agent: The AIAgent instance (for state attachment + observability stash).
        tool_calls: List of tool calls from the assistant message.
        task_id: Effective task id (falls back to ``"default"`` if empty).

    Returns:
        The ``ParallelPlan`` produced by ``plan_tool_calls`` so callers can
        inspect groups. Always returns a plan; never raises.
    """
    from tools.parallel_planner import plan_tool_calls
    from tools.recency import RecencyTracker  # noqa: F401  re-export
    from tools.plan_mode import PlanModeController  # noqa: F401  re-export

    if PLUGIN_OFF:
        return plan_tool_calls(tool_calls, enabled=False)

    state = getattr(agent, "coding_discipline", None)
    if state is None:
        state = get_state(task_id or "default")
        try:
            agent.coding_discipline = state
        except Exception:
            pass
    try:
        state.recency.advance_turn()
    except Exception:
        pass
    plan = plan_tool_calls(
        tool_calls,
        enabled=getattr(state, "parallel_enabled", True),
    )
    try:
        agent._last_parallel_plan = plan
    except Exception:
        pass
    return plan


# Backwards-compat shim — old code in run_agent.py used these names
def get_state_for_agent(agent: Any, task_id: str) -> "CodingDisciplineState":
    state = getattr(agent, "coding_discipline", None)
    if state is None:
        state = get_state(task_id or "default")
        try:
            agent.coding_discipline = state
        except Exception:
            pass
    return state
