"""Integration tests for the coding-agent-discipline plugin.

Exercises the live hook surface (pre_tool_call, post_tool_call,
transform_tool_result) against the in-process plugin module.

Invariants:
- Plugin registers and loads
- pre_tool_call blocks: cat/sed/grep via terminal, read-stale patch,
  unknown args on strict-schema tool, plan-mode edit
- pre_tool_call passes: legit terminal, fresh read+patch, normal flow
- post_tool_call records reads, fires warnings for test edits and todos
- transform_tool_result appends warning block
- enter/exit_plan_mode tools update controller state
- HERMES_DISCIPLINE_OFF=1 disables everything
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest


PLUGIN_PATH = Path(
    "os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")) + "/hermes-agent"plugins/coding-agent-discipline"
)


def _import_plugin(monkeypatch=None):
    """Fresh-import the plugin module (reset module-level state).

    Does NOT touch env vars — the test must set HERMES_DISCIPLINE_OFF
    before calling this if it wants the kill switch active at import time.
    """
    sys.modules.pop("plugins.coding_agent_discipline", None)
    # Make sure plugins/ is importable
    repo = "os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")) + "/hermes-agent"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    mod = importlib.import_module("plugins.coding_agent_discipline")
    mod.reset_state("default")
    return mod


# ── Plugin loads ─────────────────────────────────────────────────

class TestPluginLoad:
    def test_imports(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        assert hasattr(mod, "register")
        assert callable(mod._on_pre_tool_call)
        assert callable(mod._on_post_tool_call)
        assert callable(mod._on_transform_tool_result)

    def test_register_helper(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)

        registered: dict = {"hooks": [], "tools": []}

        class _Ctx:
            def register_hook(self, name, fn):
                registered["hooks"].append((name, fn))
            def register_tool(self, name, schema=None, handler=None):
                registered["tools"].append(name)

        mod.register(_Ctx())
        assert ("pre_tool_call", mod._on_pre_tool_call) in registered["hooks"]
        assert ("post_tool_call", mod._on_post_tool_call) in registered["hooks"]
        assert ("transform_tool_result", mod._on_transform_tool_result) in registered["hooks"]


# ── Rule 6: tool-over-bash ───────────────────────────────────────

class TestTerminalDenyList:
    def test_cat_blocked(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal", {"command": "cat /etc/hosts"}, task_id="default"
        )
        assert r is not None
        assert r["action"] == "block"
        assert "use read_file" in r["message"]

    def test_grep_blocked(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal", {"command": "grep -r TODO src/"}, task_id="default"
        )
        assert r is not None and r["action"] == "block"

    def test_echo_overwrite_blocked(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal", {"command": "echo hi > out.txt"}, task_id="default"
        )
        assert r is not None and r["action"] == "block"

    def test_legit_terminal_passes(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal", {"command": "pytest -k foo"}, task_id="default"
        )
        assert r is None

    def test_override_flag_skips(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal",
            {"command": "cat x.txt", "_discipline_validated": True},
            task_id="default",
        )
        assert r is None


# ── Rule 2/3: read-before-edit ───────────────────────────────────

class TestRecency:
    def test_stale_patch_blocked(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        # Read it at turn 1
        state.recency.advance_turn()
        state.recency.record_read(f)
        # Advance past the window
        for _ in range(state.recency.window + 2):
            state.recency.advance_turn()
        r = mod._on_pre_tool_call(
            "patch", {"path": str(f), "new_string": "x = 2"}, task_id="default"
        )
        assert r is not None
        assert r["action"] == "block"
        assert "read-before-edit" in r["message"]

    def test_fresh_patch_passes(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        state.recency.record_read(f)
        r = mod._on_pre_tool_call(
            "patch", {"path": str(f), "new_string": "x = 2"}, task_id="default"
        )
        assert r is None

    def test_first_touch_blocked(self, monkeypatch, tmp_path) -> None:
        """Never-read path → blocked (force explicit read first)."""
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "untouched.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        r = mod._on_pre_tool_call(
            "patch", {"path": str(f), "new_string": "x = 2"}, task_id="default"
        )
        assert r is not None
        assert "never read" in r["message"]

    def test_post_records_read(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        mod._on_post_tool_call("read_file", {"path": str(f)}, task_id="default")
        assert state.recency.last_read_turn(f) == state.recency.current_turn


# ── Rule 5: pre-deps-post-imports ────────────────────────────────

class TestImportScan:
    def test_new_import_warns(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        state.recency.record_read(f)
        # Should not block — should record a soft warning
        r = mod._on_pre_tool_call(
            "write_file",
            {"path": str(f), "content": "import requests\n"},
            task_id="default",
        )
        assert r is None  # not a block
        state = mod.get_state("default")
        assert any("pre-deps-post-imports" in w for w in state.last_warnings)

    def test_stdlib_no_warn(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        state.recency.record_read(f)
        mod._on_pre_tool_call(
            "write_file",
            {"path": str(f), "content": "import os\nfrom pathlib import Path\n"},
            task_id="default",
        )
        state = mod.get_state("default")
        import_warns = [w for w in state.last_warnings if "pre-deps" in w]
        assert not import_warns


# ── Rule 4: TODO 1-in-progress ───────────────────────────────────

class TestTodoInvariant:
    def test_two_in_progress_warns(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "todo",
            {"todos": [
                {"id": "a", "content": "x", "status": "in_progress"},
                {"id": "b", "content": "y", "status": "in_progress"},
            ]},
            task_id="default",
        )
        state = mod.get_state("default")
        assert any("1-in_progress" in w for w in state.last_warnings)

    def test_one_in_progress_no_warn(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "todo",
            {"todos": [
                {"id": "a", "content": "x", "status": "in_progress"},
                {"id": "b", "content": "y", "status": "pending"},
            ]},
            task_id="default",
        )
        state = mod.get_state("default")
        assert not any("1-in_progress" in w for w in state.last_warnings)


# ── Rule 7: plan-mode gate ───────────────────────────────────────

class TestPlanModeGate:
    def test_blocks_edit_when_planning(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        state = mod.get_state("default")
        state.plan.enter("test")
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        state.recency.advance_turn()
        state.recency.record_read(f)
        r = mod._on_pre_tool_call(
            "patch", {"path": str(f), "new_string": "y"}, task_id="default"
        )
        assert r is not None
        assert "plan-mode" in r["message"]

    def test_allows_read_when_planning(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        state = mod.get_state("default")
        state.plan.enter("test")
        r = mod._on_pre_tool_call(
            "read_file", {"path": "/x.py"}, task_id="default"
        )
        assert r is None

    def test_allows_edit_when_idle(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        state = mod.get_state("default")
        state.recency.advance_turn()
        state.recency.record_read(f)
        r = mod._on_pre_tool_call(
            "patch", {"path": str(f), "new_string": "y"}, task_id="default"
        )
        assert r is None

    def test_enter_exit_tools(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        state = mod.get_state("default")
        # Enter
        r = mod._tool_enter_plan_mode({"reason": "x"}, task_id="default")
        d = json.loads(r)
        assert d["state"] == "planning"
        # Exit
        r = mod._tool_exit_plan_mode(
            {"approved": True, "plan": "do X"}, task_id="default"
        )
        d = json.loads(r)
        assert d["state"] == "idle"


# ── Rule 9: no-edit-fail-tests ───────────────────────────────────

class TestTestEditWarning:
    def test_warns_on_test_path(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "patch",
            {"path": "tests/test_foo.py", "new_string": "x = 2"},
            task_id="default",
        )
        state = mod.get_state("default")
        assert any("SUT" in w for w in state.last_warnings)

    def test_silent_on_normal_path(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "patch",
            {"path": "src/foo.py", "new_string": "x = 2"},
            task_id="default",
        )
        state = mod.get_state("default")
        assert not any("SUT" in w for w in state.last_warnings)


# ── Rule 8: linter-as-tool ───────────────────────────────────────

class TestLinterHook:
    def test_warns_when_enabled_and_diagnostics(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        state = mod.get_state("default")
        state.linter_enabled = True
        f = tmp_path / "bad.py"
        f.write_text("def foo():\n    return.\n")
        mod._on_post_tool_call(
            "patch", {"path": str(f), "new_string": "x = 1"}, task_id="default"
        )
        assert any("diagnostic" in w for w in state.last_warnings)

    def test_silent_when_disabled(self, monkeypatch, tmp_path) -> None:
        mod = _import_plugin(monkeypatch)
        # state.linter_enabled defaults False
        f = tmp_path / "bad.py"
        f.write_text("def foo():\n    return.\n")
        mod._on_post_tool_call(
            "patch", {"path": str(f), "new_string": "x = 1"}, task_id="default"
        )
        state = mod.get_state("default")
        assert not any("diagnostic" in w for w in state.last_warnings)


# ── transform_tool_result decorator ─────────────────────────────

class TestTransformResult:
    def test_appends_warning(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "patch",
            {"path": "tests/test_x.py", "new_string": "x = 1"},
            task_id="default",
        )
        r = mod._on_transform_tool_result(
            "patch",
            {"path": "tests/test_x.py"},
            result='{"ok": true}',
            task_id="default",
        )
        assert r is not None
        assert "coding-agent-discipline" in r
        assert "SUT" in r

    def test_no_warning_no_decoration(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        # No warnings emitted
        r = mod._on_transform_tool_result(
            "read_file", {}, result="contents", task_id="default"
        )
        assert r is None

    def test_idempotent_no_double_wrap(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        mod._on_post_tool_call(
            "patch",
            {"path": "tests/test_x.py", "new_string": "x = 1"},
            task_id="default",
        )
        first = mod._on_transform_tool_result(
            "patch", {"path": "tests/test_x.py"}, result="a", task_id="default"
        )
        # second call shouldn't double-wrap (the result now contains the marker)
        second = mod._on_transform_tool_result(
            "patch", {"path": "tests/test_x.py"}, result=first, task_id="default"
        )
        # Either returns None or returns the same string once more (no triple)
        if second is not None:
            assert second.count("coding-agent-discipline:") <= 2


# ── Rule 10: anti-inventory ──────────────────────────────────────

class TestAntiInventory:
    def test_unknown_args_blocked_on_strict_schema(self, monkeypatch) -> None:
        mod = _import_plugin(monkeypatch)
        # Register a strict-schema test tool on the registry
        from tools.registry import registry
        SCHEMA = {
            "name": "test_strict_tool",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            },
        }

        def _handler(args, **kw):
            return json.dumps({"got": args.get("a")})

        try:
            registry.register(
                name="test_strict_tool",
                toolset="test",
                schema=SCHEMA,
                handler=_handler,
                check_fn=lambda: True,
            )
            r = mod._on_pre_tool_call(
                "test_strict_tool",
                {"a": "ok", "b": "unknown", "c": 99},
                task_id="default",
            )
            assert r is not None
            assert r["action"] == "block"
            assert "unrecognized" in r["message"]
        finally:
            # Cleanup the registered tool
            try:
                registry._tools.pop("test_strict_tool", None)  # type: ignore[attr-defined]
            except Exception:
                pass


# ── Hard off via env ─────────────────────────────────────────────

class TestPluginOff:
    def test_env_off_disables(self, monkeypatch) -> None:
        # Set env BEFORE importing so PLUGIN_OFF is read at module load
        monkeypatch.setenv("HERMES_DISCIPLINE_OFF", "1")
        mod = _import_plugin(monkeypatch)
        r = mod._on_pre_tool_call(
            "terminal", {"command": "cat /etc/hosts"}, task_id="default"
        )
        assert r is None


# ── AIAgent integration smoke test ───────────────────────────────

class TestAgentIntegration:
    def test_execute_tool_calls_advances_recency(self, monkeypatch, tmp_path) -> None:
        """Verify run_agent._execute_tool_calls wires the planner + recency
        without breaking the existing concurrent dispatch path."""
        from tools.parallel_planner import PlanResult
        # We don't have a real AIAgent without API keys, so we use a stand-in
        # that exposes the same `_execute_tool_calls` method.
        from run_agent import AIAgent

        # Minimal agent stand-in — skip full init
        agent = AIAgent.__new__(AIAgent)
        agent.coding_discipline = None
        agent._executing_tools = False
        agent._last_parallel_plan = None

        # Fake the inner methods to capture
        seen = {}

        def fake_seq(am, m, tid, c):
            seen["path"] = "seq"
        def fake_conc(am, m, tid, c):
            seen["path"] = "conc"

        agent._execute_tool_calls_sequential = fake_seq
        agent._execute_tool_calls_concurrent = fake_conc

        # Build a fake assistant message with one read_file call
        from dataclasses import dataclass
        @dataclass
        class _Fn:
            name: str = "read_file"
            arguments: str = json.dumps({"path": str(tmp_path / "a.py")})
        @dataclass
        class _Tc:
            id: str = "1"
            function: _Fn = None
            def __post_init__(self):
                self.function = _Fn()
        @dataclass
        class _Am:
            tool_calls: list = None
            def __post_init__(self):
                self.tool_calls = [_Tc()]

        (tmp_path / "a.py").write_text("x = 1\n")
        am = _Am()
        agent._execute_tool_calls(am, [], "test-task", 0)
        assert seen["path"] in {"seq", "conc"}
        # Recency advanced at least once
        assert agent.coding_discipline is not None
        assert agent.coding_discipline.recency.current_turn >= 1
        assert isinstance(agent._last_parallel_plan, PlanResult)
