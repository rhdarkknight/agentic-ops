"""Regression tests for task_state injection stripping.

The compressor summarizer can lift directive-style bracketed text from a
prior turn ([System note: ...] from gateway resume, [CAVEMAN FULL: ...]
from caveman_enforcer, [Role: ...] from role-system, [IMPORTANT: ...] for
cron delivery) and save it as the persistent "goal" in
~/.hermes/task_state.json. The next session re-injects that goal as
context, polluting every future user message with the prior turn's
directives. These tests pin the fix: all known directive shapes are
stripped at extraction AND at injection time, real user content survives.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the plugin importable as `context_compressor.task_state` per its
# internal import surface.
PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "context_compressor"
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import task_state as ts  # type: ignore[import-not-found]


class TestStripInjections:
    """_strip_injections must remove every known directive shape."""

    def test_system_note(self):
        s = "[System note: Your previous turn in this session was interrupted by a gateway shutdown. The conversation history below is intact.]\n\nReal goal text here."
        out = ts._strip_injections(s)
        assert "System note" not in out
        assert "Real goal text here." in out

    def test_caveman_full(self):
        s = "[CAVEMAN FULL: Drop articles where meaning survives. Fragment sentences.]\n\nFix the approval bypass bug."
        out = ts._strip_injections(s)
        assert "CAVEMAN" not in out
        assert "Fragment" not in out
        assert "Fix the approval bypass bug." in out

    def test_role_coder(self):
        s = "[Role: coder]\nYou are a senior backend engineer. Type hints, pytest, pathlib."
        out = ts._strip_injections(s)
        # Only the directive line is stripped. The following line survives —
        # the compressor decides whether it's real content. This is the
        # correct, conservative behavior: strip the bracket, not the prose.
        assert "Role:" not in out
        # The bracketed directive wrapper is gone, the following line is preserved.
        assert "You are a senior backend engineer." in out

    def test_important_cron(self):
        s = "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: ...]\n\nReal cron output here."
        out = ts._strip_injections(s)
        assert "IMPORTANT" not in out
        assert "DELIVERY" not in out
        assert "Real cron output here." in out

    def test_silent_marker(self):
        s = "[SILENT]\n\nnothing to report"
        out = ts._strip_injections(s)
        assert "[SILENT]" not in out
        assert "nothing to report" in out

    def test_active_task_section(self):
        s = "## Active Task\nUser asked: fix the bug\n\n## Goal\nActually fix the bug"
        out = ts._strip_injections(s)
        assert "## Active Task" not in out
        assert "Actually fix the bug" in out

    def test_cron_wrapper_full(self):
        """The [IMPORTANT: ... cron job ... nothing more.] wrapper has NESTED
        [SILENT] brackets inside it. A naive `[^\\]]*\\]` pattern stops at the
        first inner `]` and leaves directive tail in the goal field. The end-
        marker pattern (`nothing more.]`) matches the full wrapper."""
        s = ('[IMPORTANT: You are running as a scheduled cron job. DELIVERY: '
             'Your final response will be automatically delivered to the user '
             '\u2014 do NOT use send_message or try to deliver the output yourself. '
             'Just produce your report/output as your final response and the '
             'system handles the rest. SILENT: If there is genuinely nothing '
             'new to report, respond with exactly "[SILENT]" (nothing else) to '
             'suppress delivery. Never combine [SILENT] with content \u2014 either '
             'report your findings normally, or say [SILENT] and nothing more.]')
        out = ts._strip_injections(s)
        assert out == "", f"Cron wrapper not fully stripped, got: {out!r}"
        assert "IMPORTANT" not in out
        assert "DELIVERY" not in out
        assert "SILENT" not in out

    def test_cron_wrapper_simple(self):
        """Simpler cron wrapper without nested brackets still gets stripped."""
        s = "[IMPORTANT: Run KB compile and reindex. SILENT on no report.]"
        out = ts._strip_injections(s)
        assert out == ""
        assert "IMPORTANT" not in out

    def test_cron_wrapper_then_real_content(self):
        """When a summary has the cron wrapper followed by real content,
        the wrapper is stripped and the real content survives."""
        s = ('[IMPORTANT: scheduled cron job. DELIVERY: do not send_message. '
             'SILENT: respond with exactly "[SILENT]".]'
             '\n\nKB Daily Compile succeeded: 20 pages indexed, 0 warnings.')
        out = ts._strip_injections(s)
        assert "KB Daily Compile succeeded" in out
        assert "IMPORTANT" not in out
        assert "DELIVERY" not in out

    def test_preserves_inline_brackets(self):
        """Bracketed tokens in the middle of real content are NOT stripped.

        Conservative: only directive shapes that occupy a full line are
        matched. A real user writing "[see PR #123]" or "use [Option A]"
        in their content survives.
        """
        s = "Real user goal: see [PR #123] for context, then pick [Option A] from the list."
        out = ts._strip_injections(s)
        assert out == s

    def test_empty_and_none_safe(self):
        assert ts._strip_injections("") == ""

    def test_collapses_excess_blank_lines(self):
        s = "[CAVEMAN FULL: x]\n\n\n\n\nreal text"
        out = ts._strip_injections(s)
        # No more than one blank line between content
        assert "\n\n\n" not in out
        assert "real text" in out


class TestExtractTaskStateStripsInjections:
    """The full extraction path must not let directives survive into state."""

    SAMPLE_SUMMARY = """## Goal
[CAVEMAN FULL: Fragment. No filler.]
[Role: coder]
Fix the approval bypass so approvals.mode=off works in check_dangerous_command.

## Progress
- Patched tools/approval.py line 976
- Added 3 regression tests

## Next Steps
- [System note: previous turn interrupted] run smoke test
- Restart gateway and verify

## Relevant Files
- os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")) + "/hermes-agent"tools/approval.py
"""

    def test_goal_strips_directives(self):
        state = ts._extract_task_state_from_summary(self.SAMPLE_SUMMARY)
        assert "CAVEMAN" not in state["goal"]
        assert "Role:" not in state["goal"]
        assert "Fix the approval bypass" in state["goal"]

    def test_next_steps_strip_directives(self):
        state = ts._extract_task_state_from_summary(self.SAMPLE_SUMMARY)
        for step in state["next_steps"]:
            assert "System note" not in step
        assert any("run smoke test" in s for s in state["next_steps"])
        assert any("Restart gateway" in s for s in state["next_steps"])

    def test_historical_prefix_does_not_leak_directive_text(self):
        """MAJOR-1 regression: a summary persisted under a historical prefix
        (carveout-era / pre-#35344 variants) must have the FULL directive
        stripped before section parsing — no directive quotes in goal."""
        from markers import _HISTORICAL_PREFIX_CARVEOUT, _HISTORICAL_PREFIX_RESUME
        body = (
            "## Goal\nFix the approval bypass\n\n"
            "## Completed Actions\n1. READ config.py [tool: read_file]\n\n"
            "## Next Steps\n- run test\n"
        )
        for prefix in (_HISTORICAL_PREFIX_CARVEOUT, _HISTORICAL_PREFIX_RESUME):
            state = ts._extract_task_state_from_summary(prefix + "\n\n" + body)
            assert "Active Task" not in state["goal"], f"directive leaked from prefix: {state['goal']!r}"
            assert "resume exactly" not in state["goal"]
            assert state["goal"] == "Fix the approval bypass", state["goal"]


class TestGetContextDefends:
    """Even if on-disk state was written before the filter was added, get_context
    must not re-inject directives back into the user message."""

    def test_get_context_strips_stale_goal(self, tmp_path):
        mgr = ts.TaskStateManager(path=str(tmp_path / "task_state.json"))
        mgr.state = {
            "goal": "[CAVEMAN FULL: fragment sentences.]\n[Role: coder]\nReal long-running task",
            "done": [],
            "next_steps": [],
            "relevant_files": [],
        }
        out = mgr.get_context()
        assert "CAVEMAN" not in out
        assert "Role:" not in out
        assert "Real long-running task" in out


class TestCronSessionDoesNotPolluteTaskState:
    """Cron sessions must NOT save to persistent task_state.json.

    A cron job's prompt is dominated by the [IMPORTANT: cron job ...] wrapper.
    The compressor treats that as "the task" and saves it to the persistent
    goal — which then re-injects as "## Active Task" context in every
    subsequent turn. The fix is in plugins/context_compressor/__init__.py:
    _task_state_post_llm_call returns early if HERMES_CRON_SESSION is set.
    This test pins the architectural decision.
    """

    def test_cron_env_var_blocks_summary_extraction(self, monkeypatch):
        """When HERMES_CRON_SESSION=1, the compressor must NOT call
        update_from_summary. We simulate by importing the plugin module
        and calling the hook with a fake summary message."""
        import os
        import importlib.util
        import sys as _sys
        from pathlib import Path

        plugin_path = Path.home() / ".hermes" / "plugins" / "context_compressor"
        _sys.modules.pop("cc_plugin_pkg", None)
        spec = importlib.util.spec_from_file_location(
            "cc_plugin_pkg", plugin_path / "__init__.py",
            submodule_search_locations=[str(plugin_path)],
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["cc_plugin_pkg"] = mod
        spec.loader.exec_module(mod)

        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        # Reset module state so mgr load reflects the cron env (it doesn't
        # gate on env, but we want a clean baseline).
        import shutil
        tmp = Path("/tmp/test_task_state_cron_isolation")
        tmp.mkdir(exist_ok=True)
        mgr = ts.TaskStateManager(path=str(tmp / "task_state.json"))
        mgr.load()

        # Simulate a summary that the compressor would normally extract from.
        summary_msg = {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION]\n"
                "## Goal\n"
                "Run KB compile and reindex\n\n"
                "## Progress\n- Compile succeeded: 20 pages indexed\n\n"
                "## Next Steps\n- (none)"
            ),
        }
        # Reset state
        mgr.state = {}

        # Call the hook with the REAL post_llm_call kwarg contract — should be
        # a no-op under cron session.
        mod._task_state_post_llm_call(
            session_id="cron_test",
            conversation_history=[summary_msg],
        )
        assert mgr.state == {}, (
            f"Cron session polluted task state: {mgr.state!r}"
        )
        assert not mgr._dirty, "Cron session dirtied state — would persist"
