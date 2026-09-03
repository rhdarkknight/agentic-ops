"""Tests for the silent_build_enforcer plugin v0.2.

Covers the user-installed plugin at ``~/.hermes/plugins/silent_build_enforcer/``:
  * Mode state management (default, set, persistence, env-var migration)
  * Slash command handler (/silent off/on/auto/quiet)
  * ``transform_llm_output`` hook — KEEP patterns deliver verbatim
  * ``transform_llm_output`` hook — narrative gets suppressed to ""
  * Narration override: bullet-pointed progress that matches a KEEP
    pattern is still suppressed if it starts with a narration phrase
  * Suppression log: suppressed text lands in suppressed.log
  * ``quiet`` mode: only gates + errors pass through, closeouts suppressed
  * Mode gating: off=noop, on=suppress, auto respects platform set
  * Empty responses are no-ops
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "silent_build_enforcer"


def _load_plugin():
    """Import the silent_build_enforcer plugin module directly from ~/.hermes."""
    spec = importlib.util.spec_from_file_location(
        "silent_build_enforcer_under_test",
        PLUGIN_DIR / "plugin.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load plugin spec from {PLUGIN_DIR / 'plugin.py'}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin()


@pytest.fixture
def plugin_with_state(tmp_path, monkeypatch):
    """Yield (plugin_module, state_file_path) with STATE_FILE + log isolated."""
    plugin = _load_plugin()
    fake_state = tmp_path / "state.json"
    fake_log = tmp_path / "suppressed.log"
    monkeypatch.setattr(plugin, "STATE_FILE", fake_state)
    monkeypatch.setattr(plugin, "SUPPRESSED_LOG", fake_log)
    # Isolate HOME so no real config can leak in.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return plugin, fake_state, fake_log


# ---------------------------------------------------------------------------
# Mode state management
# ---------------------------------------------------------------------------

class TestModeManagement:
    def test_default_mode_is_on(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        assert plugin.get_mode() == "on"

    @pytest.mark.parametrize("mode", ["off", "on", "auto", "quiet"])
    def test_set_and_get_mode(self, plugin_with_state, mode):
        plugin, _, _ = plugin_with_state
        plugin.set_mode(mode)
        assert plugin.get_mode() == mode

    def test_invalid_mode_rejected(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        result = plugin.set_mode("invalid")
        assert "Invalid" in result

    def test_state_persists_across_loads(self, plugin_with_state):
        plugin, state_file, _ = plugin_with_state
        plugin.set_mode("auto")
        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded["mode"] == "auto"

    def test_env_var_migrates_to_state(self, plugin_with_state, monkeypatch):
        plugin, state_file, _ = plugin_with_state
        monkeypatch.setenv("HERMES_SILENT_BUILD", "off")
        assert plugin.get_mode() == "off"
        assert json.loads(state_file.read_text())["mode"] == "off"


# ---------------------------------------------------------------------------
# KEEP / SUPPRESS heuristics
# ---------------------------------------------------------------------------

class TestKeepResponse:
    @pytest.mark.parametrize(
        "text",
        [
            "## Summary\nIt worked.",
            "## Status\nGREEN",
            "## Result\n- foo\n- bar",
            "## Next steps\n- run tests",
            "```bash\nls -la\n```",
            "Run `pip install foo`",
            "MEDIA:/tmp/report.pdf",
            "Which one do you prefer?",
            "Please provide your SSH key.",
            "Do you want me to deploy now?",
            "I need your input on the config.",
            "Need clarification on the cron schedule.",
            "Traceback (most recent call last):\n  ...",
            "Error 503: service unavailable",
            "errno = EAGAIN",
            "Option A: keep current\nOption B: switch",
            "Saved to /tmp/output.json",
            "✅ Done",
            "🚫 Blocked",
            "Done.",
            "Done — see output above.",
        ],
    )
    def test_keep_patterns(self, plugin, text):
        """KEEP is union-semantic in 'on' mode: any pattern matching is
        enough to deliver the response."""
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True, f"expected KEEP for: {text!r}"
        assert reason is not None, f"expected named reason for: {text!r}"


class TestNarrationOverride:
    """v0.2 fix: bullet-pointed progress reports must NOT slip through
    just because they happen to match the 'bullet' KEEP pattern. The
    narration layer is checked first."""

    @pytest.mark.parametrize(
        "text",
        [
            "Let me check the config first.\n- reading line 1\n- reading line 2",
            "Building the project now.\n- compiling\n- linking",
            "Found it. Here's what I see:\n- bug on line 42\n- bug on line 87",
            "Step 1: read the file\nStep 2: patch it\nStep 3: run tests",
            "Plan:\n1. read config\n2. patch it\n3. run tests",
            "Working on it. One moment.",
            "Sure, let me check that for you.",
            "OK, I'll start by reading the file.",
            "Now I need to read the config.",
            "First, I'll read the file. Then I'll patch it.",
        ],
    )
    def test_narration_is_always_suppressed(self, plugin, text):
        keep, reason = plugin._keep_response(text, "on")
        assert keep is False, f"expected SUPPRESS for: {text!r}"
        assert reason == "narration", (
            f"expected reason='narration', got {reason!r} for: {text!r}"
        )


class TestSuppressedCases:
    @pytest.mark.parametrize(
        "text",
        [
            "Got it. Working on it now.",
            "Let me check the file first.",
            "Building the project now.",
            "Running tests.",
            "Found it. The bug is on line 42.",
            "OK, I'll start by reading the file.",
        ],
    )
    def test_narrative_is_suppressed(self, plugin, text):
        keep, reason = plugin._keep_response(text, "on")
        assert keep is False, f"expected SUPPRESS for: {text!r}"
        assert reason is None or reason == "narration"


class TestEmptyAndWhitespace:
    def test_empty_string_is_kept(self, plugin):
        keep, reason = plugin._keep_response("", "on")
        assert keep is True
        assert reason == "empty"

    def test_whitespace_only_is_kept(self, plugin):
        keep, reason = plugin._keep_response("   \n\n   ", "on")
        assert keep is True
        assert reason == "empty"


# ---------------------------------------------------------------------------
# Quiet mode
# ---------------------------------------------------------------------------

class TestQuietMode:
    """quiet mode: only gates + errors pass through. Closeouts,
    summaries, code blocks (unless error-context) are suppressed."""

    def test_closeout_suppressed_in_quiet(self, plugin):
        text = "## Summary\nEverything works."
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is False
        assert reason is None  # narration override didn't fire, but closeout blocked

    def test_bullet_suppressed_in_quiet(self, plugin):
        text = "- first thing\n- second thing"
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is False

    def test_question_kept_in_quiet(self, plugin):
        text = "Which one do you want?"
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is True
        # KEEP is union-semantic: any gate pattern matches. We just
        # need to verify it was a gate (not narration or closeout).
        assert reason in {"ask_phrase", "trailing_question", "please_ask"}

    def test_error_kept_in_quiet(self, plugin):
        text = "Traceback (most recent call last):\n  File \"x.py\""
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is True
        assert reason == "error_block"

    def test_need_input_kept_in_quiet(self, plugin):
        text = "I need your input on the cron schedule."
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is True

    def test_options_kept_in_quiet(self, plugin):
        text = "Option A: keep\nOption B: change"
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is True
        assert reason == "options_list"

    def test_narration_suppressed_in_quiet(self, plugin):
        text = "Let me check that for you."
        keep, reason = plugin._keep_response(text, "quiet")
        assert keep is False
        assert reason == "narration"


# ---------------------------------------------------------------------------
# transform_llm_output_handler — hook integration
# ---------------------------------------------------------------------------

class TestTransformHandler:
    def test_off_mode_is_noop(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("off")
        result = plugin.transform_llm_output_handler(
            response_text="Got it. Working on it now.",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # delivered verbatim

    def test_on_mode_suppresses_to_empty_string(self, plugin_with_state):
        """v0.2: suppression is empty string, not a marker. Empty
        string is what stops the gateway from sending a message."""
        plugin, _, log_file = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="Let me check that file.",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == ""  # empty string, not "..."

    def test_suppressed_text_logged(self, plugin_with_state):
        plugin, _, log_file = plugin_with_state
        plugin.set_mode("on")
        plugin.transform_llm_output_handler(
            response_text="This is a long narration block that should be logged.",
            session_id="s1", model="m", platform="telegram",
        )
        assert log_file.exists()
        content = log_file.read_text()
        assert "This is a long narration block" in content
        assert "platform=telegram" in content

    def test_on_mode_keeps_closeout(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="## Summary\nAll tests passed.",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # delivered verbatim

    def test_on_mode_keeps_code_block(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="```bash\nls -la\n```",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # delivered verbatim

    def test_on_mode_keeps_question(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="Which one do you want?",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # delivered verbatim

    def test_on_mode_suppresses_bullet_progress(self, plugin_with_state):
        """v0.2 fix: bullet-pointed progress that previously slipped
        through now gets suppressed."""
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="Let me check the config.\n- reading line 1\n- reading line 2",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == ""

    def test_quiet_mode_suppresses_closeout(self, plugin_with_state):
        """quiet mode: even a closeout summary is suppressed."""
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="## Summary\nAll tests passed.",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == ""

    def test_quiet_mode_keeps_question(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="Which one do you want?",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None

    def test_quiet_mode_keeps_error(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="Traceback (most recent call last):\n  ...",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None

    def test_auto_mode_silent_platform_suppresses(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("auto")
        for platform in ("telegram", "cli", "tui", "discord", "slack"):
            result = plugin.transform_llm_output_handler(
                response_text="Working on it now.",
                session_id="s1", model="m", platform=platform,
            )
            assert result == "", (
                f"expected suppress on {platform!r}, got {result!r}"
            )

    def test_auto_mode_non_silent_platform_delivers(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("auto")
        for platform in ("cron", "api_server", "script"):
            result = plugin.transform_llm_output_handler(
                response_text="Working on it now.",
                session_id="s1", model="m", platform=platform,
            )
            assert result is None, (
                f"expected deliver on {platform!r}, got {result!r}"
            )

    def test_unknown_platform_is_not_silent(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("auto")
        result = plugin.transform_llm_output_handler(
            response_text="Working on it now.",
            session_id="s1", model="m", platform="weird_unknown",
        )
        assert result is None

    def test_empty_platform_is_silent(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("auto")
        result = plugin.transform_llm_output_handler(
            response_text="Working on it now.",
            session_id="s1", model="m", platform=None,
        )
        assert result == ""

    def test_empty_response_is_noop(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

class TestSilentCommand:
    def test_no_args_shows_help(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        result = plugin._handle_silent_command("")
        assert "Silent Build Mode" in result
        assert "on" in result
        assert "off" in result
        assert "auto" in result
        assert "quiet" in result
        assert "suppressed.log" in result

    @pytest.mark.parametrize("mode", ["on", "off", "auto", "quiet"])
    def test_set_mode(self, plugin_with_state, mode):
        plugin, _, _ = plugin_with_state
        result = plugin._handle_silent_command(mode)
        assert mode in result.lower()
        assert plugin.get_mode() == mode

    def test_invalid_mode(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        result = plugin._handle_silent_command("bogus")
        assert "Unknown" in result or "Invalid" in result

    def test_case_insensitive(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin._handle_silent_command("OFF")
        assert plugin.get_mode() == "off"

    def test_cron_platform_never_silent(self, plugin_with_state):
        """cron platform should never be silenced, even in quiet mode."""
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        for platform in ("cron", "api_server"):
            result = plugin.transform_llm_output_handler(
                "## Summary\n- done",
                session_id="s1", model="m", platform=platform,
            )
            assert result is None, f"expected deliver on {platform!r}, got {result!r}"

    def test_cron_platform_delivers_closeout(self, plugin_with_state):
        """cron platform should deliver closeout summaries."""
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        text = "## Result\n\nAll checks passed."
        result = plugin.transform_llm_output_handler(
            text,
            session_id="s1", model="m", platform="cron",
        )
        assert result is None, f"expected deliver on cron, got {result!r}"

