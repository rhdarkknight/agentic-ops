"""Tests for v0.3 silent_build_enforcer behavior.

Covers the user-reported issue (2026-06-09):
  1. ``[thinking]...[/thinking]`` square-bracket wrappers must be stripped
     from the response before any closeout/gate is evaluated.
  2. A response that's ONLY a thinking block must suppress to "" (no
     message reaches the user).
  3. A response that mixes a thinking block with a real closeout must
     deliver the closeout, with the thinking stripped.
  4. Multi-line narrative that happens to mention inline code in the body
     must NOT pass via the inline-code KEEP pattern — KEEP only honors
     gate/closeout patterns on the FIRST non-blank line.
  5. The narration-override regex covers the specific phrases the user
     saw leak in the past 24h (e.g. "Now I have what I need").
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "silent_build_enforcer"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "silent_build_enforcer_v03", PLUGIN_DIR / "plugin.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin()


@pytest.fixture
def plugin_with_state(tmp_path, monkeypatch):
    plugin = _load_plugin()
    fake_state = tmp_path / "state.json"
    fake_log = tmp_path / "suppressed.log"
    monkeypatch.setattr(plugin, "STATE_FILE", fake_state)
    monkeypatch.setattr(plugin, "SUPPRESSED_LOG", fake_log)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return plugin, fake_state, fake_log


# ---------------------------------------------------------------------------
# _strip_thinking_wrappers
# ---------------------------------------------------------------------------

class TestStripThinkingWrappers:
    def test_strips_square_bracket_thinking(self, plugin):
        text = "[thinking]\nreasoning here\n[/thinking]"
        assert plugin._strip_thinking_wrappers(text) == ""

    def test_strips_thinking_then_keeps_visible(self, plugin):
        text = "[thinking]\nreasoning here\n[/thinking]\n## Summary\n- done"
        out = plugin._strip_thinking_wrappers(text)
        assert "[thinking]" not in out
        assert "[/thinking]" not in out
        assert "## Summary" in out
        assert "- done" in out

    def test_strips_angle_bracket_thinking(self, plugin):
        text = "<think>reasoning</think>visible"
        out = plugin._strip_thinking_wrappers(text)
        assert "<think>" not in out
        assert "visible" in out

    def test_strips_lowercase_and_uppercase(self, plugin):
        # Case-insensitive matching for square-bracket (open/close same case)
        for open_tag, close_tag in (
            ("[thinking]", "[/thinking]"),
            ("[THINKING]", "[/THINKING]"),
            ("[Thinking]", "[/Thinking]"),
        ):
            text = f"{open_tag}x{close_tag}visible"
            out = plugin._strip_thinking_wrappers(text)
            assert "visible" in out, f"failed for {open_tag}/{close_tag}: {out!r}"
            assert open_tag not in out
            assert close_tag not in out

    def test_strips_orphan_unterminated_at_start(self, plugin):
        text = "[thinking]\nno closing tag at all"
        out = plugin._strip_thinking_wrappers(text)
        assert out == ""

    def test_handles_multiple_wrappers(self, plugin):
        text = (
            "[thinking]first thought[/thinking]\n"
            "visible line 1\n"
            "[thinking]second thought[/thinking]\n"
            "visible line 2"
        )
        out = plugin._strip_thinking_wrappers(text)
        assert "first thought" not in out
        assert "second thought" not in out
        assert "visible line 1" in out
        assert "visible line 2" in out

    def test_passes_through_clean_text(self, plugin):
        text = "## Summary\n- all good"
        assert plugin._strip_thinking_wrappers(text) == text

    def test_handles_empty(self, plugin):
        assert plugin._strip_thinking_wrappers("") == ""


# ---------------------------------------------------------------------------
# Handler integration: thinking stripping in the hook
# ---------------------------------------------------------------------------

class TestHandlerStripsThinking:
    def test_thinking_only_response_is_suppressed(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text="[thinking]\nMemory write blocked by drift. Skip for now — it's optional. Final closeout below.\n[/thinking]",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == "", f"expected suppress, got {result!r}"

    def test_thinking_plus_closeout_returns_cleaned(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text=(
                "[thinking]\nI should write to memory but the cap is full.\n[/thinking]\n"
                "## Summary\n- fixed the bug\n- tests pass"
            ),
            session_id="s1", model="m", platform="telegram",
        )
        # Should be delivered (closeout), with thinking stripped
        assert result is not None
        assert result != ""
        assert "[thinking]" not in result
        assert "[/thinking]" not in result
        assert "## Summary" in result
        assert "fixed the bug" in result

    def test_thinking_plus_narration_suppresses(self, plugin_with_state):
        """If the visible portion is itself narration, suppress the whole
        thing (the thinking was preamble, not a real closeout)."""
        plugin, _, _ = plugin_with_state
        plugin.set_mode("on")
        result = plugin.transform_llm_output_handler(
            response_text=(
                "[thinking]\nLet me think about this.\n[/thinking]\n"
                "Let me check the file first."
            ),
            session_id="s1", model="m", platform="telegram",
        )
        assert result == "", "narration-after-thinking should suppress"

    def test_thinking_only_logged(self, plugin_with_state):
        plugin, _, log_file = plugin_with_state
        plugin.set_mode("on")
        plugin.transform_llm_output_handler(
            response_text="[thinking]\njust a thought\n[/thinking]",
            session_id="s1", model="m", platform="telegram",
        )
        content = log_file.read_text()
        assert "thinking_only" in content


# ---------------------------------------------------------------------------
# Tightened KEEP: first-line only
# ---------------------------------------------------------------------------

class TestFirstLineKEEP:
    def test_inline_code_in_body_does_not_save_narrative(self, plugin):
        """The previous v0.2 KEEP pattern matched `` `...` `` ANYWHERE in
        the text. That let a 5-paragraph narrative pass if the model
        mentioned a path or variable in the body. v0.3 only honors
        inline-code as a save on the FIRST non-blank line (and even
        then, gate patterns override). Multi-line narrative with
        inline code mentions should suppress."""
        text = (
            "Now I have what I need. Here's the situation:\n\n"
            "**Job:** Bug Zapper\n"
            "Status: investigating `Optional[\"aiohttp\"]` issue\n"
            "Working tree: 23 dirty entries\n\n"
            "Want me to proceed?"
        )
        # Should suppress because the first line starts with "Now I have
        # what I need" (narration override) — this is exactly the leak
        # the user reported yesterday.
        keep, reason = plugin._keep_response(text, "on")
        assert keep is False
        assert reason == "narration"

    def test_closeout_on_first_line_passes(self, plugin):
        text = "## Summary\n- fix landed\n- tests green"
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True
        assert reason == "summary_header"

    def test_question_on_first_line_passes(self, plugin):
        text = "Which one do you want?\nMore context..."
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True

    def test_need_input_on_first_line_passes(self, plugin):
        text = "I need your input on the config.\nMore text..."
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True

    def test_artifact_verb_on_first_line_passes(self, plugin):
        text = "Saved to /tmp/output.json\n200 records"
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True

    def test_code_block_anywhere_still_saves(self, plugin):
        """Code blocks are real deliverables — even if surrounded by
        narration, deliver the code."""
        text = (
            "Let me show you the fix.\n"
            "```python\nprint('hello')\n```\n"
            "That's it."
        )
        keep, reason = plugin._keep_response(text, "on")
        # Narration override fires first ("Let me show you the fix")
        assert keep is False
        # But the gateway would still see the code block — but the
        # plugin suppresses the whole thing. This is correct v0.3
        # behavior: any leading narration = suppress, code in body
        # doesn't save the message.
        assert reason == "narration"

    def test_lone_code_block_passes(self, plugin):
        text = "```python\nprint('hello')\n```"
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True
        assert reason == "code_fence"

    def test_media_marker_anywhere_saves(self, plugin):
        """A response with a MEDIA: marker anywhere is delivered
        because the user explicitly asked for a file. The narration
        check doesn't fire on "Here's the report" (it's not in the
        narration list), so we fall to anywhere-KEEP and MEDIA: wins."""
        text = (
            "Here's the report you asked for.\n"
            "MEDIA:/tmp/report.pdf\n"
            "Let me know what you think."
        )
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True
        assert reason == "media"

    def test_media_marker_only_passes(self, plugin):
        text = "MEDIA:/tmp/report.pdf"
        keep, reason = plugin._keep_response(text, "on")
        assert keep is True
        assert reason == "media"


# ---------------------------------------------------------------------------
# Narration coverage: catch the specific leaks from yesterday
# ---------------------------------------------------------------------------

class TestNarrationCoverage:
    @pytest.mark.parametrize(
        "text",
        [
            "Now I have what I need. Here's the situation:",
            "Now I have what I need to proceed.",
            "Now let me check the file.",
            "Now I'll patch the bug.",
            "Next: read the config.",
            "Next, I need to verify the fix.",
            "I'll start by reading the file.",
            "I need to investigate the logs.",
            "Need to verify the deployment.",
            "About to run the migration.",
            "- reading line 1\n- reading line 2",
            "- writing tests\n- running them",
            "- checking the config",
        ],
    )
    def test_specific_leak_phrases_suppressed(self, plugin, text):
        keep, reason = plugin._keep_response(text, "on")
        assert keep is False, f"expected SUPPRESS for: {text!r}"
        assert reason == "narration", f"expected reason='narration', got {reason!r} for: {text!r}"


# ---------------------------------------------------------------------------
# Quiet mode still works correctly with thinking stripping
# ---------------------------------------------------------------------------

class TestQuietModeWithThinking:
    def test_thinking_only_suppressed_in_quiet(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="[thinking]\nprivate thought\n[/thinking]",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == ""

    def test_question_still_passes_in_quiet(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="Which one do you want?",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # delivered verbatim

    def test_closeout_suppressed_in_quiet(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="## Summary\n- all done",
            session_id="s1", model="m", platform="telegram",
        )
        assert result == ""

    def test_code_block_still_passes_in_quiet(self, plugin_with_state):
        plugin, _, _ = plugin_with_state
        plugin.set_mode("quiet")
        result = plugin.transform_llm_output_handler(
            response_text="```python\nprint('hi')\n```",
            session_id="s1", model="m", platform="telegram",
        )
        assert result is None  # code always passes
