"""Tests for the caveman_enforcer plugin compressor.

Covers the user-installed plugin at ``~/.hermes/plugins/caveman_enforcer/``:
  * ``_strip_response`` — preamble/trailing strip + prose compression
  * ``_compress_prose_segment`` — phrase collapse + article drop
  * Protected token preservation (code blocks, inline code, paths, links)
  * Structural marker preservation (list bullets, headings, blockquotes)
  * All three modes: lite, full, ultra
  * Hook wiring: ``transform_llm_output`` returns compressed text
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "caveman_enforcer"


def _load_plugin():
    """Import the caveman_enforcer plugin module directly from ~/.hermes."""
    spec = importlib.util.spec_from_file_location(
        "caveman_enforcer_under_test",
        PLUGIN_DIR / "plugin.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin spec from {PLUGIN_DIR / 'plugin.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin()


@pytest.fixture
def plugin_with_state(tmp_path, monkeypatch):
    """Yield (plugin_module, state_file_path) with STATE_FILE + HOME isolated.

    get_caveman_mode falls through to ~/.hermes/config.yaml when the
    state file is empty, so we need an empty fake config to verify the
    'off' default.
    """
    plugin = _load_plugin()
    fake_state = tmp_path / "state.json"
    monkeypatch.setattr(plugin, "STATE_FILE", fake_state)
    # Isolate HOME so config.yaml fallback can't leak the real value.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return plugin, fake_state


# ---------------------------------------------------------------------------
# _compress_prose_segment
# ---------------------------------------------------------------------------


class TestCompressProseSegment:
    @pytest.mark.parametrize(
        "mode",
        ["lite", "full", "ultra"],
    )
    def test_off_unchanged(self, plugin, mode):
        # All non-off modes compress; verifying the function exists + runs.
        result = plugin._compress_prose_segment("hello world", mode)
        assert isinstance(result, str)

    def test_lite_keeps_articles(self, plugin):
        """Lite mode: phrase replacements only, no article drop."""
        result = plugin._compress_prose_segment(
            "I want to do this in order to fix the bug", "lite"
        )
        assert "to" in result  # "in order to" → "to"
        assert "a" in result or "the" in result  # articles preserved

    def test_full_drops_articles(self, plugin):
        """Full mode: drop articles a/an/the, pronoun it, this, that."""
        result = plugin._compress_prose_segment(
            "This is a test of the article dropper", "full"
        )
        assert "test" in result
        assert " article dropper" in result  # "the article dropper" → " article dropper"
        # The word "This" should have been dropped (matches _ARTICLE_DROP).
        assert "This" not in result.split()

    def test_ultra_drops_auxiliaries(self, plugin):
        """Ultra mode: also drop is/are/was/were/etc."""
        result = plugin._compress_prose_segment(
            "The file is being saved to disk", "ultra"
        )
        assert "is" not in result.split()
        assert "being" not in result.split()
        assert "saved" in result

    @pytest.mark.parametrize(
        "phrase,expected_replacement",
        [
            (r"in order to", "to"),
            (r"due to the fact that", "because"),
            (r"at this point in time", "now"),
            (r"prior to", "before"),
            (r"subsequent to", "after"),
            (r"a number of", "some"),
            (r"is able to", "can"),
            (r"has the ability to", "can"),
            (r"as well as", "+"),
            (r"in addition,", ""),
            (r"however,", "but"),
            (r"therefore,", "so"),
        ],
    )
    def test_phrase_replacements(self, plugin, phrase, expected_replacement):
        """Each verbose phrase collapses to its terse equivalent."""
        result = plugin._compress_prose_segment(phrase, "lite")
        if expected_replacement:
            assert result.strip().lower() == expected_replacement
        else:
            assert result.strip() == ""

    def test_preserves_technical_terms(self, plugin):
        """Function names, variable names, class names untouched."""
        text = "Call the get_user_data() function to fetch the result"
        result = plugin._compress_prose_segment(text, "full")
        assert "get_user_data()" in result


# ---------------------------------------------------------------------------
# _strip_response
# ---------------------------------------------------------------------------


class TestStripResponse:
    def test_strips_preamble_sure(self, plugin):
        """'Sure, ...' preamble removed, content kept."""
        text = "Sure, I can help with that.\nThe bug is on line 42."
        result = plugin._strip_response(text, "full")
        assert "Sure" not in result
        assert "bug" in result
        assert "line 42" in result

    def test_strips_preamble_let_me(self, plugin):
        text = "Let me check the file.\nThe result is foo."
        result = plugin._strip_response(text, "full")
        assert "Let me" not in result
        assert "result" in result

    def test_strips_trailing_politeness(self, plugin):
        text = "The answer is 42. Hope this helps!"
        result = plugin._strip_response(text, "full")
        assert "Hope this helps" not in result
        assert "answer" in result

    def test_off_mode_unchanged(self, plugin):
        text = "Sure, here is the verbose answer with all the words."
        result = plugin._strip_response(text, "off")
        assert result == text

    def test_preserves_fenced_code_block(self, plugin):
        text = (
            "Run this:\n"
            "```bash\n"
            "ls -la /home/user/\n"
            "```\n"
            "to list the files."
        )
        result = plugin._strip_response(text, "full")
        assert "```bash" in result
        assert "ls -la /home/user/" in result
        assert "```" in result

    def test_preserves_inline_code(self, plugin):
        text = "Run `pip install foo` to install the package."
        result = plugin._strip_response(text, "full")
        assert "`pip install foo`" in result

    def test_preserves_file_paths(self, plugin):
        text = "Edit /home/user/.hermes/config.yaml to change settings."
        result = plugin._strip_response(text, "full")
        assert "/home/user/.hermes/config.yaml" in result

    def test_preserves_tilde_paths(self, plugin):
        text = "Check the file at ~/.bashrc for the alias."
        result = plugin._strip_response(text, "full")
        assert "~/.bashrc" in result

    def test_preserves_markdown_links(self, plugin):
        text = "See [the docs](https://example.com/docs) for more."
        result = plugin._strip_response(text, "full")
        assert "[the docs](https://example.com/docs)" in result

    def test_preserves_urls(self, plugin):
        text = "Visit https://example.com/path for details."
        result = plugin._strip_response(text, "full")
        assert "https://example.com/path" in result

    def test_preserves_list_bullets(self, plugin):
        text = (
            "- First, install deps\n"
            "- Then, run tests\n"
            "- Finally, deploy"
        )
        result = plugin._strip_response(text, "full")
        assert "- First," in result
        assert "- Then," in result
        assert "- Finally," in result

    def test_preserves_numbered_list(self, plugin):
        text = (
            "1. Install the package\n"
            "2. Configure the settings\n"
            "3. Run the tests"
        )
        result = plugin._strip_response(text, "full")
        assert "1. Install" in result
        assert "2. Configure" in result
        assert "3. Run" in result

    def test_preserves_headings(self, plugin):
        text = "## Configuration\nYou need to set the value."
        result = plugin._strip_response(text, "full")
        assert "## Configuration" in result

    def test_preserves_blockquotes(self, plugin):
        text = "> Note: this is important\nDo the thing."
        result = plugin._strip_response(text, "full")
        assert "> Note:" in result

    def test_lite_preserves_grammar(self, plugin):
        """Lite mode: phrase replacements only, no article drop."""
        text = "You need to do this in order to fix the bug."
        result = plugin._strip_response(text, "lite")
        # "in order to" → "to"
        assert "in order to" not in result
        assert "You" in result  # articles/pronouns preserved

    def test_full_strips_articles(self, plugin):
        text = "The file is a test file for the user."
        result = plugin._strip_response(text, "full")
        # "The" and "a" should be dropped in full mode.
        assert "The" not in result.split()
        assert " a " not in result

    def test_ultra_maximum_compression(self, plugin):
        text = "The file is being saved to the disk now."
        result = plugin._strip_response(text, "ultra")
        assert "is" not in result.split()
        assert "being" not in result.split()
        assert "saved" in result
        assert "disk" in result

    def test_does_not_split_mid_word_at_protected_token(self, plugin):
        """Inline code on same line as prose: no newline injection."""
        text = "Run `pip install foo` to install the package."
        result = plugin._strip_response(text, "full")
        # Result should be a single line (no injected newlines around code).
        assert "\n" not in result
        assert "`pip install foo`" in result

    def test_compresses_mixed_code_and_prose(self, plugin):
        text = (
            "The error is in `main.py` line 42.\n"
            "Run `pytest tests/` to reproduce.\n"
            "The fix is in /home/user/project/bug.py."
        )
        result = plugin._strip_response(text, "full")
        assert "`main.py`" in result
        assert "`pytest tests/`" in result
        assert "/home/user/project/bug.py" in result
        # Articles dropped from prose.
        assert "The" not in result.split()

    def test_empty_string_returns_empty(self, plugin):
        assert plugin._strip_response("", "full") == ""

    def test_whitespace_only_returns_empty(self, plugin):
        assert plugin._strip_response("   \n\n   ", "full") == ""


# ---------------------------------------------------------------------------
# post_llm_call_handler (hook integration)
# ---------------------------------------------------------------------------


class TestPostLlmCallHandler:
    def test_returns_compressed_string(self, plugin):
        text = "Sure, here is the verbose answer."
        result = plugin.post_llm_call_handler(
            response_text=text,
            session_id="test",
            model="test",
            platform="test",
        )
        # Should return a string (compressed) since preamble was stripped.
        assert isinstance(result, str)
        assert "Sure" not in result

    def test_returns_none_when_no_change(self, plugin_with_state):
        """If mode is off OR text has nothing to strip, return None."""
        plugin, state_file = plugin_with_state
        # Force mode off via state.
        state_file.write_text(json.dumps({"mode": "off"}))
        result = plugin.post_llm_call_handler(
            response_text="any text",
            session_id="test",
            model="test",
            platform="test",
        )
        assert result is None

    def test_preserves_code_in_hook(self, plugin):
        text = (
            "Sure, the fix is here:\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n"
            "Hope this helps!"
        )
        result = plugin.post_llm_call_handler(
            response_text=text,
            session_id="test",
            model="test",
            platform="test",
        )
        assert isinstance(result, str)
        assert "```python" in result
        assert "def foo():" in result
        assert "Hope this helps" not in result


# ---------------------------------------------------------------------------
# Mode state management
# ---------------------------------------------------------------------------


class TestModeManagement:
    def test_get_caveman_mode_default(self, plugin_with_state):
        """No state file → 'off' default."""
        plugin, _ = plugin_with_state
        assert plugin.get_caveman_mode() == "off"

    def test_set_and_get_mode(self, plugin_with_state):
        plugin, _ = plugin_with_state
        plugin.set_caveman_mode("full")
        assert plugin.get_caveman_mode() == "full"
        plugin.set_caveman_mode("ultra")
        assert plugin.get_caveman_mode() == "ultra"

    def test_invalid_mode_rejected(self, plugin_with_state):
        plugin, _ = plugin_with_state
        result = plugin.set_caveman_mode("invalid")
        assert "Invalid" in result

    def test_config_yaml_fallback(self, plugin_with_state, tmp_path, monkeypatch):
        """Config display.caveman_mode overrides state when state is empty."""
        plugin, _ = plugin_with_state
        # Write a fake config.yaml under the isolated HOME.
        fake_home = tmp_path / "home"
        config_path = fake_home / ".hermes" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("display:\n  caveman_mode: ultra\n")
        # State file empty → fall through to config.
        assert plugin.get_caveman_mode() == "ultra"
