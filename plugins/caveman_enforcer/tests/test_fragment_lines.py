"""Tests for caveman plugin _fragment_lines (one-sentence-per-line enforcement)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin import _fragment_lines  # type: ignore  # noqa: E402


def test_full_mode_splits_multi_sentence_line():
    text = "Plugin doesn't fire. Mode is quiet. Check state.json."
    out = _fragment_lines(text, "full")
    lines = [l for l in out.split("\n") if l.strip()]
    assert len(lines) == 3
    assert lines[0] == "Plugin doesn't fire."
    assert lines[1] == "Mode is quiet."
    assert lines[2] == "Check state.json."


def test_full_mode_preserves_single_sentence_line():
    text = "Plugin doesn't fire."
    out = _fragment_lines(text, "full")
    assert out == "Plugin doesn't fire."


def test_full_mode_preserves_list_markers():
    text = "- Plugin doesn't fire\n- Mode is quiet\n- Check state.json"
    out = _fragment_lines(text, "full")
    assert "- Plugin doesn't fire" in out
    assert "- Mode is quiet" in out
    assert "- Check state.json" in out
    # Output should equal input — list lines are passed through unchanged.
    assert out == text


def test_full_mode_preserves_code_blocks():
    text = "```python\nprint('hello. world')\n```"
    out = _fragment_lines(text, "full")
    # Code block should pass through unchanged
    assert "```python" in out
    assert "print('hello. world')" in out


def test_full_mode_preserves_headings():
    text = "## Section title\nA sentence here. Another one."
    out = _fragment_lines(text, "full")
    assert "## Section title" in out


def test_full_mode_preserves_blank_lines():
    text = "Para one. With two sentences.\n\nPara two. Also two."
    out = _fragment_lines(text, "full")
    assert "\n\n" in out  # blank between paragraphs preserved


def test_ultra_mode_splits_on_commas_too():
    text = "Plugin doesn't fire because the hook never registered, and mode is quiet, and state.json says quiet, and nobody told the loop"
    out = _fragment_lines(text, "ultra")
    # ultra: split on sentence boundaries AND on commas if line is long
    lines = [l for l in out.split("\n") if l.strip()]
    # Should have more lines than the full-mode equivalent
    assert len(lines) >= 2


def test_ultra_mode_no_op_on_short_line():
    text = "Plugin off."
    out = _fragment_lines(text, "ultra")
    assert out == "Plugin off."


def test_lite_mode_does_not_split():
    text = "Para one. With two sentences. Para two. Also two."
    out = _fragment_lines(text, "lite")
    # lite should pass through unchanged
    assert out == text


def test_off_mode_does_not_split():
    text = "Para one. With two sentences. Para two. Also two."
    out = _fragment_lines(text, "off")
    assert out == text


def test_full_mode_handles_quote_starts():
    text = 'He said "hello". Then he left.'
    out = _fragment_lines(text, "full")
    lines = [l for l in out.split("\n") if l.strip()]
    assert len(lines) == 2


def test_full_mode_preserves_decimal_numbers():
    text = "Version 1.2.3 is current. Use that."
    out = _fragment_lines(text, "full")
    # Should NOT split on "1.2.3" — only on sentence boundaries
    # The "1.2" in 1.2.3 is followed by ".3" not a capital, so won't split
    # The "3 is" is followed by ". U" so WILL split
    lines = [l for l in out.split("\n") if l.strip()]
    assert "Version 1.2.3 is current." in lines
    assert "Use that." in lines


def test_full_mode_preserves_exclamation_and_question():
    text = "Done! Now what? Next step here."
    out = _fragment_lines(text, "full")
    lines = [l for l in out.split("\n") if l.strip()]
    assert len(lines) == 3


def test_full_mode_preserves_inline_code():
    text = "Run `pip install foo`. Then `pip install bar`."
    out = _fragment_lines(text, "full")
    # Should split on `. \`` patterns
    lines = [l for l in out.split("\n") if l.strip()]
    assert any("`pip install foo`" in l for l in lines)
    assert any("`pip install bar`" in l for l in lines)


def test_full_mode_handles_mixed_content():
    text = "Investigation done. Plugin registered at 22:22.\n\n- bullet one\n- bullet two"
    out = _fragment_lines(text, "full")
    lines = [l for l in out.split("\n") if l.strip()]
    assert "Investigation done." in lines
    assert "Plugin registered at 22:22." in lines
    assert "- bullet one" in lines
    assert "- bullet two" in lines


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
