"""Regression tests: caveman_enforcer must never truncate content."""

import pytest
import sys
from pathlib import Path

# Add plugin root to path so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from plugin import _strip_response, post_llm_call_handler


def test_preamble_stripped():
    text = "Sure, here's the answer.\n\nThe real content."
    result = _strip_response(text, "full")
    assert "Sure" not in result
    assert result == "real content."


def test_trailing_politeness_stripped():
    text = "The real content.\n\nLet me know if you need anything else."
    result = _strip_response(text, "full")
    assert "Let me know" not in result
    assert result == "real content."


@pytest.mark.parametrize("mode", ("full", "ultra"))
def test_article_drop_preserves_contractions(mode):
    """Articles drop anywhere; apostrophe contractions must remain intact."""
    result = _strip_response("It's fine. That's the one.", mode)
    assert result == "It's fine. That's one."


def test_long_content_never_truncated():
    """Critical regression: 10k char blob must survive intact."""
    text = "A" * 10_000
    result = _strip_response(text, "ultra")
    assert result == text


@pytest.mark.parametrize("mode", ("full", "ultra"))
def test_multiline_fenced_python_stays_verbatim_while_prose_compresses(mode, monkeypatch):
    code = '''```python
the_identifier = "in order to preserve this string"
this_value = "It's the exact code payload."
fence_literal = "literal ``` must not end this block"
if this_identifier:
    print(the_identifier, this_value)
```'''
    text = f"At this point in time, the report is ready.\n\n{code}\n\nThe next step is to run it."

    compressed = _strip_response(text, mode)

    assert code in compressed
    assert compressed != text
    assert "now" in compressed

    monkeypatch.setattr("plugin.get_caveman_mode", lambda: mode)
    transformed = post_llm_call_handler(text, "sid", "model", "tui")
    assert code in transformed


def test_off_mode_passes_through():
    text = "Sure.\n\nContent."
    assert _strip_response(text, "off") == text


def test_mixed_preamble_and_content():
    text = "Okay.\nAlright.\n\nActual technical content here.\n\nHope this helps."
    result = _strip_response(text, "full")
    assert "Okay" not in result
    assert "Alright" not in result
    assert "Hope this helps" not in result
    assert "Actual technical content here." in result
