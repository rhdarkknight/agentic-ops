"""
Caveman Enforcer Plugin v2
==========================

Merge-safe output compression with a first-class slash command.

State lives in the plugin directory (~/.hermes/plugins/caveman_enforcer/state.json),
so /personality and config.yaml wipes never affect it.

Slash command
-------------
    /caveman          → show current mode and usage
    /caveman off      → disable compression
    /caveman lite     → light compression
    /caveman full     → full telegraphic mode
    /caveman ultra    → maximum compression

The pre_llm_call hook injects instructions on every turn based on the stored mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State management (merge-safe)
# ---------------------------------------------------------------------------

STATE_FILE = Path(__file__).with_name("state.json")

VALID_MODES = ("off", "lite", "full", "ultra")
DEFAULT_MODE = "off"

CAVEMAN_INSTRUCTIONS: Dict[str, str] = {
    "lite": (
        "[CAVEMAN LITE: Rules: "
        "1. Lead with the answer or next action. "
        "2. Number multi-step tasks. Cap lists at 5. "
        "3. End with one concrete next step. No closers. "
        "4. No preamble, no recap, no hedges, no apologies. "
        "5. Keep grammar. Drop filler words only. "
        "6. Preserve code, paths, identifiers verbatim. "
        "7. Anti-drift: these rules apply every reply.]"
    ),
    "full": (
        "[CAVEMAN FULL: Rules: "
        "1. Lead with the answer or next action. First sentence = conclusion. "
        "2. Number multi-step tasks. Cap lists at 5; split if more. "
        "3. End with one concrete next step. No 'let me know', no closers. "
        "4. No preamble, no recap, no hedges, no apologies, no tangents. "
        "5. Drop articles (a/an/the) where meaning survives. Fragments OK. "
        "6. Preserve code, paths, identifiers, error strings verbatim. "
        "No invented abbreviations (cfg/impl/fn/auth). No arrows in prose. "
        "7. State unknowns. Label estimates/opinions. "
        "Quote one decisive error line, not full dumps. "
        "8. Time estimates in minutes for multi-step tasks. "
        "9. Anti-drift: these rules apply EVERY reply, mid-tool-chain, mid-debugging. "
        "Drop caveman for security warnings, destructive ops, ambiguous sequences — resume after. "
        "Compress style, never the language.]"
    ),
    "ultra": (
        "[CAVEMAN ULTRA: Rules: "
        "1. Lead with answer. Fragments only. No sentences. "
        "2. Number steps. Cap 5. "
        "3. End with next step. No closers. "
        "4. No preamble, no recap, no hedges. "
        "5. Drop articles, prepositions, conjunctions. Symbols > words. "
        "6. Preserve code, paths, identifiers verbatim. No invented abbreviations. "
        "7. Anti-drift: every reply.]"
    ),
}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.debug("caveman: corrupt state file, resetting: %s", exc)
    return {}


def _save_state(data: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("caveman: failed to save state: %s", exc)


def _config_caveman_mode(config_text: str) -> str:
    """Read a simple display.caveman_mode mapping without PyYAML."""
    display_indent: Optional[int] = None
    for line in config_text.splitlines():
        if display_indent is None:
            match = re.match(r"^([ \t]*)display\s*:\s*(?:#.*)?$", line)
            if match:
                display_indent = len(match.group(1).expandtabs(8))
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line[: len(line) - len(line.lstrip(" \t"))].expandtabs(8))
        if indent <= display_indent:
            return ""
        match = re.match(r"^[ \t]*caveman_mode\s*:\s*(.*?)\s*$", line)
        if match:
            return match.group(1).split("#", 1)[0].strip().strip("'\"")
    return ""


def get_caveman_mode() -> str:
    """Return current mode.

    Order:
    1. State file (slash-command source of truth, merge-safe)
    2. Config `display.caveman_mode` (lets the desktop personality picker
       drive compression without going through a slash command)
    3. Legacy env var (one-time migration)
    4. off
    """
    state = _load_state()
    mode = state.get("mode", "")
    if mode in VALID_MODES:
        return mode

    # Config fallback — desktop UI writes `display.personality` for cosmetic
    # skin, but the user expects picking "caveman" to also engage compression.
    # The desktop app maps the personality to a mode in PERSONALITY_CAVEMAN_MODE
    # and writes it to `display.caveman_mode` in ~/.hermes/config.yaml. We read
    # that here so the picker actually does something.
    try:
        cfg_path = Path.home() / ".hermes" / "config.yaml"
        if cfg_path.exists():
            config_text = cfg_path.read_text(encoding="utf-8")
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(config_text) or {}
                cfg_mode = (cfg.get("display") or {}).get("caveman_mode", "")
            except Exception:
                # PyYAML is optional; user config must still work without it.
                cfg_mode = _config_caveman_mode(config_text)
            if isinstance(cfg_mode, str) and cfg_mode.lower().strip() in VALID_MODES:
                return cfg_mode.lower().strip()
    except Exception:
        pass

    # Backward-compat: read from legacy env var one time, then migrate to state file
    env_mode = os.environ.get("HERMES_CAVEMAN_MODE", "").lower().strip()
    if env_mode in VALID_MODES:
        _save_state({"mode": env_mode})
        logger.info("caveman: migrated mode '%s' from env var to state file", env_mode)
        return env_mode

    return DEFAULT_MODE


def set_caveman_mode(mode: str) -> str:
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        return f"Invalid mode: '{mode}'. Valid: {', '.join(VALID_MODES)}"
    _save_state({"mode": mode})
    return mode


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

def _handle_caveman_command(args: str) -> str:
    args = args.strip().lower()
    current = get_caveman_mode()

    if not args:
        lines = [
            "🗿 **Caveman Mode**",
            f"Current: `{current}`\n",
            "Usage:",
            "  `/caveman off`   — disable compression",
            "  `/caveman lite`  — drop filler, keep grammar",
            "  `/caveman full`  — telegraphic, fragment sentences",
            "  `/caveman ultra` — maximum compression\n",
            f"Instructions active on next message when mode is set.",
        ]
        return "\n".join(lines)

    if args in VALID_MODES:
        new_mode = set_caveman_mode(args)
        if new_mode == args:
            return f"🗿 Caveman mode set to **{args}**\n_(takes effect on next message)_"
        return new_mode  # error string

    return f"Unknown mode: `{args}`. Use `/caveman` for options."


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------

def pre_llm_call_handler(
    session_id,
    user_message,
    conversation_history,
    is_first_turn,
    model,
    platform,
    **kwargs,
):
    """Inject caveman instructions into the user message context."""
    mode = get_caveman_mode()
    if mode == "off":
        return {}

    instruction = CAVEMAN_INSTRUCTIONS.get(mode)
    if not instruction:
        return {}

    logger.debug("caveman: injecting instructions (mode=%s)", mode)
    return {"context": instruction}


# Preamble phrases the model opens with even when told to be telegraphic.
# Strip at start of response. Order matters: longest first so the
# regex doesn't eat part of a longer match.
_PREAMBLE_PATTERNS = (
    r"^(Sure|Okay|OK|Alright|Certainly|Of course|Absolutely|Got it|"
    r"Gotcha|Understood|Right|Yeah|Yes|No problem|You're welcome)[,!.]\s*",
    r"^(Let me|I'll|I will|We can|We should|Here's|This is|That is|"
    r"There is|There are|It is|Here are|These are|Those are)\s+",
    r"^(So |Well |Now |As |In )",
    r"^(Let me know if|Hope this helps|If you have any|"
    r"Feel free to|Don't hesitate|Let me know)[^.]*\.?\s*$",
)

# Trailing politeness / hand-off phrases.
_TRAILING_PATTERNS = (
    r"\n*(Let me know if.*|Hope this helps.*|If you have any.*|"
    r"Feel free to.*|Don't hesitate.*)\s*$",
)

# Phrases to collapse to terse equivalents. Order: longest first.
_PHRASE_REPLACEMENTS = (
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bwith regard to\b", "re:"),
    (r"\bwith respect to\b", "re:"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bnot later than\b", "by"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bin the process of\b", "while"),
    (r"\bthe majority of\b", "most"),
    (r"\ba number of\b", "some"),
    (r"\bthe fact that\b", "that"),
    (r"\bhas the ability to\b", "can"),
    (r"\bis able to\b", "can"),
    (r"\bare able to\b", "can"),
    (r"\bwas able to\b", "could"),
    (r"\bthere is no need to\b", "no need to"),
    (r"\bthere is no\b", "no"),
    (r"\bthere are no\b", "no"),
    (r"\bit is important to note that\b", "note:"),
    (r"\bit should be noted that\b", "note:"),
    (r"\bplease note that\b", "note:"),
    (r"\bkeep in mind that\b", "note:"),
    (r"\bas well as\b", "+"),
    (r"\bin addition to\b", "+"),
    (r"\bin addition,\s*", ""),
    (r"\badditionally,\s*", ""),
    (r"\bfurthermore,\s*", ""),
    (r"\bmoreover,\s*", ""),
    (r"\bhowever,\s*", "but "),
    (r"\bnevertheless,\s*", "but "),
    (r"\bnonetheless,\s*", "but "),
    (r"\btherefore,\s*", "so "),
    (r"\bthus,\s*", "so "),
    (r"\bhence,\s*", "so "),
    (r"\bconsequently,\s*", "so "),
    (r"\bin order that\b", "so"),
    (r"\bin conclusion,\s*", ""),
    (r"\bto summarize,\s*", ""),
    (r"\bin summary,\s*", ""),
)

# Articles + auxiliaries to drop (full mode only). Order matters.
_ARTICLE_DROP = re.compile(
    # Drop articles anywhere, including sentence-initial. Full-mode spec is
    # "drop articles where meaning survives" — a leading article drops cleanly
    # in telegraphic style and never truncates the sentence. \b anchors keep
    # us from clipping inside longer words (e.g. "that" != "thatch"), and the
    # (?!') negative lookahead preserves contractions ("it's", "that's",
    # "the'd") so we never leave a dangling "'s" or "'d".
    r"\b(a|an|the|that|this|these|those|it|its|some|any)\b(?!')\s*",
    re.IGNORECASE,
)

# Strip leading/trailing whitespace on each line.
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_FENCED_CODE_RE = re.compile(
    r"(?ms)^[ \t]*```[^\n]*(?:\n|$).*?^[ \t]*```[^\n]*(?=\n|$)"
)
_PROTECTED_TOKEN_RE = re.compile(
    r"```[\s\S]*?```"           # fenced code block
    r"|`[^`\n]+`"               # inline code
    r"|\[[^\]]+\]\([^)]+\)"     # markdown link
    r"|https?://\S+"            # URL
    r"|(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}>]+"  # file path
)


def _protect_fenced_code(text: str) -> tuple[str, dict[str, str]]:
    """Replace fenced blocks with collision-free sentinels during prose edits."""
    prefix = "\ue000CAVEMAN_FENCE_"
    while prefix in text:
        prefix += "X"
    blocks: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        marker = f"{prefix}{len(blocks)}\ue001"
        blocks[marker] = match.group(0)
        return marker

    return _FENCED_CODE_RE.sub(replace, text), blocks


def _compress_prose_segment(seg: str, mode: str) -> str:
    """Compress a single prose segment (no code, no markup).

    Preserves meaning by collapsing verbose phrases, dropping articles
    (full/ultra mode), and tightening punctuation. No length cap—
    real compression, not truncation.
    """
    if mode == "lite":
        # Lite: phrase replacements only, keep grammar.
        for pat, repl in _PHRASE_REPLACEMENTS:
            seg = re.sub(pat, repl, seg, flags=re.IGNORECASE)
        return seg

    # full / ultra: phrase replacements + article drop.
    for pat, repl in _PHRASE_REPLACEMENTS:
        seg = re.sub(pat, repl, seg, flags=re.IGNORECASE)

    if mode == "full":
        # Drop articles on a second pass so phrase replacements
        # like "the fact that" → "that" don't double-fire.
        seg = _ARTICLE_DROP.sub("", seg)

    if mode == "ultra":
        # Ultra: also drop common auxiliaries + conjunctions.
        seg = re.sub(
            r"\b(is|are|was|were|be|been|being|will|would|should|could|"
            r"may|might|must|shall|do|does|did|have|has|had)\b\s*",
            "",
            seg,
            flags=re.IGNORECASE,
        )
        # Collapse double spaces aggressively.
        seg = re.sub(r"\s+", " ", seg)
        # Drop leading articles again (some survive the aux pass).
        seg = _ARTICLE_DROP.sub("", seg)

    # Final whitespace cleanup.
    seg = re.sub(r"\s+", " ", seg)
    seg = _TRAILING_WS.sub("", seg)
    return seg.strip()


# Split a single line that contains multiple sentences into one sentence per line.
# Preserves code blocks / inline code / URLs (passed through as protected tokens).
# Conservative: only splits on `. `, `! `, `? ` followed by capital letter or end-of-line.
# Doesn't split inside code, links, or paths.
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\'])"
)


def _fragment_lines(text: str, mode: str) -> str:
    """Enforce one-sentence-per-line in full/ultra mode.

    - lite: no-op (keep paragraph style)
    - full: one sentence per line, sentences joined by newline
    - ultra: one sentence per line + break on commas (sub-fragments)
    """
    if mode not in ("full", "ultra"):
        return text
    if not text.strip():
        return text

    # Walk line-by-line. For each non-empty line, split on sentence boundaries
    # into separate lines. Preserve protected tokens (code/URLs/paths).
    out_lines: list[str] = []
    in_fenced_code = False
    for line in text.split("\n"):
        is_fence_line = bool(re.match(r"^\s*```", line))
        if in_fenced_code:
            out_lines.append(line)
            if is_fence_line:
                in_fenced_code = False
            continue
        if is_fence_line:
            out_lines.append(line)
            in_fenced_code = True
            continue
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        # Skip lines that are pure protected (code block, heading, list marker only)
        if re.match(r"^\s*([-*+]\s+|\d+\.\s+|#+\s+|>\s+|```)", line):
            out_lines.append(line)
            continue
        # If line already has multiple sentences, split.
        if _SENT_SPLIT_RE.search(stripped):
            parts = _SENT_SPLIT_RE.split(stripped)
            for p in parts:
                p = p.strip()
                if p:
                    out_lines.append(p)
        else:
            out_lines.append(stripped)
        if mode == "ultra":
            # ultra: also break on commas if line is long (>40 chars)
            expanded = []
            for ln in out_lines[-1:]:
                if len(ln) > 40 and "," in ln and not re.match(r"^\s*([-*+]\s+|\d+\.\s+|#+\s+|>\s+|```)", ln):
                    expanded.extend([s.strip() for s in ln.split(",") if s.strip()])
                else:
                    expanded.append(ln)
            out_lines = out_lines[:-1] + expanded
    return "\n".join(out_lines)


def _strip_response(text: str, mode: str) -> str:
    """Post-LLM compressor. Strips preamble, trailing politeness, AND
    compresses prose to telegraphic style (full/ultra).

    **Preserves verbatim**:
      - Fenced code blocks (```...```)
      - Inline code (`...`)
      - URLs (http://, https://, file paths like /abs/path or ~/path)
      - Markdown links [text](url) — keeps both parts
      - Headings, list markers, blockquotes

    No length cap—caveman means STYLE, not information loss.
    """
    if mode == "off":
        return text
    import re

    # Replace entire fenced blocks before preamble, politeness, and per-line
    # prose transforms; restore them verbatim after those transforms finish.
    text, fenced_code_blocks = _protect_fenced_code(text)

    # 1) Strip leading preambles line by line.
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in _PREAMBLE_PATTERNS):
            start = i + 1
            continue
        break
    lines = lines[start:]
    text = "\n".join(lines)

    # 2) Trailing politeness.
    for p in _TRAILING_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 3) For each line, compress prose around inline protected tokens.
    # Fenced blocks remain sentinels until the final verbatim restoration.
    out_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue

        # Find protected tokens within this single line.
        line_parts = []
        last = 0
        for m in _PROTECTED_TOKEN_RE.finditer(line):
            if m.start() > last:
                line_parts.append(("prose", line[last:m.start()]))
            line_parts.append(("protected", m.group(0)))
            last = m.end()
        if last < len(line):
            line_parts.append(("prose", line[last:]))

        # If no protected tokens, compress the whole line as one unit.
        if not any(k == "protected" for k, _ in line_parts):
            out_lines.append(_compress_prose_segment(line, mode))
            continue

        # Preserve structural markers (list/heading/blockquote).
        marker_match = re.match(
            r"^(\s*[-*+]\s+|\s*\d+\.\s+|\s*#+\s+|\s*>\s+)(.*)$", line
        )
        if marker_match:
            marker = marker_match.group(1)
            # Re-tokenize the body after the marker.
            body = marker_match.group(2)
            body_parts = []
            last = 0
            for m in _PROTECTED_TOKEN_RE.finditer(body):
                if m.start() > last:
                    body_parts.append(("prose", body[last:m.start()]))
                body_parts.append(("protected", m.group(0)))
                last = m.end()
            if last < len(body):
                body_parts.append(("prose", body[last:]))
            # Compress prose segments, join with spaces.
            body_out = []
            for kind, chunk in body_parts:
                if kind == "protected":
                    body_out.append(chunk)
                else:
                    body_out.append(_compress_prose_segment(chunk, mode))
            out_lines.append(marker + " ".join(body_out).strip())
            continue

        # Compress prose segments, join with spaces.
        line_out = []
        for kind, chunk in line_parts:
            if kind == "protected":
                line_out.append(chunk)
            else:
                line_out.append(_compress_prose_segment(chunk, mode))
        out_lines.append(" ".join(line_out).strip())

    result = "\n".join(out_lines).strip()
    for marker, fenced_code in fenced_code_blocks.items():
        result = result.replace(marker, fenced_code)
    return result


# (sentence-fragment enforcement added below; see post_llm_call_handler)


def post_llm_call_handler(
    response_text,
    session_id,
    model,
    platform,
    **kwargs,
):
    """Strip preamble + trailing politeness + cap length by mode.

    Returns the (possibly compressed) string. The transform_llm_output
    hook contract picks the first non-empty string return; this
    function is also safe to register against post_llm_call (fire-and-
    forget audit log).
    """
    mode = get_caveman_mode()
    if mode == "off" or not response_text:
        return None
    compressed = _strip_response(response_text, mode)
    if mode in ("full", "ultra"):
        compressed = _fragment_lines(compressed, mode)
    if compressed != response_text:
        logger.debug(
            "caveman: stripped %d -> %d chars (mode=%s)",
            len(response_text), len(compressed), mode,
        )
    return compressed


def register(ctx):
    ctx.register_command("caveman", _handle_caveman_command, description="Toggle output compression (off/lite/full/ultra)")
    ctx.register_hook("pre_llm_call", pre_llm_call_handler)
    # transform_llm_output is the hook that actually mutates the response.
    # post_llm_call is fire-and-forget; ignored here.
    ctx.register_hook("transform_llm_output", post_llm_call_handler)
    logger.info("caveman_enforcer v2 registered: /caveman command + pre_llm_call + transform_llm_output hooks")
