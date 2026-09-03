"""
Silent Build Enforcer Plugin v0.3
=================================

Drop mid-build narration entirely (no notification). Keep only user-input
gates and closeout summaries.

v0.2 changes (in response to user feedback):
- Suppressed responses are replaced with the empty string, not a marker.
  Telegram renders an empty string as no message, so the user gets NO
  notification at all — no phone ping, no scroll item.
- Suppressed text is logged to
  ``~/.hermes/plugins/silent_build_enforcer/suppressed.log`` so the user
  can still review what got hidden if they want to.
- New "narration override" layer: if a response matches a known
  narration pattern (e.g. "Let me check...", "Building now...") it is
  always suppressed, even if it also matches a KEEP pattern. This fixes
  the v0.1 false-positive of bullet-pointed progress reports leaking
  through.
- New ``quiet`` mode: absolute silence. Only input gates (questions to
  the user) and explicit errors are delivered. Closeout summaries are
  also suppressed. Use this for long unattended builds.
- Streaming tool-result deltas are out of scope — they go through a
  different path that doesn't hit ``transform_llm_output``. Live with it.

v0.3 changes (2026-06-09, user escalation):
- Strip ``[thinking]...[/thinking]`` (square-bracket variant) and any other
  reasoning wrappers the model emits inline. The previous stripper only
  handled ``<think>...</think>`` angle-bracket variants — square-bracket
  wrappers (used by some providers when the model is given a system prompt
  that wraps reasoning in brackets) leaked through and reached the user as
  visible text. Now stripped first, so a response that is ONLY a thinking
  block becomes empty and falls into the "no visible content" branch.
- Tightened KEEP check: gate/closeout patterns are now only honored when
  they appear on the FIRST non-blank line of the response. Inline code
  spans and code fences anywhere in the body still KEEP (those are real
  deliverables the user asked for), but they no longer let a multi-line
  narrative pass just because the model happened to mention ``code`` in
  the third paragraph. The narration override is the primary filter;
  KEEP only rescues responses that LOOK like a clean closeout or gate
  on the first line.
- New ``strip_then_keep`` order: thinking blocks are stripped BEFORE
  narration/KEEP evaluation, so a "[thinking]\\n...reasoning...\\n[/thinking]\\n
  ## Summary\\n- done" sequence correctly evaluates the visible closeout
  rather than the raw thinking wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State management (merge-safe)
# ---------------------------------------------------------------------------

STATE_FILE = Path(__file__).with_name("state.json")
SUPPRESSED_LOG = Path(__file__).with_name("suppressed.log")

VALID_MODES = ("off", "on", "auto", "quiet")
DEFAULT_MODE = "on"

# Platforms where mid-build chatter is the most painful.
_DEFAULT_SILENT_PLATFORMS = frozenset({"telegram", "cli", "tui", "discord", "slack"})

# Max lines we keep in the suppressed log (rotated).
_SUPPRESSED_LOG_MAX_LINES = 2000


# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.debug("silent_build: corrupt state file, resetting: %s", exc)
    return {}


def _save_state(data: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("silent_build: failed to save state: %s", exc)


def get_mode() -> str:
    """Return current mode.

    Order:
      1. State file (slash-command source of truth, merge-safe)
      2. Env var HERMES_SILENT_BUILD (one-time migration)
      3. on (default — user wants the build to be invisible)
    """
    state = _load_state()
    mode = state.get("mode", "")
    if mode in VALID_MODES:
        return mode

    env_mode = os.environ.get("HERMES_SILENT_BUILD", "").lower().strip()
    if env_mode in VALID_MODES:
        _save_state({"mode": env_mode})
        logger.info("silent_build: migrated mode '%s' from env var to state file", env_mode)
        return env_mode

    return DEFAULT_MODE


def set_mode(mode: str) -> str:
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        return f"Invalid mode: '{mode}'. Valid: {', '.join(VALID_MODES)}"
    _save_state({"mode": mode})
    return mode


# ---------------------------------------------------------------------------
# Suppression log (so the user can still see what got hidden)
# ---------------------------------------------------------------------------

def _log_suppressed(text: str, reason: str, platform: Optional[str]) -> None:
    """Append a suppressed response to the local log file.

    The user said even a marker is too loud — but they should still be
    able to grep the log if they want to know what happened. Local file
    only, no notification.
    """
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n--- {ts} platform={platform or '?'} reason={reason} ---\n"
            f"{text.rstrip()}\n"
        )
        with open(SUPPRESSED_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
        # Rotate if the log gets large.
        try:
            line_count = sum(1 for _ in open(SUPPRESSED_LOG, "rb"))
            if line_count > _SUPPRESSED_LOG_MAX_LINES:
                # Keep the last 1000 lines.
                with open(SUPPRESSED_LOG, "rb") as f:
                    lines = f.readlines()
                with open(SUPPRESSED_LOG, "wb") as f:
                    f.writelines(lines[-1000:])
        except Exception:
            pass
    except Exception as exc:
        logger.debug("silent_build: log write failed: %s", exc)


# ---------------------------------------------------------------------------
# KEEP / SUPPRESS heuristics
# ---------------------------------------------------------------------------

# Narration patterns — these ALWAYS suppress, even if KEEP also matches.
# These are the phrases the model emits when narrating its own work.
# Order: longer phrases first so they win ties.
_NARRATION_PATTERNS = (
    r"(?i)^\s*(sure[,.]?|okay[,.]?|ok[,.]?|alright[,.]?|certainly[,.]?|of course[,.]?|absolutely[,.]?|got it[,.]?|gotcha[,.]?|understood[,.]?|right[,.]?|yeah[,.]?|yes[,.]?|no problem[,.]?)\s",
    r"(?i)^\s*(let me|i'?ll|i will|we can|we should|i'?m going to|i am going to)\s",
    r"(?i)^\s*(so,?\s+)?(first,? i|next,? i|now i|so i|then i|after that i)\b",
    r"(?i)^\s*(building|compiling|running|installing|fetching|downloading|reading|opening|loading|checking|looking at|examining|inspecting|scanning|searching|analyzing|parsing|verifying|validating|testing|deploying|pushing|committing|writing|reading)\b",
    r"(?i)^\s*(now (i|let|let'?s)|next[,:]?\s|i'?ll\s|i\s+need\s+to|need\s+to\s|about to|going to|gonna|wanna)\b",
    r"(?i)\b(found it|here'?s what|here is what|i see|i found|i noticed|i discovered|i see that|it looks like|seems like)\b",
    r"(?i)\b(working on it|on it|hang on|one moment|one sec|just a moment|hold on)\b",
    r"(?i)\b(now i have what i need)\b",
    r"(?i)^\s*plan:\s*\n\s*1\.",  # "Plan:\n1. do thing\n2. do thing"
    r"(?im)^\s*step\s+\d+[:\.\)]\s",  # "Step 1: do thing"
    r"(?im)^\s*[-*]\s+(reading|writing|running|checking|fetching|loading|opening|installing|building|compiling|deploying|testing|patching|verifying|analyzing|parsing|examining|scanning|searching|looking)\b",  # "- reading config"
)
_NARRATION_RE = [re.compile(p) for p in _NARRATION_PATTERNS]


def _is_narration(text: str) -> bool:
    """Return True if the response is the model narrating its own work."""
    # Short-circuit: narration heuristics fire on the first 200 chars
    # (model rarely narrates in the middle of a closeout).
    head = text[:200]
    for compiled in _NARRATION_RE:
        if compiled.search(head):
            return True
    return False


# Thinking/reasoning wrappers the model emits inline in content.
# Both angle-bracket and square-bracket variants — square-bracket is what
# shows up when the system prompt wraps reasoning in [thinking]...[/thinking]
# (e.g. some MiniMax M-series deployments) and that variant slipped past the
# upstream ``strip_think_blocks`` scrubber.
_THINKING_WRAPPERS = (
    # square-bracket variants (the leak path) — compiled case-insensitive
    (r"\[thinking\][\s\S]*?\[/thinking\]", "square"),
    (r"\[reasoning\][\s\S]*?\[/reasoning\]", "square"),
    (r"\[thought\][\s\S]*?\[/thought\]", "square"),
    (r"\[reflect\][\s\S]*?\[/reflect\]", "square"),
    (r"\[scratchpad\][\s\S]*?\[/scratchpad\]", "square"),
    # angle-bracket variants (already handled upstream, but defensive)
    (r"<think>[\s\S]*?</think>", "angle"),
    (r"<reasoning>[\s\S]*?</reasoning>", "angle"),
    (r"<REASONING_SCRATCHPAD>[\s\S]*?</REASONING_SCRATCHPAD>", "angle"),
    (r"<thought>[\s\S]*?</thought>", "angle"),
    # orphan unterminated open tags (model dropped the close) — strip
    # from the open tag to end of string when at a line boundary.
    (r"(?im)^\[thinking\][\s\S]*\Z", "orphan_square"),
    (r"(?im)^\[reasoning\][\s\S]*\Z", "orphan_square"),
    (r"(?im)^\s*<think>[\s\S]*\Z", "orphan_angle"),
    (r"(?im)^\s*<thinking>[\s\S]*\Z", "orphan_angle"),
    (r"(?im)^\s*<reasoning>[\s\S]*\Z", "orphan_angle"),
)
# All thinking-wrapper regexes run case-insensitive — providers and
# prompt-templates vary the casing of the tags.
_THINKING_RE = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), kind)
    for p, kind in _THINKING_WRAPPERS
]


def _strip_thinking_wrappers(text: str) -> str:
    """Remove any ``[thinking]…[/thinking]`` / ``<think>…</think>`` blocks.

    Returns the cleaned text. If the entire content was inside a wrapper
    (so the cleaned text is empty or whitespace-only), the caller
    treats that as "no visible response" — a separate code path from
    "response is narration."
    """
    if not text:
        return text
    for compiled, _kind in _THINKING_RE:
        text = compiled.sub("", text)
    # Collapse any blank-line noise left behind by removed wrappers.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip()


# Hard KEEP patterns: anything matching these is delivered verbatim.
# In ``on`` and ``auto`` mode these pass through. In ``quiet`` mode
# only the *gates* subset is honored (questions, errors, options).
#
# v0.3: split into FIRST-LINE patterns (gate/closeout headers — must appear
# at the start of a non-blank line to count) and ANYWHERE patterns
# (code blocks and MEDIA: directives — those are always real deliverables
# the user explicitly asked for, so position doesn't matter).
_FIRST_LINE_KEEP = (
    # Markdown section headers that signal structure
    (r"(?im)^#{1,4}\s+(summary|status|result|results|done|blocked|error|errors|fail|next|next\s+steps|verdict|output|recap)\b", "summary_header"),
    # Direct questions to the user
    (r"\?\s*$", "trailing_question"),
    (r"(?i)^\s*(please (provide|enter|specify|confirm|choose|pick|select|share|tell))\b", "please_ask"),
    (r"(?i)^\s*(which (one|of these|do you)|do you want|what (would you|do you|should|about))\b", "ask_phrase"),
    (r"(?i)^\s*(i )?need (your |the |a |an )?(input|info|information|clarification|confirmation|answer|response|decision)\b", "need_input"),
    (r"(?i)^\s*(clarify|clarification|missing (info|information|detail))", "clarify_phrase"),
    # Error / failure blocks the user needs to see
    (r"(?i)^\s*(traceback|exception|stack\s*trace|error\s*\d+|errno|e\d{3,4})\b", "error_block"),
    # Choices/options (front-end data gathering) — first line lists them
    (r"(?i)^\s*option\s+[a-d1-4]\b", "options_list"),
    # File artifacts (deliverables) — first line announces the action
    (r"(?i)^\s*(saved|written|created|modified|updated|patched|installed|deployed|committed|pushed) (to|at|in|on)\b", "artifact_verb"),
    # Explicit closing markers (emoji + word)
    (r"(?i)^\s*(?:✓|✗|✅|❌|🚫|🔚)\s+(done|complete|finished|blocked|fail|error|shipped)", "emoji_closeout"),
    (r"(?i)^\s*(done|complete|finished|shipped|blocked|fail(ed|ure)?|error)\s*[:\.\!]?\s*(?:[\u2014\u2013\-].*)?$", "single_word_closeout"),
)
# Anywhere-KEEP: real artifacts (code blocks, media). These are
# delivered even if surrounded by narration, because the user asked
# for the code/output.
_ANYWHERE_KEEP = (
    (r"```", "code_fence"),
    (r"`[^`\n]+`", "inline_code"),
    (r"MEDIA:[^\s]+", "media"),
)
_KEEP_PATTERNS = _FIRST_LINE_KEEP + _ANYWHERE_KEEP
_FIRST_LINE_RE = [(re.compile(p), name) for p, name in _FIRST_LINE_KEEP]
_ANYWHERE_RE = [(re.compile(p), name) for p, name in _ANYWHERE_KEEP]

# In ``quiet`` mode only these are delivered (gates + errors + code/media).
_QUIET_KEEP_SUBSET = frozenset({
    "trailing_question", "please_ask", "ask_phrase", "need_input",
    "clarify_phrase", "error_block", "options_list", "media",
    "code_fence",  # code blocks are always signal, not narration
    "inline_code",
})


def _first_nonblank_line(text: str) -> str:
    """Return the first non-blank line of ``text`` (no trailing newline)."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _keep_response(text: str, mode: str) -> Tuple[bool, Optional[str]]:
    """Return (keep?, reason?) for ``text``.

    Order of evaluation (v0.3):
      1. Empty / whitespace-only → KEEP (no-op, nothing to suppress)
      2. Narration override (first 200 chars) → SUPPRESS
      3. First-line KEEP (gate/closeout header) → KEEP
      4. Anywhere KEEP (code/media) → KEEP
      5. Otherwise → SUPPRESS
    """
    if not text or not text.strip():
        return True, "empty"

    if _is_narration(text):
        return False, "narration"

    first_line = _first_nonblank_line(text)
    for compiled, name in _FIRST_LINE_RE:
        if mode == "quiet" and name not in _QUIET_KEEP_SUBSET:
            continue
        if compiled.search(first_line):
            return True, name

    for compiled, name in _ANYWHERE_RE:
        if mode == "quiet" and name not in _QUIET_KEEP_SUBSET:
            continue
        if compiled.search(text):
            return True, name

    return False, None


def _is_silent_platform(platform: Optional[str]) -> bool:
    """Return True if the platform should run in silent mode by default.

    Strict default: only platforms explicitly in the silent set are
    silenced. Unknown non-empty platforms are NOT silenced. Empty/missing
    platform IS silenced (the "in doubt, hide it" path).
    """
    if not platform:
        return True
    return platform.lower() in _DEFAULT_SILENT_PLATFORMS


# ---------------------------------------------------------------------------
# Hook handler
# ---------------------------------------------------------------------------

def transform_llm_output_handler(
    response_text: str,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    platform: Optional[str] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Suppress mid-build narration; keep closeouts and input gates.

    v0.3: strip thinking wrappers FIRST so a response that's entirely
    ``[thinking]…[/thinking]`` is treated as empty (no visible content)
    rather than getting routed through the narration override with
    a misleading "[thinking]" header.

    Returns:
      - None   → leave the response unchanged (KEEP).
      - ""     → suppress entirely (the gateway already treats empty
                 string as "don't send" — no Telegram message is created,
                 so the user gets no notification).
      - str    → replace the response with this string (e.g. the
                 thinking-stripped text — used when the response was
                 "[thinking]\\n...reasoning...\\n[/thinking]\\n## Summary\\n- done"
                 and we want to deliver "## Summary\\n- done").
    """
    mode = get_mode()
    if mode == "off":
        return None  # no-op

    # ACP (Zed editor) should never be silenced — it's an interactive coding
    # session where the user needs to see all output, including mid-build
    # narration. The silent build enforcer is for chat platforms only.
    if platform and platform.lower() == "acp":
        return None

    # Cron and api_server should never be silenced — their output IS the
    # deliverable. The user explicitly designed this: cron jobs produce
    # reports, api_server produces API responses. Silencing them loses
    # the entire output. This applies to ALL modes (on/auto/quiet).
    if platform and platform.lower() in ("cron", "api_server"):
        return None

    if mode == "auto" and not _is_silent_platform(platform):
        return None  # auto + non-silent platform → deliver

    # v0.3: strip any [thinking]…[/thinking] / <think>…</think> wrappers
    # the model emitted inline. If stripping found any wrappers, the
    # cleaned text is what the user should see (or nothing, if the
    # response was entirely inside a wrapper).
    stripped = _strip_thinking_wrappers(response_text or "")
    if stripped != (response_text or ""):
        # Wrappers were found. If the cleaned text is empty, the
        # response was *only* thinking — no visible content to deliver.
        if not stripped.strip():
            _log_suppressed(response_text, "thinking_only", platform)
            logger.info(
                "silent_build: suppressed %d chars (thinking-only, mode=%s, platform=%s, session=%s)",
                len(response_text or ""), mode, platform, session_id,
            )
            return ""
        # Otherwise: evaluate the cleaned text against the keep/suppress
        # policy. If the visible portion passes (clean closeout, real
        # code block, etc.), deliver the CLEANED text as a replacement
        # (so the [thinking]…[/thinking] wrapper never reaches the user).
        # If the visible portion is still narration, suppress the whole
        # thing — the thinking was just preamble.
        keep, reason = _keep_response(stripped, mode)
        if keep:
            logger.debug(
                "silent_build: delivering cleaned text (keep_reason=%s, mode=%s, platform=%s)",
                reason, mode, platform,
            )
            return stripped
        # Cleaned text was narration → suppress entirely.
        _log_suppressed(response_text, reason or "narration_after_thinking_strip", platform)
        logger.info(
            "silent_build: suppressed %d chars (narration-after-thinking-strip, mode=%s, platform=%s, session=%s)",
            len(response_text or ""), mode, platform, session_id,
        )
        return ""
    # No wrappers found — original KEEP/SUPPRESS evaluation.
    keep, reason = _keep_response(response_text, mode)
    if keep:
        logger.debug(
            "silent_build: delivering (keep_reason=%s, mode=%s, platform=%s)",
            reason, mode, platform,
        )
        return None  # KEEP — let the original through

    # Suppress — return empty string. The gateway's `if _text:` check
    # skips the send entirely, so the user gets no Telegram message,
    # no phone ping, no scroll item. Suppressed text is logged locally
    # for the curious.
    _log_suppressed(response_text, reason or "unknown", platform)
    logger.info(
        "silent_build: suppressed %d chars (mode=%s, platform=%s, reason=%s, session=%s)",
        len(response_text), mode, platform, reason, session_id,
    )
    return ""


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

def _handle_silent_command(args: str) -> str:
    args = args.strip().lower()
    current = get_mode()

    if not args:
        lines = [
            "**Silent Build Mode**",
            f"Current: `{current}`",
            "",
            "Modes:",
            "  `on`   — suppress narration, keep closeouts + gates (default)",
            "  `off`  — deliver every response",
            "  `auto` — on for chat/CLI; off for cron/api",
            "  `quiet`— absolute silence; only gates + errors (no closeouts)",
            "",
            "Suppressed text is logged to:",
            "  `~/.hermes/plugins/silent_build_enforcer/suppressed.log`",
            "",
            "Platforms always-silent: telegram, cli, tui, discord, slack.",
            "Platforms never-silent: cron, api_server.",
        ]
        return "\n".join(lines)

    if args in VALID_MODES:
        new_mode = set_mode(args)
        if new_mode == args:
            label = {
                "on": "🔕 narration suppressed, closeouts + gates pass through",
                "off": "🔔 all responses delivered",
                "auto": "🔕 chat/CLI silenced, cron/api delivered",
                "quiet": "🔕🔕 absolute silence — only gates + errors",
            }.get(args, args)
            return f"Silent build mode set to **{args}** — {label}"
        return new_mode

    return f"Unknown mode: `{args}`. Use `/silent` for options."


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    ctx.register_command(
        "silent",
        _handle_silent_command,
        description="Toggle silent build mode (off/on/auto/quiet) — suppresses mid-build narration entirely",
    )
    ctx.register_hook("transform_llm_output", transform_llm_output_handler)
    logger.info(
        "silent_build_enforcer v0.3 registered: /silent command + transform_llm_output hook "
        "(default mode=%s, suppressed → empty string, log=%s; v0.3: strips [thinking] wrappers, "
        "tightens KEEP to first-line only)",
        DEFAULT_MODE, SUPPRESSED_LOG,
    )
