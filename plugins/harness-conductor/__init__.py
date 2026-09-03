"""Bounded verification-loop conductor plugin.

This plugin implements the two hooks emitted by agent/conversation_loop.py:

- pre_exit_verify: inspect the final assistant response and decide whether
  it is good enough to return, or whether a short re-verify pass is needed.
- post_tool_batch_reflect: inspect the tool results from a completed batch
  and decide whether a reflection pass is warranted.

Decisions are deliberately narrow. The plugin only requests a loop when the
final response is empty or ends inside an unclosed code block.
Tool failures are left to the agent's normal tool-use flow; they must never
create a user-visible reflection/retry loop on their own.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Error-detection helpers (three-layer defense)
# -----------------------------------------------------------------------------

# Layer 1: PLUGIN REFLECT NUDGE blocks re-trigger the scanner on their own
# (the block text contains "tool error detected:" and "error"). Strip them
# entirely before any further error-marker matching to break the loop.
_NUDGE_BLOCK_RE = re.compile(
    r"\[PLUGIN REFLECT NUDGE[^\]]*\].*?\[/PLUGIN REFLECT NUDGE\]",
    re.DOTALL,
)


def _strip_nudge_blocks(content: str) -> str:
    """Remove all PLUGIN REFLECT NUDGE blocks from content."""
    return _NUDGE_BLOCK_RE.sub("", content)


def _is_json_success_envelope(content: str) -> bool:
    """Layer 2: if content is a JSON success envelope, return True.

    Recognizes three classes of tool success envelopes:

    1. Terminal-style: ``{"output": ..., "exit_code": 0, "error": null}``
    2. Data envelopes (read_file, search_files, web_extract, etc.):
       ``{"content": ..., "total_lines": N}`` or similar — no ``error`` field
       and no non-zero ``exit_code`` means the call succeeded regardless of
       whether the returned *content* contains the substring "error".
    3. List/object payloads that simply lack any error indicator.

    This is structurally robust against the substring-match brittleness of
    pure pattern enumeration and eliminates the read_file/search_files
    false-positive class (Biweekly Research Insights 2026-07-02).
    """
    if not content or not content.lstrip().startswith(("{", "[")):
        return False
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False

    # Class 1: terminal-style envelope with explicit error field
    if "error" in parsed:
        error_field = parsed["error"]
        error_clean = error_field in (None, "", "null", "None")
        exit_code = parsed.get("exit_code", 0)
        exit_ok = exit_code in (0, "0", 0.0, None, "")
        if error_clean and exit_ok:
            return True
        return False

    # Class 2: data envelope (read_file, search_files, etc.) — no error field
    # AND no non-zero exit_code means success regardless of content text.
    # The narrow edge case of a malformed tool result with no `error` field
    # but real error markers in content (e.g. `{"output": "Traceback...",
    # "exit_code": 0}`) is essentially impossible in practice — terminal
    # always emits an `error` field, and read_file/search_files return
    # requested file/grep content, not error output. Keeping this branch
    # permissive avoids re-introducing the Biweekly Research Insights
    # false positive (read_file content legitimately containing "error",
    # "failed", "refused" as topic words) that motivated the entire
    # three-layer defense. See adversarial review M1 for the trade-off.
    exit_code = parsed.get("exit_code")
    if exit_code is None or exit_code in (0, "0", 0.0, ""):
        return True

    # Has a non-zero exit_code but no error field — ambiguous, don't claim success
    return False


# Success-envelope string patterns (Layer 3 fallback for non-JSON content).
_SUCCESS_PATTERNS = (
    '"error": null',
    '"error": null,',
    '"error":null',
    '"error":null,',
    "'error': none",
    "'error':none",
    "'error': None",
    '"error": ""',
    '"error":""',
    "'error': ''",
    "'error':''",
)

ERROR_MARKERS = ("error", "exception", "traceback", "failed", "failure", "refused")


def _is_real_error(content: str, lowered: str, error_markers: tuple) -> bool:
    """Three-layer defense against false-positive error detection.

    Layer 1: Strip PLUGIN REFLECT NUDGE blocks (they self-trigger).
    Layer 2: If content is a valid JSON success envelope, never an error.
    Layer 3: Strip known success-envelope string patterns, then check
             for error markers only in what remains.

    Returns True only if an error marker survives all three filters.
    """
    # Layer 1: strip self-triggering nudge blocks
    stripped = _strip_nudge_blocks(content)
    stripped_lowered = stripped.lower()

    # Layer 2: structural JSON check
    if _is_json_success_envelope(stripped):
        return False

    # Layer 3: pattern-strip + substring check
    for marker in error_markers:
        if marker not in stripped_lowered:
            continue
        cleaned = stripped_lowered
        for sp in _SUCCESS_PATTERNS:
            cleaned = cleaned.replace(sp, "")
        if marker in cleaned:
            return True
    return False


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def register(ctx) -> None:
    register_hook = getattr(ctx, "register_hook", None)
    if not callable(register_hook):
        logger.warning("harness-conductor: ctx has no register_hook (%s)", type(ctx).__name__)
        return
    register_hook("pre_exit_verify", _pre_exit_verify)
    register_hook("post_tool_batch_reflect", _post_tool_batch_reflect)
    register_hook("post_api_request", _post_api_request)
    register_hook("transform_llm_output", _transform_llm_output)
    register_hook("post_tool_call", _post_tool_call)
    register_hook("on_session_reset", _on_session_reset)
    # on_session_end fires after every turn. Keep verdict state until the
    # conversation lifetime finalizer, so the next turn can consume it.
    register_hook("on_session_finalize", _on_session_finalize)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    """Return the most recent assistant message text, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _collect_tool_errors(messages: List[Dict[str, Any]]) -> List[str]:
    """Scan tool-result messages for obvious error markers.

    Scoped to tool messages that arrived AFTER the most recent user-turn
    boundary. Old tool results that false-positive-matched in earlier
    turns must not keep gating new turns (the 60-message loop we hit on
    2026-06-29 was caused by an `ifconfig`-style output containing the
    substring "failed" surviving from a prior turn).
    """
    errors: List[str] = []
    error_markers = ("error", "exception", "traceback", "failed", "failure", "refused")
    # Find the index of the last user-role message; only inspect tool
    # messages that come after it. Anything before is stale context.
    boundary = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            boundary = i
    scan_slice = messages[boundary + 1:] if boundary >= 0 else messages
    for msg in scan_slice:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        lowered = content.lower()
        if _is_real_error(content, lowered, error_markers):
            # Keep it short; the hook reason is surfaced to the user.
            snippet = content.strip().split("\n")[0][:120]
            errors.append(snippet)
    return errors


def _looks_incomplete(response_text: str) -> Optional[str]:
    """Return a reason only when a Markdown fence is left unclosed."""
    text = response_text.strip()

    # Only line-start fence runs are Markdown delimiters. Track the active
    # delimiter length so a literal ``` inside a ```` block is content, not a
    # closing fence. Closing fences must use the same delimiter character and
    # be at least as long as the opener, with no non-whitespace suffix.
    open_fence: Optional[tuple[str, int]] = None
    for line in text.splitlines():
        match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        fence, suffix = match.groups()
        if open_fence is None:
            open_fence = (fence[0], len(fence))
        elif (
            fence[0] == open_fence[0]
            and len(fence) >= open_fence[1]
            and not suffix.strip()
        ):
            open_fence = None

    if open_fence is not None:
        return "response ends inside an unclosed code block"

    return None


def _looks_high_stakes(messages: List[Dict[str, Any]]) -> bool:
    """Heuristic: is the user asking for code, config, tests, or production ops?"""
    user_text_parts: List[str] = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            user_text_parts.append(msg["content"])
    combined = "\n".join(user_text_parts).lower()
    stakes_keywords = (
        "write", "create", "edit", "patch", "fix", "test", "deploy",
        "refactor", "implement", "build", "script", "configure",
        "migration", "database", "schema", "api", "production",
        "deploy", "release", "commit", "pr ", "pull request",
    )
    return any(kw in combined for kw in stakes_keywords)


# -----------------------------------------------------------------------------
# Hook: pre_exit_verify
# -----------------------------------------------------------------------------

def _pre_exit_verify(
    response_text: str,
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Decide whether the assistant's final response needs a re-verify pass.

    Returns:
        {"approved": False, "reason": "...", "max_reverify_passes": N}
        to force a bounded retry inside the conversation loop, or
        {"approved": True} / None to allow exit.
    """
    # If the response is empty, always re-verify.
    if not response_text or not response_text.strip():
        return {
            "approved": False,
            "reason": "final response is empty",
            "max_reverify_passes": 1,
        }

    # Incomplete / hedged responses get a single re-verify pass.
    incomplete_reason = _looks_incomplete(response_text)
    if incomplete_reason:
        return {
            "approved": False,
            "reason": incomplete_reason,
            "max_reverify_passes": 2,
        }

    # Otherwise let the agent exit.
    return {"approved": True}


# -----------------------------------------------------------------------------
# Hook: post_tool_batch_reflect
# -----------------------------------------------------------------------------

def _post_tool_batch_reflect(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Decide whether the completed tool batch warrants a reflection pass.

    Returns:
        {"reflect": True, "reason": "...", "max_reflect_passes": N}
        to force a bounded reflection pass, or
        {"reflect": False} / None to continue normally.
    """
    # Deliberately do not turn a tool failure or batch size into a second
    # user-visible agent loop. A subsequent successful tool result therefore
    # cannot revive a stale error at closeout, either.
    return {"reflect": False}


# -----------------------------------------------------------------------------
# Hooks: hermes core integration. The shipped hermes build does not expose
# pre_exit_verify / post_tool_batch_reflect, so we wrap the same logic in
# post_api_request / transform_llm_output / post_tool_call.
# -----------------------------------------------------------------------------

_HARNESS_VERDICTS = {}
# Kept for compatibility with callers that clear the old per-session state.
# Tool errors no longer create or consume reflection state.
_HARNESS_TOOL_ERRORS = {}


def _session_key(kwargs: Dict[str, Any]) -> Optional[str]:
    session_id = kwargs.get("session_id")
    return str(session_id) if session_id else None


def _clear_session_state(**kwargs: Any) -> None:
    for key in ("old_session_id", "session_id", "new_session_id"):
        if session_id := kwargs.get(key):
            session_id = str(session_id)
            _HARNESS_VERDICTS.pop(session_id, None)
            _HARNESS_TOOL_ERRORS.pop(session_id, None)


def _post_api_request(**kwargs):
    """After each final LLM response, run the shape/incompleteness gate."""
    session_id = _session_key(kwargs)
    if session_id is None:
        return None
    finish_reason = kwargs.get("finish_reason") or ""
    assistant_message = kwargs.get("assistant_message")
    has_tool_calls = bool(getattr(assistant_message, "tool_calls", None) or [])
    if has_tool_calls:
        return None
    if finish_reason and finish_reason != "stop":
        return None

    if assistant_message is None:
        return None

    response_text = ""
    c = getattr(assistant_message, "content", None)
    if isinstance(c, str):
        response_text = c

    messages = []
    verdict = _pre_exit_verify(response_text, messages)
    if not isinstance(verdict, dict):
        return None

    _HARNESS_VERDICTS[session_id] = verdict
    logger.debug(
        "harness-conductor: post_api_request verdict=%s",
        verdict.get("approved"),
    )
    return None


def _transform_llm_output(**kwargs):
    """If the gate rejected the response, append a notice prompting a retry."""
    session_id = _session_key(kwargs)
    if session_id is None:
        return None
    cached = _HARNESS_VERDICTS.get(session_id)
    if not cached:
        return None
    response_text = kwargs.get("response_text") or ""

    if cached.get("approved"):
        _HARNESS_VERDICTS.pop(session_id, None)
        return None

    reason = str(cached.get("reason", ""))
    _HARNESS_VERDICTS.pop(session_id, None)
    if not reason:
        return None

    notice = (
        "\n\n---\n_[harness-conductor] response flagged: "
        + reason
        + ". Review and revise._"
    )
    if not response_text or notice not in response_text:
        return response_text + notice
    return None


def _on_session_reset(**kwargs: Any) -> None:
    _clear_session_state(**kwargs)
    return None


def _on_session_end(**kwargs: Any) -> None:
    """Compatibility no-op: on_session_end is a per-turn boundary."""
    return None


def _on_session_finalize(**kwargs: Any) -> None:
    """Clear lifecycle state once the conversation is finalized.

    Accept arbitrary finalizer metadata while only consuming known session ID
    fields through the shared cleanup helper.
    """
    _clear_session_state(**kwargs)
    return None


def _is_raw_dict_error(result: Dict[str, Any]) -> bool:
    """Classify raw dict results without treating their ``content`` as success.

    A JSON *string* with a data-envelope shape may legitimately contain the
    word "error" in file/search content. A raw dict passed to this hook has no
    tool schema to establish that guarantee, so ``{"content": "Error: ..."}``
    must remain an error instead of being promoted to a success envelope.
    """
    error_field = result.get("error")
    if "error" in result and error_field not in (None, "", "null", "None"):
        return True
    if result.get("success") is False:
        return True
    if result.get("exit_code") not in (None, "", 0, "0", 0.0):
        return True

    error_markers = ("error", "exception", "traceback", "failed", "failure", "refused")
    for field in ("content", "output", "message"):
        value = result.get(field)
        if isinstance(value, str) and _is_real_error(value, value.lower(), error_markers):
            return True
    return False


def _post_tool_call(**kwargs):
    """Observe tool-result shape without requesting reflection or retry.

    The classification is retained for safe diagnostics and regression
    coverage, but this hook intentionally returns ``None`` for both failures
    and successes. Tool-use recovery remains in the normal agent flow.
    """
    result = kwargs.get("result")
    if isinstance(result, dict):
        if _is_raw_dict_error(result):
            logger.debug("harness-conductor: observed raw dict tool error")
    elif isinstance(result, str):
        error_markers = ("error", "exception", "traceback", "failed", "failure", "refused")
        if _is_real_error(result, result.lower(), error_markers):
            logger.debug("harness-conductor: observed tool error")
    return None
