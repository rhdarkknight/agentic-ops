"""Compounding Loops — make plan→build→test→audit the default.

This plugin enforces the M3 looping discipline: after a non-trivial build,
the agent must run an adversarial review pass, fix findings, and re-review
until two consecutive clean review passes are observed.

Hook behaviour:

- ``pre_exit_verify``: inspect the final assistant response + message
  history. If a build happened in this turn but no review evidence is
  present, reject the exit and request a review pass. If the most recent
  review found blockers or majors, reject until they are addressed and a
  re-review confirms they are gone. Approve only after two consecutive
  clean passes (or the configured cap is hit).

- ``post_tool_batch_reflect``: if a build batch just landed but the agent
  is heading toward an exit without review, request a small reflection
  pass so the loop enters before closeout.

Gate decisions are derived from message-history evidence and the
session-scoped hint cache. When ``pre_exit_verify`` evaluates a build, it
also best-effort writes an advisory loop-state snapshot for cross-session
handoff and operational visibility. The snapshot records the latest verdict,
review/build progress, open findings, and heartbeat. It survives resets for
recovery and stuck-loop monitoring, but is never the sole basis for approval.

Configuration (environment variables):

- ``HERMES_LOOPS_ENABLED`` (default ``1``) — set to ``0`` to bypass.
- ``HERMES_LOOPS_MAX_PASSES`` (default ``3``) — oscillation cap: if a
  clean pass was seen at pass N but the loop hasn't converged after this
  many *additional* passes, ship with findings. Catches clean→dirty→clean
  oscillation where each fix round introduces a new issue.
- ``HERMES_LOOPS_STUCK_CAP`` (default ``6``) — stuck cap: if no clean pass
  has been seen after this many dirty passes, ship with findings. The
  agent is not resolving blockers/majors. Higher than MAX_PASSES to give
  the agent room to fix complex tasks before the stuck cap fires.
- ``HERMES_LOOPS_MIN_BUILD_TOOLS`` (default ``2``) — minimum write/patch/
  execute_code/terminal tool calls in the recent history to count as a
  build.
- ``HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN`` (default ``1``) — when set, two
  consecutive clean review passes are required before exit. Set ``0``
  to allow exit after a single clean pass.
- ``HERMES_LOOPS_BYPASS_KEYWORDS`` (default ``"quick,trivial,one-liner"``)
  — comma-separated keywords; if any appear in the user message, bypass
  the loop for that turn.
- ``HERMES_LOOPS_REVIEW_TOOLS`` (default ``""``) — comma-separated tool
  names that constitute a *real* review (e.g.
  ``adversarial_review,review,audit``). When set, the gate additionally
  requires at least one such tool call in the message history before
  accepting any review evidence — this converts the gate from
  narrative-police to process-police. Empty (default) preserves the
  original text-evidence behaviour.
- ``HERMES_LOOPS_HARD_FAIL`` (default ``0``) — when set, ``register``
  raises if the plugin manager provides no ``register_hook``; otherwise
  it logs a warning and silently no-ops (the historical behaviour).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    from loop_state import (
        clear_state as _clear_loop_state,
        read_state as _read_loop_state,
        write_state as _write_loop_state,
    )
except ImportError:
    # loop_state.py lives alongside this __init__.py in the plugin dir,
    # but the plugin manager's loader doesn't put the plugin dir on
    # sys.path for bare imports. Load it explicitly from __file__.
    try:
        import importlib.util as _ils
        _ls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_state.py")
        _ls_spec = _ils.spec_from_file_location("_loop_state_inline", _ls_path)
        if _ls_spec and _ls_spec.loader:
            _ls_mod = _ils.module_from_spec(_ls_spec)
            _ls_spec.loader.exec_module(_ls_mod)
            _clear_loop_state = _ls_mod.clear_state
            _write_loop_state = _ls_mod.write_state
            _read_loop_state = _ls_mod.read_state
        else:
            raise ImportError("loop_state spec creation failed")
    except Exception:
        _clear_loop_state = None  # type: ignore[assignment]
        _write_loop_state = None  # type: ignore[assignment]
        _read_loop_state = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration (re-read each call so test fixtures can flip env vars)
# -----------------------------------------------------------------------------

def _truthy(raw: str) -> bool:
    return (raw or "").lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int, lo: int = 1, hi: Optional[int] = None) -> int:
    """Parse an int env var with a safe fallback on garbage input."""
    raw = os.environ.get(name, str(default))
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    if v < lo:
        return lo
    if hi is not None and v > hi:
        return hi
    return v


def _config() -> Dict[str, Any]:
    return {
        "enabled": _truthy(os.environ.get("HERMES_LOOPS_ENABLED", "1")),
        "max_passes": _int_env("HERMES_LOOPS_MAX_PASSES", 3, lo=1, hi=9),
        "stuck_cap": _int_env("HERMES_LOOPS_STUCK_CAP", 6, lo=1),
        "min_build_tools": _int_env("HERMES_LOOPS_MIN_BUILD_TOOLS", 2, lo=1),
        "require_double_clean": _truthy(
            os.environ.get("HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN", "1")
        ),
        "bypass_keywords": tuple(
            kw.strip().lower()
            for kw in os.environ.get(
                "HERMES_LOOPS_BYPASS_KEYWORDS", "quick,trivial,one-liner"
            ).split(",")
            if kw.strip()
        ),
        "review_tools": tuple(
            t.strip().lower()
            for t in os.environ.get("HERMES_LOOPS_REVIEW_TOOLS", "").split(",")
            if t.strip()
        ),
        "hard_fail": _truthy(os.environ.get("HERMES_LOOPS_HARD_FAIL", "0")),
        # #2: step cap — total build-tool calls per session. 0 = disabled.
        "max_turns": _int_env("HERMES_LOOPS_MAX_TURNS", 0, lo=0),
        # #2: circuit breaker — trip when the same tool+args repeats N times
        # in a row. 0 = disabled.
        "circuit_breaker": _int_env("HERMES_LOOPS_CIRCUIT_BREAKER", 0, lo=0),
        # #8: proof-of-fix — when enabled, every blocker fix must carry a
        # structured proof-of-fix block in the review output proving the
        # fix was verified by revert-or-mutate.
        "proof_of_fix": _truthy(os.environ.get("HERMES_LOOPS_PROOF_OF_FIX", "0")),
    }


# -----------------------------------------------------------------------------
# Patterns for review evidence detection
# -----------------------------------------------------------------------------

# Matches "review pass 3:", "Pass 1:", "audit pass #2:", "adversarial pass 4:".
# A colon after the number is REQUIRED. This drops false positives like
# "we did 3 passes over the data" or "pass 5 of the compiler" while keeping
# every documented review-evidence form (all of which use a colon).
_PASS_RE = re.compile(
    r"(?:(?:review|audit|adversarial)[^\n]{0,20}?)?\bpass\s*#?\s*(\d+)\b\s*:",
    re.IGNORECASE,
)
# Matches "3 blockers" / "1 blocker".
_BLOCKER_RE = re.compile(r"\b(\d+)\s+blockers?\b", re.IGNORECASE)
# Matches "no blockers" / "zero blockers" / "0 blockers".
_NO_BLOCKER_RE = re.compile(r"\b(?:no|zero|0)\s+blockers?\b", re.IGNORECASE)
_MAJOR_RE = re.compile(r"\b(\d+)\s+majors?\b", re.IGNORECASE)
_NO_MAJOR_RE = re.compile(r"\b(?:no|zero|0)\s+majors?\b", re.IGNORECASE)
# Explicit clean-review language. Kept for optional logging/confirmation but
# no longer required for a pass to count as clean — "0 blockers, 0 majors"
# alone is sufficient evidence of cleanliness.
_CLEAN_RE = re.compile(
    r"\b(?:review\s+clean|no\s+findings|"
    r"(?:all\s+)?findings?\s+resolved|review\s+passed|"
    r"pass\s+complete|review\s+complete)\b",
    re.IGNORECASE,
)

# Tools that modify files. These are the only calls that genuinely constitute
# "a build happened" — read-only shell/exec calls do not.
_MUTATING_BUILD_TOOLS = frozenset({"write_file", "patch"})
# All build-class tools (mutating + exec/terminal). Used for the total count
# threshold once a mutating call has confirmed a build.
_ALL_BUILD_TOOLS = frozenset({"write_file", "patch", "execute_code", "terminal"})

# Words in the user message that suggest a non-trivial build. Tightened
# to verbs that almost always imply code (vs. natural prose that
# mentions "add"/"fix"/"write" in a discussion). Adding an entry here
# widens the gate's surface area, so the bar is high.
_BUILD_KEYWORDS = (
    "build", "implement", "refactor", "deploy", "wire up", "scaffold",
    "integrate", "migrate",
)

# #8: Proof-of-fix block parser. Matches a structured proof block the
# agent emits in its review output to prove a fix was verified by
# revert-or-mutate. Format:
#   Proof-of-fix <fix-id>:
#     test: <test name/path>
#     red-before: <description>
#     green-after: <description>
#     revert-verified: yes
# The gate requires one proof block per blocker fix when
# HERMES_LOOPS_PROOF_OF_FIX=1. `revert-verified: yes` is the key field —
# it asserts the agent reverted the fix and confirmed the test went red.
_PROOF_OF_FIX_RE = re.compile(
    r"proof-of-fix\b[^\n]{0,80}?:\s*\n"
    r"(?:\s*test:\s*(?P<test>.+?)\n)?"
    r"(?:\s*red-before:\s*(?P<red>.+?)\n)?"
    r"(?:\s*green-after:\s*(?P<green>.+?)\n)?"
    r"(?:\s*revert-verified:\s*(?P<revert>\w+))?",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def register(ctx) -> None:
    # Defensive: some plugin managers pass dicts, others PluginContext.
    register_hook = getattr(ctx, "register_hook", None)
    if not callable(register_hook):
        msg = (
            "compounding-loops: ctx has no register_hook (type=%s); "
            "plugin will not enforce loop discipline"
            % type(ctx).__name__
        )
        if _config()["hard_fail"]:
            raise RuntimeError(msg)
        logger.warning(msg)
        return
    # Newer hermes builds expose only the standard hooks (pre_llm_call,
    # post_llm_call, post_api_request, transform_llm_output, ...). Older
    # forward-future builds also expose pre_exit_verify and
    # post_tool_batch_reflect. Register every name the build knows; the
    # plugin manager silently drops unknown ones. This keeps a single
    # plugin file working across both trees.
    register_hook("pre_exit_verify", _pre_exit_verify)
    register_hook("post_tool_batch_reflect", _post_tool_batch_reflect)
    register_hook("pre_llm_call", _pre_llm_call)
    register_hook("post_llm_call", _post_llm_call)
    register_hook("post_api_request", _post_api_request)
    register_hook("transform_llm_output", _transform_llm_output)
    register_hook("on_session_reset", _on_session_reset)
    register_hook("on_session_start", _on_session_reset)
    # on_session_end runs after every turn, so it must not discard the
    # cross-turn verdict/streak state. Terminal cleanup belongs to the
    # conversation-lifetime finalizer instead.
    register_hook("on_session_finalize", _on_session_finalize)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _user_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            parts.append(msg["content"])
    return "\n".join(parts)


def _user_message(messages: List[Dict[str, Any]]) -> str:
    """Return the most recent user message text (for bypass-keyword check)."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _tool_name(tc: Dict[str, Any]) -> Optional[str]:
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or {}
    name = fn.get("name") if isinstance(fn, dict) else None
    # Fallback for dispatcher paths that normalize to tc["name"].
    if not name:
        name = tc.get("name")
    return name if isinstance(name, str) else None


def _tool_call_signature(tc: Dict[str, Any]) -> Optional[str]:
    """A stable signature for circuit-breaker detection: tool name + args.

    Two calls with the same signature are "the same tool + same args."
    Args are hashed (sha1, truncated to 16 hex chars) so:
    - different file paths / commands hash distinctly (no false-trip)
    - the signature is bounded size (no blow-up from huge payloads)
    """
    name = _tool_name(tc)
    if not name:
        return None
    fn = tc.get("function") or {}
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if args is None:
        args_str = ""
    elif isinstance(args, str):
        args_str = args
    else:
        try:
            args_str = str(args)
        except Exception:
            args_str = repr(args)
    # sha1 over the full args string gives a 40-hex digest; truncate
    # to 16 chars for compactness. Collision probability is ~1 in 2^64,
    # negligible compared to legitimate duplicate calls.
    import hashlib as _hl
    h = _hl.sha1(args_str.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{name}:{h}"


def _detect_circuit_breaker(
    messages: List[Dict[str, Any]],
    threshold: int,
) -> tuple:
    """Return (tool_name, args_snippet) if the same tool+args repeats
    ``threshold`` times in any sliding window of size ``threshold`` in
    the tool-call history, else (None, "").

    Catches runaway loops that happened *anywhere* in the recent history,
    not just at the very end — the older "look at the last N calls only"
    miss loop patterns that were recovered from but left evidence
    earlier in the session (the user's "runaway loops" complaint).

    We scan from the *end backwards* and return as soon as we find the
    most-recent streak of ``threshold`` identical signatures. This
    keeps the algorithm O(streak_size) per detection rather than
    O(history) — critical when a runaway session has thousands of
    tool calls.
    """
    if threshold <= 0:
        return (None, "")
    sigs_rev: List[tuple] = []  # (sig, tool_name, args_snippet) newest-first
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        # Walk newest-first so the resulting list is reverse-chronological.
        for tc in reversed(tool_calls):
            if not isinstance(tc, dict):
                continue
            sig = _tool_call_signature(tc)
            if sig is None:
                continue
            fn = tc.get("function") or {}
            a = fn.get("arguments") if isinstance(fn, dict) else ""
            args_snippet = a if isinstance(a, str) else (str(a) if a else "")
            sigs_rev.append((sig, _tool_name(tc) or "", args_snippet))
    if len(sigs_rev) < threshold:
        return (None, "")
    # Look for a streak of ``threshold`` identical sigs, scanning from
    # the most recent. Track the (last) tool name + args for reporting.
    for i in range(len(sigs_rev) - threshold + 1):
        window = [sigs_rev[i + j][0] for j in range(threshold)]
        if len(set(window)) == 1:
            _, name, args = sigs_rev[i]
            return (name, args)
    return (None, "")


def _write_state_snapshot(
    verdict: str,
    hints: Dict[str, Any],
    highest_pass: int,
    turn_count: int,
    review: Optional[Dict[str, Any]],
    consecutive_clean: int,
    session_id: Optional[str],
    circuit_breaker_tripped: bool = False,
) -> None:
    """Write a snapshot to the persisted loop-state file (#1).

    Best-effort: a write error is swallowed so the gate never blocks on
    a state-file write. No-op if ``loop_state`` isn't importable (standalone
    test layout).
    """
    if _write_loop_state is None:
        return
    snapshot: Dict[str, Any] = {
        "session_id": session_id or "",
        "build_count": int(hints.get("mutating_build_count", 0)),
        "review_pass_count": int(highest_pass),
        "last_review_clean": bool(review and review.get("clean")),
        "open_blockers": int(review["blockers"]) if review else 0,
        "open_majors": int(review["majors"]) if review else 0,
        "consecutive_clean": int(consecutive_clean),
        "turn_count": int(turn_count),
        "circuit_breaker_tripped": bool(circuit_breaker_tripped),
        "last_exit_verdict": verdict,
    }
    try:
        _write_loop_state(snapshot)
    except Exception as exc:  # never block the gate on a state write
        logger.debug("loop-state write failed: %s", exc)


def _count_build_tool_calls(
    messages: List[Dict[str, Any]],
    tools: "frozenset[str]" = _ALL_BUILD_TOOLS,
) -> int:
    """Count build-tool calls in recent history matching ``tools``."""
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if _tool_name(tc) in tools:
                count += 1
    return count


def _recent_mutating_signatures(
    messages: List[Dict[str, Any]],
    tools: "frozenset[str]",
    limit: int,
) -> tuple:
    """Return the last ``limit`` mutating tool-call signatures, in order.

    Each signature is ``f"{name}:{sha1(args)[:16]}"`` from
    ``_tool_call_signature``. Used by the reflect-nudge dedup so the
    check survives context compaction (counts can shrink when old
    messages are erased; signatures are intrinsic to the call and
    don't depend on position in the list).
    """
    sigs: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if _tool_name(tc) in tools:
                sig = _tool_call_signature(tc)
                if sig:
                    sigs.append(sig)
    return tuple(sigs[-limit:])


def _has_review_tool_call(
    messages: List[Dict[str, Any]],
    review_tools: "tuple[str, ...]",
) -> bool:
    """True if any assistant message invoked one of the configured review tools."""
    if not review_tools:
        return True
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            name = _tool_name(tc)
            if name and name.lower() in review_tools:
                return True
    return False


def _user_requested_build(messages: List[Dict[str, Any]]) -> bool:
    """Heuristic: did the user ask for a build-class operation?

    Checks the most recent user message only (not the whole history) so an
    earlier "let's not implement X yet" doesn't poison a later unrelated
    turn.
    """
    user_text = _user_message(messages).lower()
    if not user_text:
        return False
    return any(kw in user_text for kw in _BUILD_KEYWORDS)


def _is_build_response(messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> bool:
    """True if a non-trivial build happened recently.

    A build requires at least one *mutating* tool call (write_file/patch).
    Read-only terminal/execute_code calls do not by themselves constitute a
    build. Once a mutating call is present, the total build-tool count must
    meet ``min_build_tools`` (or the user explicitly requested a build).
    """
    mutating = _count_build_tool_calls(messages, _MUTATING_BUILD_TOOLS)
    if mutating < 1:
        return False
    total = _count_build_tool_calls(messages, _ALL_BUILD_TOOLS)
    user_intent = _user_requested_build(messages)
    return total >= cfg["min_build_tools"] or user_intent


def _bypass_requested(messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> bool:
    """True if the user explicitly asked for a quick/one-line answer."""
    user_text = _user_message(messages).lower()
    if not user_text:
        return False
    return any(kw in user_text for kw in cfg["bypass_keywords"])


def _last_count(window: str, num_re: "re.Pattern", no_re: "re.Pattern") -> int:
    """Return the count implied by the *last* blocker/major mention in
    ``window``.

    "0 blockers" matches both ``num_re`` (group=0) and ``no_re``; both yield
    0. "3 blockers" matches only ``num_re``. If mentions disagree across a
    window, the latest by position wins (correction semantics). If no
    mention is present, default 0.
    """
    if not window:
        return 0
    candidates: List[tuple] = []
    for m in num_re.finditer(window):
        try:
            candidates.append((m.end(), int(m.group(1))))
        except (TypeError, ValueError):
            continue
    for m in no_re.finditer(window):
        candidates.append((m.end(), 0))
    if not candidates:
        return 0
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _extract_all_reviews_from_text(
    response_text: str,
) -> List[Dict[str, Any]]:
    """Parse every review pass mentioned in a text blob.

    Returns a list (chronological by mention position) of dicts:
        {
          "pass": int,
          "blockers": int,
          "majors": int,
          "clean": bool,   # True when blockers==0 and majors==0
        }
    Empty list if no review evidence is present.

    2026-06-30 fix — also recognise the "headless" clean-review form
    ("**REVIEW CLEAN. 0 blockers, 0 majors, 6 known minors.**") that
    does not carry a "pass N:" marker. A response containing both
    ``_CLEAN_RE`` evidence ("review clean", "no findings", etc.) AND
    explicit 0-blocker/0-major counts is treated as an additional
    clean review pass; the pass number is synthesised as one greater
    than the highest count this function has already parsed in the
    same text (or 1 if no numbered pass was found). This prevents the
    60-message recite loop where the model writes the same
    "REVIEW CLEAN" line every turn but the gate never accepts it
    because the strict ``_PASS_RE`` requires "pass N:" and the model
    only ever emits the headless form.
    """
    if not response_text:
        return []

    matches = list(_PASS_RE.finditer(response_text))
    reviews: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
        window = response_text[start:end]
        try:
            pass_num = int(m.group(1))
        except (TypeError, ValueError):
            continue
        blockers = _last_count(window, _BLOCKER_RE, _NO_BLOCKER_RE)
        majors = _last_count(window, _MAJOR_RE, _NO_MAJOR_RE)
        clean = blockers == 0 and majors == 0
        reviews.append({
            "pass": pass_num,
            "blockers": blockers,
            "majors": majors,
            "clean": clean,
        })

    # Headless clean-review detection: if no "pass N:" marker matched
    # but the text has clean-review language AND explicit 0-blocker /
    # 0-major counts, synthesise a clean pass entry so the gate can
    # approve without forcing the model into a "pass N:" template.
    if not reviews and _CLEAN_RE.search(response_text):
        headless_blockers = _last_count(
            response_text, _BLOCKER_RE, _NO_BLOCKER_RE,
        )
        headless_majors = _last_count(
            response_text, _MAJOR_RE, _NO_MAJOR_RE,
        )
        if headless_blockers == 0 and headless_majors == 0:
            reviews.append({
                "pass": 1,
                "blockers": 0,
                "majors": 0,
                "clean": True,
            })
    return reviews


def _extract_latest_review_from_text(
    response_text: str,
) -> Optional[Dict[str, Any]]:
    """Parse the most recent review pass evidence from a single text blob.

    Returns the last pass mentioned, or ``None`` if no review evidence is
    present.
    """
    reviews = _extract_all_reviews_from_text(response_text)
    return reviews[-1] if reviews else None


def _count_dirty_passes(
    messages: List[Dict[str, Any]],
    response_text: str = "",
) -> int:
    """Count review passes that were non-clean (blockers or majors > 0)
    across messages + response_text + delegated-review tool messages (#8).
    Used to decide whether proof-of-fix blocks are required.
    """
    count = 0
    for text in _review_text_candidates(messages, response_text):
        for review in _extract_all_reviews_from_text(text):
            if not review["clean"]:
                count += 1
    return count


def _count_verified_proofs(
    response_text: str,
    messages: List[Dict[str, Any]],
) -> int:
    """Count proof-of-fix blocks with `revert-verified: yes` across the
    response and recent assistant messages (#8).

    Used by the proof-of-fix gate mode to require that every blocker fix
    carries a verified proof block.
    """
    count = 0
    for text in [response_text or ""] + [
        m.get("content") for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]:
        if not isinstance(text, str):
            continue
        for m in _PROOF_OF_FIX_RE.finditer(text):
            revert = (m.group("revert") or "").strip().lower()
            if revert in ("yes", "true", "1"):
                count += 1
    return count


def _review_text_candidates(
    messages: List[Dict[str, Any]],
    response_text: str = "",
) -> List[str]:
    """Yield de-duplicated text blobs that may carry review evidence.

    Includes the in-flight ``response_text`` and the content of both
    ``assistant`` messages AND the tool-result messages produced by a
    delegated review (``delegate_task``) or a configured review tool.

    ``tool`` messages matter because a review delegated to a subagent
    returns its report as a tool-result message — the gate must see it or
    it force-redispatches the review forever (the "delegation breaks the
    task" symptom: the build "completes", the gate rejects with "no
    adversarial review evidence found", the model re-dispatches, repeat).
    Only tool results whose originating assistant tool call was a
    delegation/review tool are scanned, so arbitrary terminal/exec output
    containing "0 blockers" cannot spoof review evidence.

    De-duplication by content keeps the consecutive-clean / highest-pass
    counters from double-counting when the dispatcher has already appended
    the response into ``messages``.
    """
    review_tool_names = {"delegate_task"}
    delegated_ids = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if _tool_name(tc) in review_tool_names:
                cid = tc.get("id")
                if cid:
                    delegated_ids.add(cid)
    seen: set = set()
    candidates: List[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content")
        elif role == "tool":
            # Only delegated/review tool results, never raw shell output.
            if msg.get("tool_call_id") not in delegated_ids:
                continue
            content = msg.get("content")
        else:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if content in seen:
            continue
        seen.add(content)
        candidates.append(content)
    if response_text and response_text.strip() and response_text not in seen:
        candidates.append(response_text)
    return candidates


def _find_latest_review_evidence(
    response_text: str,
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Look at the final response first; fall back to recent assistant
    and delegated-review tool messages if the response itself contains no
    review markers.

    This handles the case where the agent emits a "loop converged" closeout
    *after* having produced earlier review pass messages, and the case
    where the review was performed by a subagent (``delegate_task``) whose
    report landed in a ``tool`` message.
    """
    # Iterate chronological candidates; the last non-None review wins
    # (most recent). response_text is appended last by _review_text_candidates.
    latest = None
    for text in _review_text_candidates(messages, response_text):
        rev = _extract_latest_review_from_text(text)
        if rev is not None:
            latest = rev
    return latest


def _count_consecutive_clean_passes(
    messages: List[Dict[str, Any]],
    response_text: str = "",
) -> int:
    """Scan recent assistant messages for clean-review markers, return
    the count of consecutive clean passes at the end of the trace.

    ``response_text`` is the current pending final response; if it
    contains a clean pass marker, it counts as the latest entry in the
    walk. This handles the case where the dispatcher has the final
    response in flight but hasn't appended it to ``messages`` yet.

    Per-pass-mention counting: a single message that mentions several
    review passes contributes one entry *per pass*, not one per message.
    Candidates include delegated-review tool messages (see
    ``_review_text_candidates``) and are de-duplicated by content.
    """
    candidates = _review_text_candidates(messages, response_text)

    pass_sequence: List[bool] = []
    for content in candidates:
        for review in _extract_all_reviews_from_text(content):
            pass_sequence.append(review["clean"])

    # Walk backwards from the end; stop on first non-clean.
    consecutive = 0
    for clean in reversed(pass_sequence):
        if clean:
            consecutive += 1
        else:
            break
    return consecutive


def _highest_pass_seen(messages: List[Dict[str, Any]], response_text: str = "") -> int:
    """Return the highest review pass number observed in any message or
    the in-flight response, including delegated-review tool messages.
    """
    highest = 0
    for text in _review_text_candidates(messages, response_text):
        for m in _PASS_RE.finditer(text):
            try:
                n = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if n > highest:
                highest = n
    return highest


def _first_clean_pass_seen(
    messages: List[Dict[str, Any]],
    response_text: str = "",
) -> Optional[int]:
    """Return the pass number of the *first* clean review pass observed,
    or ``None`` if no clean pass has been seen yet.

    Scans messages + response_text in chronological order. A clean pass
    is one with 0 blockers and 0 majors. Used by the reworked cap logic
    to distinguish "stuck before convergence" (no clean pass yet) from
    "oscillating after convergence started" (clean pass seen, but not
    converging). Includes delegated-review tool messages.
    """
    for content in _review_text_candidates(messages, response_text):
        for review in _extract_all_reviews_from_text(content):
            if review["clean"]:
                return review["pass"]
    return None


def _count_build_tool_calls_after_pass(
    messages: List[Dict[str, Any]],
    after_pass: int,
) -> int:
    """Count *mutating* build tool calls made AFTER the most recent
    ``Review pass N`` where N == ``after_pass`` (i.e., the actual Nth
    review, not just the highest). Used to decide if a post-review fix
    needs a follow-up review pass.

    Walks messages; the first assistant message containing
    ``Review pass N`` (matching ``after_pass``) marks the boundary; any
    mutating build tool call after that boundary is counted. Only
    mutating tools (write_file/patch) count — a read-only shell call
    after a review does not warrant another pass.

    Tool calls in the *same* message as the pass marker are counted too
    (the agent often reports a review and issues a fix in one turn). The
    scan only skips the message's tool_calls if it does NOT contain the
    Nth pass marker — once the boundary is set on a message, that
    message's own tool_calls are scanned.
    """
    boundary_hit = False
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")

        # If we haven't hit the boundary yet, check whether this message
        # contains the Nth pass marker. If it does, set the boundary and
        # fall through to scan this message's own tool_calls (the fix
        # may be in the same turn as the review report).
        if not boundary_hit:
            is_pass_message = (
                isinstance(content, str)
                and after_pass > 0
                and bool(_PASS_RE.search(content))
            )
            if is_pass_message and isinstance(content, str):
                for m in _PASS_RE.finditer(content):
                    try:
                        n = int(m.group(1))
                    except (TypeError, ValueError):
                        continue
                    if n == after_pass:
                        boundary_hit = True
                        break

        if not boundary_hit:
            continue

        # Past the boundary (or the boundary message itself) — count
        # mutating build tool calls.
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if _tool_name(tc) in _MUTATING_BUILD_TOOLS:
                count += 1
    return count


# -----------------------------------------------------------------------------
# Hook: pre_exit_verify
# -----------------------------------------------------------------------------

def _check_rejection_streak(session_id: Optional[str], reason: str) -> Optional[Dict[str, Any]]:
    """Cross-call rejection-reason dedup (fix for 2026-06-29 60-message loop).

    When ``_pre_exit_verify`` would reject with the same ``reason`` string
    for ``_REJECTION_STREAK_LIMIT`` consecutive calls on the same session,
    treat the gate itself as stuck on a stale payload (the model isn't
    converging — the gate is) and return an approved verdict with a
    structured cap notice. Returns ``None`` when no streak-trip has fired.

    Different reasons (or a successful approve between calls) reset the
    streak.
    """
    if not session_id:
        return None
    state = _REJECTION_STREAK.get(session_id)
    sig = (reason or "").strip()[:160]
    if state is None:
        state = {"last_reason": sig, "streak": 1}
        _REJECTION_STREAK[session_id] = state
        return None
    if state["last_reason"] != sig:
        state["last_reason"] = sig
        state["streak"] = 1
        return None
    state["streak"] += 1
    if state["streak"] > _REJECTION_STREAK_LIMIT:
        # Reset so we don't log-spam; next NEW reason starts fresh.
        _REJECTION_STREAK[session_id] = {"last_reason": sig, "streak": 1}
        logger.warning(
            "compounding-loops: same rejection reason fired %dx for session "
            "%s; shipping with stale-rejection cap (reason=%r)",
            state["streak"], session_id, sig[:80],
        )
        return {
            "approved": True,
            "reason": (
                f"rejection-streak cap: the same review-gate rejection reason "
                f"fired {_REJECTION_STREAK_LIMIT}+ times consecutively without "
                "convergence — the gate itself is stuck on a stale payload. "
                f"Last reason: {sig[:120]}"
            ),
        }
    return None


def _reset_rejection_streak(session_id: Optional[str]) -> None:
    """Clear the rejection streak for a session (called on approve/clean-pass)."""
    if session_id:
        _REJECTION_STREAK.pop(session_id, None)


def _update_hints(
    messages: List[Dict[str, Any]],
    response_text: str,
    hints: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Refresh the session-scoped hint cache from the current message list.

    Hints are a *fallback* used when context compaction rewrites assistant
    message content and erases ``Review pass N:`` markers. They are derived
    from the same prose evidence the gate parses, but stashed on the agent
    instance (which compaction does not touch) so they survive compression.
    They are NEVER trusted alone for an approval — only as a "a build /
    review happened, demand re-evidence" signal when prose is missing.

    Returns the updated hints dict (also mutates ``hints`` in place).
    """
    if hints is None:
        hints = {}
    # Climb hints monotonically — compaction can drop messages, so we
    # never want a hint to regress to a lower value than what we saw.
    # Count ONLY mutating tools for the build hint, matching
    # _is_build_response's definition — read-only terminal/exec calls
    # must not inflate the build hint and flip is_build via the fallback.
    mutating_count = _count_build_tool_calls(messages, _MUTATING_BUILD_TOOLS)
    if mutating_count > hints.get("mutating_build_count", 0):
        hints["mutating_build_count"] = mutating_count
    highest = _highest_pass_seen(messages)
    if response_text:
        # _highest_pass_seen only scans messages; fold in response_text.
        for m in _PASS_RE.finditer(response_text):
            try:
                n = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if n > highest:
                highest = n
    if highest > hints.get("highest_review_pass", 0):
        hints["highest_review_pass"] = highest
    # Track whether any review was observed (prose or tool call).
    has_review = (
        _find_latest_review_evidence(response_text, messages) is not None
        or hints.get("has_review", False)
    )
    if has_review:
        hints["has_review"] = True
    return hints


def _pre_exit_verify(
    response_text: str,
    messages: List[Dict[str, Any]],
    *,
    force_build: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Decide whether the assistant may exit after a build.

    Returns:
        ``None`` or ``{"approved": True}`` when no enforcement is needed.
        ``{"approved": False, "reason": ..., "max_reverify_passes": N}``
        when the loop has not converged.
        ``{"approved": True, "reason": ...}`` when approving via the hard
        cap (the reason surfaces that the cap was hit so a downstream
        logger can flag a ship-with-open-findings outcome).

    ``force_build`` (default False): when True, skip the build-detection
    early-return and run the full gate logic. Used by ``_evaluate_review_gate``
    on the standard ``post_api_request`` hook path, where the dispatcher
    doesn't pass a full messages list — the existence of a review marker in
    the response is itself proof the agent is in a review loop. Replaces
    the older "append a fake tool call" hack that contaminated build counters.

    ``session_hints`` kwarg (optional): a mutable dict stashed on the
    agent instance by the caller, used as a compaction-surviving fallback
    when prose review evidence is missing. See ``_update_hints``.
    """
    cfg = _config()

    if not cfg["enabled"]:
        return {"approved": True}

    # Always allow empty-response rejections to fall through to the
    # shape-only harness-conductor (don't double-reject).
    if not response_text or not response_text.strip():
        return {"approved": True}

    # Cap-quiet brake (2026-06-30 fix). Once the gate has shipped via the
    # hard cap and the model has burned through its grace followup
    # responding with review-shaped text, the cap is permanent for the
    # session. Re-asking for review is theatre that burns tokens. Approve
    # the current response without further gate review and clear the
    # followup counter so the next user turn starts fresh.
    session_id = kwargs.get("session_id")
    if session_id and _CAP_QUIET_STREAK.get(session_id, 0) > _CAP_QUIET_STREAK_GRACE:
        logger.warning(
            "compounding-loops: cap-quiet brake tripped for session %s "
            "(post-cap followups=%d > grace=%d); auto-approving without "
            "review gate. The cap is permanent within the session.",
            session_id,
            _CAP_QUIET_STREAK[session_id],
            _CAP_QUIET_STREAK_GRACE,
        )
        # Drop the streak so subsequent turns (after a new user message)
        # re-engage the gate normally.
        _CAP_QUIET_STREAK.pop(session_id, None)
        return {
            "approved": True,
            "reason": (
                "cap-quiet brake: gate previously shipped via hard cap; "
                "auto-approving further review-shaped responses to break "
                "the post-cap recite loop"
            ),
        }

    # Update the compaction-surviving hint cache (if the caller provided
    # one) BEFORE any gate logic, so hints reflect the current state.
    hints = kwargs.get("session_hints")
    hints = _update_hints(messages, response_text, hints)

    # If the user asked for a quick/trivial answer, don't gate. Checked
    # BEFORE the step cap so a runaway-loop session can't sneak past
    # the brakes by including a generic "thanks!" reply at the end.
    if _bypass_requested(messages, cfg):
        return {"approved": True}

    # If this wasn't a build, don't gate. Compaction fallback: if the
    # hint cache remembers MORE mutating builds than are currently visible
    # in messages, evidence was erased by compaction — treat it as a
    # build and demand re-evidence rather than silently approving. Only
    # engage when the hint exceeds the current count (otherwise the hint
    # is just reflecting the current state, not a compaction loss).
    # ``force_build`` bypasses both checks (the standard-hook path
    # doesn't pass messages, so the hint cache and live counts are
    # empty — the existence of review evidence is the only signal).
    is_build = force_build or _is_build_response(messages, cfg)
    if not is_build:
        _current_mutating = _count_build_tool_calls(messages, _MUTATING_BUILD_TOOLS)
        if hints.get("mutating_build_count", 0) > _current_mutating:
            is_build = True

    if not is_build:
        _reset_rejection_streak(kwargs.get("session_id"))
        return {"approved": True}

    highest_pass = _highest_pass_seen(messages)
    # Fold in the hint cache's highest pass — compaction may have erased
    # the message that contained the highest pass marker.
    if hints.get("highest_review_pass", 0) > highest_pass:
        highest_pass = hints["highest_review_pass"]

    # Step cap (#2): total tool calls this session must not exceed the
    # configured ceiling. Prevents runaway recursion (PDF 1 death #a).
    turn_count = _count_build_tool_calls(messages, _ALL_BUILD_TOOLS)
    if cfg["max_turns"] and turn_count >= cfg["max_turns"]:
        _write_state_snapshot(
            verdict="cap", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=None, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        return {
            "approved": True,
            "reason": (
                f"step cap reached ({turn_count}/{cfg['max_turns']} tool "
                f"calls this session); shipping with any open findings"
            ),
        }

    # Circuit breaker (#2): detect the same tool + same args repeated
    # cfg["circuit_breaker"] times in a row and hard-stop. Cures the
    # runaway-recursion loop death.
    _cb_tool, _cb_args = _detect_circuit_breaker(messages, cfg["circuit_breaker"])
    if _cb_tool is not None:
        _write_state_snapshot(
            verdict="cap", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=None, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
            circuit_breaker_tripped=True,
        )
        return {
            "approved": True,
            "reason": (
                f"circuit breaker tripped: tool '{_cb_tool}' called with "
                f"identical args {_cb_args[:80]}... {cfg['circuit_breaker']}x "
                f"in a row; halting to prevent runaway recursion"
            ),
        }

    # Hard cap — two-tier:
    #
    # Tier 1 (stuck cap): No clean pass has been seen yet and we've burned
    # through ``max_passes`` dirty passes. The loop is stuck — the agent
    # isn't resolving blockers/majors. Ship with findings + a cap notice.
    #
    # Tier 2 (oscillation cap): A clean pass WAS seen at pass N, but the
    # loop hasn't converged after ``max_passes`` additional passes. This
    # catches clean→dirty→clean→dirty oscillation where each fix round
    # introduces a new issue. Ship with findings + a cap notice.
    #
    # When neither tier fires, the double-clean convergence logic below
    # owns the decision. This means a task that needs 5 dirty passes before
    # its first clean pass at pass 6 will get pass 7 to confirm — the hard
    # cap doesn't cut it off at 3.
    first_clean = _first_clean_pass_seen(messages, response_text)
    _cap_reason = None
    if first_clean is None and highest_pass >= cfg["stuck_cap"]:
        _cap_reason = (
            f"stuck cap: {highest_pass} dirty review passes with no clean "
            f"pass (limit {cfg['stuck_cap']}); shipping with open findings"
        )
    elif first_clean is not None and highest_pass >= first_clean + cfg["max_passes"]:
        _cap_reason = (
            f"oscillation cap: first clean pass was {first_clean}, but "
            f"loop hasn't converged after {cfg['max_passes']} more passes "
            f"(now at pass {highest_pass}); shipping with open findings"
        )
    if _cap_reason is not None:
        if cfg["review_tools"] and not _has_review_tool_call(messages, cfg["review_tools"]):
            _cap_reason += (
                " — WARNING: shipped WITHOUT a real review tool call "
                "(HERMES_LOOPS_REVIEW_TOOLS was set)"
            )
        _write_state_snapshot(
            verdict="cap", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=None, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        return {"approved": True, "reason": _cap_reason}

    # Process-police mode (opt-in): if review tools are configured, require
    # at least one such tool call in the history. This prevents the agent
    # from satisfying the gate with prose alone.
    if cfg["review_tools"] and not _has_review_tool_call(messages, cfg["review_tools"]):
        _write_state_snapshot(
            verdict="rejected", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=None, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        _reason = (
            "no review tool call found in history; invoke one of "
            f"{sorted(cfg['review_tools'])} and report the result "
            "before declaring done"
        )
        _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
        if _cap is not None:
            _reset_rejection_streak(kwargs.get("session_id"))
            return _cap
        return {
            "approved": False,
            "reason": _reason,
            "max_reverify_passes": max(1, cfg["max_passes"] - highest_pass),
        }

    # Was a review pass attempted in the latest response or recent
    # assistant messages?
    review = _find_latest_review_evidence(response_text, messages)

    if review is None:
        # Compaction fallback: if prose review evidence is missing but
        # the hint cache remembers a review happened earlier this
        # session, demand re-evidence rather than silently approving.
        # The hint is NOT trusted for approval — only as a signal that
        # a review existed and its evidence was lost.
        if hints.get("has_review"):
            _write_state_snapshot(
                verdict="rejected", hints=hints, highest_pass=highest_pass,
                turn_count=turn_count, review=None, consecutive_clean=0,
                session_id=kwargs.get("session_id"),
            )
            _reason = (
                "a review pass was observed earlier this session but "
                "its text evidence is no longer present in the message "
                "history (likely due to context compaction); re-run "
                "the review and report blockers/majors explicitly "
                "before declaring done"
            )
            _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
            if _cap is not None:
                _reset_rejection_streak(kwargs.get("session_id"))
                return _cap
            return {
                "approved": False,
                "reason": _reason,
                "max_reverify_passes": max(1, cfg["max_passes"] - highest_pass),
            }
        _write_state_snapshot(
            verdict="rejected", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=None, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        _reason = (
            "build completed but no adversarial review evidence found in "
            "the response; run an adversarial-work-review pass and report "
            "blockers/majors explicitly (or 'review clean' + '0 blockers, "
            "0 majors') before declaring done"
        )
        _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
        if _cap is not None:
            _reset_rejection_streak(kwargs.get("session_id"))
            return _cap
        return {
            "approved": False,
            "reason": _reason,
            "max_reverify_passes": cfg["max_passes"],
        }

    # Most recent pass found blockers or majors — must be addressed.
    if review["blockers"] > 0:
        _write_state_snapshot(
            verdict="rejected", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=review, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        _reason = (
            f"review pass {review['pass']} found {review['blockers']} "
            "blocker(s); fix each, re-run review, and report the new "
            "pass number with 0 blockers"
        )
        _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
        if _cap is not None:
            _reset_rejection_streak(kwargs.get("session_id"))
            return _cap
        return {
            "approved": False,
            "reason": _reason,
            "max_reverify_passes": max(1, cfg["max_passes"] - highest_pass),
        }

    if review["majors"] > 0:
        _write_state_snapshot(
            verdict="rejected", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=review, consecutive_clean=0,
            session_id=kwargs.get("session_id"),
        )
        _reason = (
            f"review pass {review['pass']} found {review['majors']} major "
            "finding(s); address them and re-review"
        )
        _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
        if _cap is not None:
            _reset_rejection_streak(kwargs.get("session_id"))
            return _cap
        return {
            "approved": False,
            "reason": _reason,
            "max_reverify_passes": max(1, cfg["max_passes"] - highest_pass),
        }

    # Review was clean — check for convergence.
    consecutive = _count_consecutive_clean_passes(messages, response_text)
    if cfg["require_double_clean"] and consecutive < 2:
        _write_state_snapshot(
            verdict="rejected", hints=hints, highest_pass=highest_pass,
            turn_count=turn_count, review=review, consecutive_clean=consecutive,
            session_id=kwargs.get("session_id"),
        )
        _reason = (
            f"review pass {review['pass']} clean; need one more "
            f"consecutive clean pass to confirm no regressions "
            f"(currently {consecutive}/2)"
        )
        _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
        if _cap is not None:
            _reset_rejection_streak(kwargs.get("session_id"))
            return _cap
        return {
            "approved": False,
            "reason": _reason,
            "max_reverify_passes": max(
                1, min(2, cfg["max_passes"] - highest_pass)
            ),
        }

    # #8: Proof-of-fix mode — when enabled and the loop iterated (a dirty
    # pass occurred before the clean streak), require verified proof-of-fix
    # blocks for each fix round. "Iterations happened" means at least one
    # review pass in the history was non-clean. A clean pass with no
    # proofs after fixes is rejected: the agent must prove the fix was
    # verified by revert-or-mutate, not just assert "fixed" in prose.
    _dirty_pass_count = _count_dirty_passes(messages, response_text)
    if cfg["proof_of_fix"] and _dirty_pass_count > 0:
        proof_count = _count_verified_proofs(response_text, messages)
        # Require one verified proof per dirty pass (each fix round).
        required = _dirty_pass_count
        if proof_count < required:
            _write_state_snapshot(
                verdict="rejected", hints=hints, highest_pass=highest_pass,
                turn_count=turn_count, review=review,
                consecutive_clean=consecutive,
                session_id=kwargs.get("session_id"),
            )
            _reason = (
                f"proof-of-fix mode requires {required} verified "
                f"proof-of-fix block(s) (revert-verified: yes) for the "
                f"fix round(s); found {proof_count}. Emit one block "
                f"per fix in the format:\n"
                f"  Proof-of-fix <id>:\n"
                f"    test: <test name>\n"
                f"    red-before: <fail description>\n"
                f"    green-after: <pass description>\n"
                f"    revert-verified: yes"
            )
            _cap = _check_rejection_streak(kwargs.get("session_id"), _reason)
            if _cap is not None:
                _reset_rejection_streak(kwargs.get("session_id"))
                return _cap
            return {
                "approved": False,
                "reason": _reason,
                "max_reverify_passes": max(
                    1, min(2, cfg["max_passes"] - highest_pass)
                ),
            }

    # Loop complete.
    _write_state_snapshot(
        verdict="approved", hints=hints, highest_pass=highest_pass,
        turn_count=turn_count, review=review, consecutive_clean=consecutive,
        session_id=kwargs.get("session_id"),
    )
    _reset_rejection_streak(kwargs.get("session_id"))
    return {"approved": True}


# -----------------------------------------------------------------------------
# Hook: post_tool_batch_reflect
# -----------------------------------------------------------------------------

def _post_tool_batch_reflect(
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Decide whether a reflection pass is warranted.

    Triggers a short reflection if:
    - A non-trivial build batch just landed (>= MIN_BUILD_TOOLS calls), AND
    - No adversarial review pass has been observed yet in this session.

    Reflect-nudge dedup (2026-06-30): once we've nudged for a given
    build-tool-call count on a session, we won't nudge again until the
    count grows. Without this, the nudge → empty tool call → re-nudge
    loop produces infinite `out=20` API calls (the symptom you saw in
    the kanban concurrency session). The session_id is taken from the
    kwarg the dispatcher passes (``session_id=``); anonymous calls skip
    deduplication rather than sharing state.
    """
    cfg = _config()

    if not cfg["enabled"]:
        return {"reflect": False}

    if _bypass_requested(messages, cfg):
        return {"reflect": False}

    if not _is_build_response(messages, cfg):
        return {"reflect": False}

    session_id = _session_key(kwargs)
    current_tail = _recent_mutating_signatures(
        messages, _MUTATING_BUILD_TOOLS, _LAST_REFLECT_TAIL_SIZE
    )
    last_tail = _LAST_REFLECT_TAIL.get(session_id, ()) if session_id else ()
    if last_tail and current_tail[-len(last_tail):] == last_tail:
        # The tail of current mutating signatures matches what we last
        # nudged on — no new work since. Stay silent.
        return {"reflect": False}

    # If a review pass has already happened, look for a follow-up build
    # (write/patch after the last review) — if so, nudge the agent to
    # verify the fix in another pass. Otherwise stay silent.
    highest_pass = _highest_pass_seen(messages)
    if highest_pass >= 1:
        post_review_build = _count_build_tool_calls_after_pass(
            messages, highest_pass
        )
        if post_review_build >= 1:
            if session_id:
                _LAST_REFLECT_TAIL[session_id] = current_tail
            return {
                "reflect": True,
                "reason": (
                    f"build activity detected after review pass {highest_pass}; "
                    f"consider running review pass {highest_pass + 1} to verify "
                    f"the fix didn't introduce regressions"
                ),
                "max_reflect_passes": 1,
            }
        return {"reflect": False}

    if session_id:
        _LAST_REFLECT_TAIL[session_id] = current_tail
    return {
        "reflect": True,
        "reason": (
            "build batch landed without an adversarial review pass yet; "
            "consider running one (review pass 1) before closeout"
        ),
        "max_reflect_passes": 1,
    }


# -----------------------------------------------------------------------------
# Hooks: hermes core integration (pre_llm_call / post_api_request /
# post_llm_call / transform_llm_output).
#
# The forward-future build exposed pre_exit_verify + post_tool_batch_reflect,
# which fire right at the exit boundary. The shipped hermes build only exposes
# the standard hooks. To enforce the loop across both, we drive the same
# _pre_exit_verify gate from post_api_request (which fires after every LLM
# response with finish_reason + the assistant message in hand) and act on the
# verdict:
#
#   - approved (converged OR hard cap): set a session flag so
#     transform_llm_output can append the convergence/cap notice and
#     pre_llm_call can tell the model to stop reviewing and close out.
#   - rejected: stash the reason; pre_llm_call injects it as context into
#     the next turn so the model knows what to fix and re-review.
#
# All state lives on a module-level dict keyed by session_id so we don't
# rely on the caller passing a session_hints cache (the standard hooks
# don't).
# -----------------------------------------------------------------------------

# Verdict cache: session_id -> {"verdict": "approved"|"rejected",
# "reason": str, "highest_pass": int, "approved_via_cap": bool,
# "generation": int}  — the generation counter guards against the
# pre_llm_call / post_api_request race when a verdict has been consumed
# but the manager fires pre_llm_call for a new turn before the
# corresponding post_api_request stores its verdict. pre_llm_call refuses
# to inject a verdict whose generation doesn't match the latest
# post_api_request write for that session.
_LOOP_VERDICTS: Dict[str, Dict[str, Any]] = {}

# Per-session accumulator of review passes seen across post_api_request
# calls. The standard hook doesn't hand us the full message history, so
# we reconstruct the consecutive-clean streak from this log instead of
# from _count_consecutive_clean_passes(messages, response_text).
# session_id -> List[(pass_num, clean_bool)]  — chronological order.
_SESSION_CLEAN_STREAK: Dict[str, List[tuple]] = {}

# Generation counter — bumped on every successful verdict write for that
# session. The pre_llm_call path consults its own session counter to detect
# a verdict refresh race without letting unrelated sessions invalidate it.
_VERDICT_GENERATION: Dict[str, int] = {}

# Cross-call rejection-reason dedup (2026-06-29 60-message-loop fix).
# Tracks the last N rejection reasons per session; when the same reason
# repeats consecutively more than the threshold, treat the loop as
# stuck-stale (the model isn't converging — the gate itself is stuck
# firing on a stale payload) and ship with a structured cap notice.
# session_id -> {"last_reason": str, "streak": int}
_REJECTION_STREAK: Dict[str, Dict[str, Any]] = {}
_REJECTION_STREAK_LIMIT = int(
    os.environ.get("HERMES_LOOPS_REJECTION_STREAK_LIMIT", "3")
)

# Post-cap-followup counter (2026-06-30 ACP-loop fix). After a cap-approval
# fires, the model may still emit a response shaped like a review. We count
# how many such post-cap followups have arrived; when the count reaches the
# cap-followup ceiling, the gate stops re-asking for review and auto-approves
# the response — the cap is permanent within the session, so re-asking is
# theatre that burns tokens. Reset on session start/reset (see
# ``_on_session_reset``). session_id -> int.
_CAP_QUIET_STREAK: Dict[str, int] = {}
# Default: 1 post-cap followup is allowed (so the model gets one chance to
# emit a true closeout after seeing the cap nudge). The 2nd post-cap
# followup auto-approves. Set to 0 to skip the grace followup entirely.
_CAP_QUIET_STREAK_GRACE = int(
    os.environ.get("HERMES_LOOPS_CAP_QUIET_GRACE", "1")
)

# Reflect-nudge deduplication (2026-06-30 ACP-loop fix). Tracks the
# tuple of the last few mutating-build tool-call signatures the last
# time _post_tool_batch_reflect returned reflect=True for a given
# session. Subsequent calls with the same tail return reflect=False —
# the agent can't make progress without doing new work, so nudging
# again just forces an empty tool call that re-triggers this hook.
#
# Why signatures, not a count: counts can go DOWN after context
# compaction erases old tool calls, leaving the dedup stuck in
# "same count, no nudge" mode forever even when the agent did real
# new work after compaction. Signatures don't go down. We compare
# the last N (default 8) — old entries fall off as new ones arrive.
# session_id -> Tuple[str, ...].
_LAST_REFLECT_TAIL: Dict[str, tuple] = {}
_LAST_REFLECT_TAIL_SIZE = 8


def _session_key(kwargs: Dict[str, Any]) -> Optional[str]:
    """Resolve a session_id, or return None so stateful hooks can no-op."""
    sid = kwargs.get("session_id")
    if sid:
        return str(sid)
    logger.debug(
        "compounding-loops: hook called without session_id; "
        "skipping session-scoped state"
    )
    return None


def _session_ids_for_cleanup(kwargs: Dict[str, Any]) -> set[str]:
    """Return every old/current/new session identifier supplied by a lifecycle hook."""
    return {
        str(value)
        for key in ("old_session_id", "session_id", "new_session_id")
        if (value := kwargs.get(key))
    }


def _clear_session_state(session_ids: set[str]) -> None:
    for session_id in session_ids:
        _LOOP_VERDICTS.pop(session_id, None)
        _SESSION_CLEAN_STREAK.pop(session_id, None)
        _LAST_REFLECT_TAIL.pop(session_id, None)
        _CAP_QUIET_STREAK.pop(session_id, None)
        _REJECTION_STREAK.pop(session_id, None)
        _VERDICT_GENERATION.pop(session_id, None)


def _clear_persisted_loop_state(session_ids: set[str]) -> None:
    """Remove STATUS.json only when it belongs to a lifecycle session.

    STATUS.json is a singleton, so lifecycle callbacks must never use the
    historical unconditional clear: a delayed callback for session A must not
    erase a newer snapshot for session B.
    """
    if _clear_loop_state is None:
        return
    for session_id in session_ids:
        try:
            _clear_loop_state(session_id)
        except Exception as exc:  # lifecycle cleanup must stay best-effort
            logger.debug("loop-state clear failed: %s", exc)


def _evaluate_review_gate(
    response_text: str,
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run only the review-evaluation portion of the gate.

    Skips the ``is_build`` early-return (the post_api_request hook already
    knows we're in a build/review cycle because a review pass marker is
    present in the response). Returns the gate verdict dict, or None when
    no review evidence is present in the response.

    This is a thin wrapper around _pre_exit_verify that uses ``force_build``
    instead of the older "synthesize fake tool calls" hack. The hack
    contaminated build counters (hint cache, max_turns cap); force_build
    is signal-only — the gate's cap and clean-streak logic still runs,
    but build counters are not lied about.

    ``session_id`` is threaded into ``_pre_exit_verify`` so the persisted
    state file isn't always written with an empty session_id (which would
    cross-pollute if multiple anonymous callers hit the gate concurrently).
    """
    if not response_text:
        return None
    # Only engage when the response actually carries review evidence.
    # Without this, every final text response (including "hello") would
    # be run through the blocker regexes.
    if _find_latest_review_evidence(response_text, messages) is None:
        return None
    # On the standard hook path, we have no messages; fall back to the
    # accumulator-driven cap logic instead of pretending we saw writes.
    msgs = list(messages) if messages else []
    kwargs: Dict[str, Any] = {}
    if session_id is not None:
        kwargs["session_id"] = session_id
    return _pre_exit_verify(
        response_text, msgs,
        force_build=True,
        **kwargs,
    )


def _post_api_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """After each LLM call, decide whether the loop has converged.

    Fires on every API response (tool-calling or final). We only act on
    responses that look like a *final* turn — finish_reason == "stop" and
    no tool_calls — and that carry review evidence, because that's the
    only point where a review pass would have been emitted.
    """
    cfg = _config()
    if not cfg["enabled"]:
        return None

    session_id = _session_key(kwargs)
    if session_id is None:
        return None

    finish_reason = kwargs.get("finish_reason") or ""
    assistant_message = kwargs.get("assistant_message")
    # If the response carries tool_calls, the loop isn't done — let it
    # run. (A review pass that ends with a tool call to fix something is
    # mid-flight, not a final answer.)
    has_tool_calls = bool(
        (getattr(assistant_message, "tool_calls", None) or [])
    )
    if has_tool_calls:
        return None

    # Only inspect text-bearing responses. finish_reason=="stop" is the
    # final-answer signal; "length"/"content_filter" should not trigger
    # the gate (the response was truncated, not a clean closeout).
    if finish_reason and finish_reason != "stop":
        return None

    response_text = ""
    if assistant_message is not None:
        c = getattr(assistant_message, "content", None)
        if isinstance(c, str):
            response_text = c
    if not response_text:
        return None

    # DoS guard: a runaway model emitting many MB of repetitive text would
    # otherwise be scanned in full by the regex pass. Cap at 200 KB — a
    # legitimate review closeout is <50 KB; anything bigger is treated as
    # truncated/suspicious and bypassed for gate purposes.
    _MAX_GATE_RESPONSE_BYTES = 200_000
    if len(response_text.encode("utf-8", errors="replace")) > _MAX_GATE_RESPONSE_BYTES:
        logger.warning(
            "compounding-loops: response_text %d bytes exceeds gate cap %d; "
            "bypassing review evidence check for this turn",
            len(response_text), _MAX_GATE_RESPONSE_BYTES,
        )
        return None

    session_id = _session_key(kwargs)
    # The standard post_api_request hook doesn't hand us the full message
    # history, so we can only inspect the current response. Pass an empty
    # messages list; _evaluate_review_gate uses force_build=True to engage
    # the gate without contaminating build counters.
    messages: List[Dict[str, Any]] = []
    verdict = _evaluate_review_gate(response_text, messages, session_id=session_id)
    if not isinstance(verdict, dict):
        return None

    approved = bool(verdict.get("approved"))
    reason = str(verdict.get("reason", ""))

    # Override the double-clean decision using the cross-turn streak we
    # accumulate in _SESSION_CLEAN_STREAK. _pre_exit_verify could only see
    # response_text (empty messages), so its consecutive-clean count is
    # at most 1. We track the real streak here and re-decide approval when
    # the gate rejected solely on "need one more consecutive clean pass".
    #
    # Also override the cap decision: _pre_exit_verify's cap logic fires
    # based on highest_pass from messages (which is 0 in the hook path
    # since we pass empty messages). We recompute the two-tier cap from
    # the streak accumulator instead.
    #
    # The hook path bypasses the step cap and circuit breaker because we
    # have no messages — those settings (``HERMES_LOOPS_MAX_TURNS``,
    # ``HERMES_LOOPS_CIRCUIT_BREAKER``) only fire on the ``pre_exit_verify``
    # path, where messages are available. This is an acknowledged gap:
    # operators running on the standard hook path and relying on
    # ``HERMES_LOOPS_MAX_TURNS`` to catch runaway sessions need to switch
    # to the legacy ``pre_exit_verify`` hook. Documented in README.
    reviews = _extract_all_reviews_from_text(response_text)
    if reviews:
        latest = reviews[-1]
        streak = _SESSION_CLEAN_STREAK.setdefault(session_id, [])
        # If the latest pass number is new, append; if it's a re-report of
        # the same pass (the model re-emitted it), replace the last entry.
        # A re-report must NOT inflate ``highest_pass`` — that's how
        # ``stuck_cap`` got tripped by a single-pass session in 2.2.
        if not streak or streak[-1][0] != latest["pass"]:
            streak.append((latest["pass"], latest["clean"]))
        else:
            streak[-1] = (latest["pass"], latest["clean"])
        # De-dup by pass number: if a model re-emitted the same pass after
        # an intervening higher pass, the older record is stale — drop it
        # so ``highest_pass`` reflects the most recent pass signal only.
        # This protects against the user-reported "runaway" symptom where
        # a model gets stuck re-emitting pass markers.
        seen_passes: set = set()
        deduped: List[tuple] = []
        for entry in streak:
            if entry[0] not in seen_passes:
                seen_passes.add(entry[0])
                deduped.append(entry)
        streak[:] = deduped

        # ``highest_pass`` is the max pass NUMBER seen, but unique-per-pass
        # means it's also the length of unique passes. Either way it's
        # the bound for cap decisions.
        highest_pass = max(p for p, _ in streak)
        first_clean = None
        for p, clean in streak:
            if clean:
                first_clean = p
                break

        consecutive_clean = 0
        for _pass, clean in reversed(streak):
            if clean:
                consecutive_clean += 1
            else:
                break

        # Re-evaluate the two-tier cap using the accumulated session state.
        # _pre_exit_verify returned approved=True via the cap branch because
        # it saw highest_pass=0 from empty messages and the cap never fired.
        # But it might also have approved via cap because it parsed the pass
        # number from response_text alone. We need to check both tiers here.
        cap_reason = None
        if first_clean is None and highest_pass >= cfg["stuck_cap"]:
            cap_reason = (
                f"stuck cap: {highest_pass} dirty review passes with no clean "
                f"pass (limit {cfg['stuck_cap']}); shipping with open findings"
            )
        elif first_clean is not None and highest_pass >= first_clean + cfg["max_passes"]:
            cap_reason = (
                f"oscillation cap: first clean pass was {first_clean}, but "
                f"loop hasn't converged after {cfg['max_passes']} more passes "
                f"(now at pass {highest_pass}); shipping with open findings"
            )
        if cap_reason is not None:
            approved = True
            reason = cap_reason
        elif (
            not approved
            and latest["clean"]
            and "need one more consecutive clean pass" in reason
            and consecutive_clean >= (2 if cfg["require_double_clean"] else 1)
        ):
            # The gate rejected only for needing one more clean pass, and
            # we now have enough from the accumulated streak. Flip to
            # approved.
            approved = True
            reason = ""
    # Clear the streak when approved so a new build starts fresh.
    if approved:
        _SESSION_CLEAN_STREAK.pop(session_id, None)

    # Bump only this session's generation so another session cannot
    # invalidate this pending verdict.
    generation = _VERDICT_GENERATION.get(session_id, 0) + 1
    _VERDICT_GENERATION[session_id] = generation
    _LOOP_VERDICTS[session_id] = {
        "verdict": "approved" if approved else "rejected",
        "reason": reason,
        "approved_via_cap": "stuck cap" in reason.lower()
                             or "oscillation cap" in reason.lower()
                             or "circuit breaker" in reason.lower(),
        "generation": generation,
    }
    logger.debug(
        "compounding-loops: post_api_request verdict=%s reason=%s gen=%d",
        _LOOP_VERDICTS[session_id]["verdict"],
        reason[:120],
        generation,
    )
    return None


def _pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Before the next LLM call, inject guidance based on the last verdict.

    - If the previous final-response was REJECTED by the gate, inject the
      rejection reason as context so the model knows what to fix and that
      it must run another review pass.
    - If the previous final-response was APPROVED, inject a terse
      "loop converged, deliver your final answer now" instruction so the
      model doesn't start another spontaneous review pass.

    Returns ``{"context": str}`` to inject into the user message, or None.

    Race safety: ``_post_api_request`` writes a verdict under a unique
    per-session ``generation`` tag. After reading the cached verdict,
    we re-read that session's generation. If it advanced, the verdict
    was refreshed before consumption and is dropped rather than injected.
    """
    cfg = _config()
    if not cfg["enabled"]:
        return None

    session_id = _session_key(kwargs)
    if session_id is None:
        return None
    cached = _LOOP_VERDICTS.get(session_id)
    if not cached:
        return None
    cached_gen = cached.get("generation", 0)

    # Re-snapshot this session's generation after reading the verdict.
    # A refresh in the same session means the cached value is stale.
    gen_after_read = _VERDICT_GENERATION.get(session_id, 0)
    if cached_gen < gen_after_read:
        # A newer verdict was written between our read and now — drop
        # the stale one. The next pre_llm_call will inject the newer
        # verdict (or a fresh one will be written and consumed).
        _LOOP_VERDICTS.pop(session_id, None)
        return None

    if cached["verdict"] == "rejected":
        # Clear after consuming so we don't re-inject on every subsequent
        # turn — the model gets one nudge, then the next final response
        # produces a fresh verdict.
        reason = cached.get("reason", "")
        _LOOP_VERDICTS.pop(session_id, None)
        return {
            "context": (
                "[compounding-loops] The previous response was rejected by "
                "the review gate: " + reason + " Address the findings and "
                "run another review pass, reporting the new pass number "
                "with explicit blocker/major counts. Do NOT simply repeat "
                "the prior answer."
            )
        }

    # Approved — nudge the model to close out cleanly.
    via_cap = cached.get("approved_via_cap", False)
    _LOOP_VERDICTS.pop(session_id, None)
    if via_cap:
        # Bump the cap-followup counter so the gate knows the model
        # got the cap nudge; subsequent identical-shape responses
        # auto-approve instead of re-asking for review.
        _CAP_QUIET_STREAK[session_id] = _CAP_QUIET_STREAK.get(session_id, 0) + 1
        # Tight, command-form nudge (2026-06-30 fix): the previous text
        # said "Present your final answer now, summarising what was built"
        # which the model interpreted as "run another review pass". The
        # gate is already past its hard cap; the only thing the model
        # should do is emit a one-line closeout and stop. Forbid review
        # prose and tool calls explicitly so the agent cannot re-enter
        # the review loop.
        return {
            "context": (
                "[compounding-loops] The review loop hit its hard cap. "
                "The build is shipped-ready with whatever open findings the cap "
                "captured. Do NOT run another review pass. Do NOT call any tools. "
                "Do NOT write a review, summary, or list of findings. "
                "Output exactly one short closeout line (e.g. 'Build shipped. "
                "Cap hit, X open findings noted.') and stop. Further turns on "
                "this topic will be auto-approved without gate review."
            )
        }
    return {
        "context": (
            "[compounding-loops] The review loop converged (consecutive "
            "clean passes). Do NOT run another review pass. Present your "
            "final answer now."
        )
    }


def _post_llm_call(**kwargs: Any) -> None:
    """Fire-and-forget bookkeeping after the turn completes.

    Used to clear stale verdicts so a new user turn starts fresh.
    """
    session_id = _session_key(kwargs)
    if session_id is None:
        return None
    # Keep the verdict for one pre_llm_call cycle (the next turn), then
    # drop it. post_llm_call fires after the final response is returned,
    # so the verdict has already been consumed by pre_llm_call of this
    # turn — clear it now.
    _LOOP_VERDICTS.pop(session_id, None)
    return None


def _transform_llm_output(**kwargs: Any) -> Optional[str]:
    """Append a convergence/cap notice to the final response when the
    gate approved via the cap (ship-with-findings outcome).

    A clean convergence approval leaves the response untouched — the
    model's own closeout is the final answer.
    """
    cfg = _config()
    if not cfg["enabled"]:
        return None

    session_id = _session_key(kwargs)
    if session_id is None:
        return None
    cached = _LOOP_VERDICTS.get(session_id)
    if not cached or cached.get("verdict") != "approved":
        return None

    response_text = kwargs.get("response_text") or ""
    if not response_text:
        return None

    if cached.get("approved_via_cap"):
        notice = (
            "\n\n---\n_[compounding-loops] review-loop cap reached; "
            "shipped with any open findings._"
        )
        if notice not in response_text:
            return response_text + notice
    return None


def _on_session_reset(**kwargs: Any) -> None:
    """Clear old and replacement session state after a reset."""
    session_ids = _session_ids_for_cleanup(kwargs)
    _clear_session_state(session_ids)
    _clear_persisted_loop_state(session_ids)
    return None


def _on_session_end(**kwargs: Any) -> None:
    """Compatibility no-op: this hook is a per-turn boundary, not finalization."""
    return None


def _on_session_finalize(**kwargs: Any) -> None:
    """Clear all supplied session state at conversation finalization.

    Finalizers may supply lifecycle metadata in addition to any combination
    of old/current/new session IDs; ``**kwargs`` keeps this callback ABI-safe.
    """
    session_ids = _session_ids_for_cleanup(kwargs)
    _clear_session_state(session_ids)
    _clear_persisted_loop_state(session_ids)
    return None
