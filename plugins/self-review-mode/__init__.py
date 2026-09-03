"""Self-review mode — pre-emptive adversarial self-speculation.

Implements the self-speculation idea from arXiv:2607.25816
("Speculate While You Reason") as an inference-time workflow add-on that sits
alongside the existing post-hoc adversarial-work-review loop.

The paper's key transferable move: the *deployed agent itself* is the best
predictor of what a hostile reviewer will find, if you prompt it from its own
partial trajectory before it ships. External draft reviewers (fresh subagents)
carry a "speculator-agent gap" — they guess a different call than the agent
will actually make. Self-speculation closes that gap: same model, suffix-triggered
mode switch, with explicit mode isolation.

This plugin is deliberately NOT the compounding-loops gate. It does not block
exits, does not own convergence, and does not touch message-history verdict
parsing. It provides:

Tools
-----
- ``self_review`` — emit a speculator-mode suffix prompting the model to predict
  (before shipping) the findings a hostile reviewer WOULD raise, along with a
  mode-isolation card. Cheap, on-policy, aligned with the agent's own reasoning.
- ``self_review_score`` — score a set of predicted findings against actual
  findings using the hard-gate + token-F1 + miss-penalty alignment metric
  (see scoring.py). Returns the alignment score and a miss breakdown.

Hooks
-----
- ``pre_llm_call`` (opt-in, default OFF via ``SELF_REVIEW_SUFFIX=1``) — appends
  a compact self-speculation suffix to build-bound prompts so the model scrapes
  its own likely blockers before it answers. Default off because suffix
  injection on every call can degrade quality; prefer the explicit
  ``self_review`` tool or a manual mode card.

Mode isolation
--------------
A small on-disk mode marker (``~/.hermes/loop-state/self-review-mode.json``)
records the current role so a builder does not carry the reviewer's framing
into a subsequent build and vice-versa — the "reset optimizer state at mode
switch" analog from the paper's training recipe.

Configuration (environment variables)
-------------------------------------
- ``SELF_REVIEW_ENABLED`` (default ``1``) — master switch.
- ``SELF_REVIEW_MODE_FILE`` (default ``~/.hermes/loop-state/self-review-mode.json``)
- ``SELF_REVIEW_SUFFIX`` (default ``0``) — enable the pre_llm_call suffix
  injection.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _truthy(raw: Any) -> bool:
    return (raw or "").lower() in ("1", "true", "yes", "on")


_scoring_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring.py")
_scoring_spec = importlib.util.spec_from_file_location("_selfreview_scoring", _scoring_path)
if _scoring_spec and _scoring_spec.loader:
    _scoring = importlib.util.module_from_spec(_scoring_spec)
    _scoring_spec.loader.exec_module(_scoring)
else:
    _scoring = None  # type: ignore[assignment]

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _config() -> Dict[str, Any]:
    """Re-read configuration each call so tests can flip env vars (mirrors
    compounding-loops)."""
    return {
        "enabled": _truthy(os.environ.get("SELF_REVIEW_ENABLED", "1")),
        "suffix_on": _truthy(os.environ.get("SELF_REVIEW_SUFFIX", "0")),
        "mode_file": Path(
            os.environ.get(
                "SELF_REVIEW_MODE_FILE",
                str(_HERMES_HOME / "loop-state" / "self-review-mode.json"),
            )
        ),
    }


# ---- speculator mode suffix ------------------------------------------------

_SPEC_SUFFIX = """\
[SELF-REVIEW SPECULATOR MODE]
Before finalizing, run a 20-second adversarial self-speculation: stepping into
the role of a hostile reviewer, enumerate the findings you WOULD raise against
this change right now — severity (nit/minor/major/blocker), a one-line
evidence citation, and the fix direction. Be unsentimental about your own work;
a false positive costs nothing, a missed blocker ships undetected. If you find
real blockers/majors, fix them before closeout. End the speculator mode card
with '[MODE RESET]' to return to builder mode.
"""

_MODE_CARD = {
    "roles": {
        "BUILDER": "producing the change; must not carry reviewer framing into new code",
        "SPECULATOR": "predicting findings a hostile reviewer would raise; must not edit",
        "REVIEWER": "post-hoc hostile review; must not edit (see adversarial-work-review)",
    },
    "rule": "Switch roles only via an explicit mode card. After a SPECULATOR/REVIEWER "
    "block, emit '[MODE RESET]' before writing more code, so builder mode starts "
    "clean (the 'reset optimizer state at mode switch' analog).",
}


# ---- mode isolation --------------------------------------------------------

def _read_mode() -> Dict[str, Any]:
    try:
        _mode_file = _config()["mode_file"]
        if _mode_file.exists():
            data = json.loads(_mode_file.read_text() or "{}")
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("self-review-mode: mode read failed: %s", exc)
    return {}


def _write_mode(mode: str) -> Dict[str, Any]:
    rec = _read_mode()
    rec["mode"] = mode
    try:
        _mode_file = _config()["mode_file"]
        _mode_file.parent.mkdir(parents=True, exist_ok=True)
        _mode_file.write_text(json.dumps(rec, indent=2, sort_keys=True))
    except OSError as exc:
        logger.debug("self-review-mode: mode write failed: %s", exc)
    return rec


def _has_mode_reset(text: str) -> bool:
    return bool(text and "[MODE RESET]" in text)


# ---- tools ----------------------------------------------------------------

def _self_review_handler(args: Optional[Dict[str, Any]] = None, **_kw) -> str:
    """Emit the speculator-mode suffix and set the on-disk mode marker."""
    if not _config()["enabled"]:
        return json.dumps({"enabled": False, "note": "self-review-mode disabled"})
    target = str((args or {}).get("target", ""))
    _write_mode("SPECULATOR")
    payload = {
        "action": "self_review",
        "mode": "SPECULATOR",
        "suffix": _SPEC_SUFFIX,
        "mode_isolation_card": _MODE_CARD,
        "target": target,
        "note": (
            "Predict findings BEFORE shipping. Emit '[MODE RESET]' after your "
            "speculation list to return to builder mode."
        ),
    }
    return json.dumps(payload, indent=2)


def _self_review_score_handler(args: Optional[Dict[str, Any]] = None, **_kw) -> str:
    """Score predicted vs actual findings via hard-gate alignment."""
    if not _config()["enabled"]:
        return json.dumps({"error": "self-review-mode disabled"})
    if _scoring is None:
        return json.dumps({"error": "scoring module failed to load"})
    a = args or {}
    gold = a.get("gold_findings") or []
    pred = a.get("predicted_findings") or []
    if not isinstance(gold, list) or not isinstance(pred, list):
        return json.dumps({"error": "gold_findings and predicted_findings must be lists"})
    # The tool consumes LLM-generated JSON; guard against malformed items
    # (bare strings, None, numbers) that would raise AttributeError in scoring.
    for item in list(gold) + list(pred):
        if not isinstance(item, dict):
            return json.dumps(
                {"error": f"each finding must be a dict with 'severity'/'evidence', got item of type {type(item).__name__}"}
            )
    try:
        score = _scoring.alignment_score(gold, pred)
        breakdown = _scoring.miss_breakdown(gold, pred)
        thrash = _scoring.detect_thrash(
            a.get("pass_new_finding_counts") or [],
            min_new=int(a.get("min_new", 1)),
            window=int(a.get("window", 2)),
        )
    except _scoring.InvalidSeverity as exc:
        return json.dumps({"error": f"invalid finding: {exc}"})
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"bad input: {exc}"})
    return json.dumps(
        {
            "alignment_score": round(score, 4),
            "miss_breakdown": breakdown,
            "thrash_detected": thrash,
        },
        indent=2,
    )


# ---- hooks ----------------------------------------------------------------

def _pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Opt-in: inject the speculator suffix as context on the next LLM call.

    Contract mirrors compounding-loops: ``(**kwargs) -> Optional[dict]``, a
    returned ``{"context": str}`` is merged into the user message for the
    upcoming call. Only fires when BOTH the master switch and
    ``SELF_REVIEW_SUFFIX`` are on and the user message is a build request.
    Default OFF because unconditional suffix injection on every call would
    degrade quality and fight the prompt cache; prefer the explicit
    ``self_review`` tool.
    """
    if not _config()["enabled"] or not _config()["suffix_on"]:
        return None
    user_message = kwargs.get("user_message") or ""
    if not isinstance(user_message, str):
        return None
    # Only nudge build-bound turns; don't inject into review/QA chatter.
    build_hint = any(k in user_message.lower() for k in (
        "build", "implement", "refactor", "scaffold", "write", "deploy",
    ))
    if not build_hint:
        return None
    return {"context": _SPEC_SUFFIX}


def register(ctx) -> None:
    if not _config()["enabled"]:
        logger.debug("self-review-mode: disabled")
        return
    register_tool = getattr(ctx, "register_tool", None)
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_tool):
        register_tool(
            name="self_review",
            toolset="review",
            schema={
                "name": "self_review",
                "description": (
                    "Enter SPECULATOR mode: predict (before shipping) the findings "
                    "a hostile reviewer would raise against the current change, "
                    "with severity + evidence + fix direction. Then emit "
                    "'[MODE RESET]' to return to builder mode."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Optional description of the work under review"},
                    },
                },
            },
            handler=_self_review_handler,
        )
        register_tool(
            name="self_review_score",
            toolset="review",
            schema={
                "name": "self_review_score",
                "description": (
                    "Score predicted self-review findings against actual findings "
                    "using hard-gate severity alignment + token-F1 + miss penalty. "
                    "Also reports a thrash (diminishing-returns) signal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gold_findings": {
                            "type": "array",
                            "description": "Actual findings: [{'severity','evidence'}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string"},
                                    "evidence": {"type": "string"},
                                },
                            },
                        },
                        "predicted_findings": {
                            "type": "array",
                            "description": "Predicted findings: [{'severity','evidence'}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string"},
                                    "evidence": {"type": "string"},
                                },
                            },
                        },
                        "pass_new_finding_counts": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "New findings per review pass (optional, for thrash detect)",
                        },
                        "min_new": {"type": "integer", "default": 1},
                        "window": {"type": "integer", "default": 2},
                    },
                    "required": ["gold_findings", "predicted_findings"],
                },
            },
            handler=_self_review_score_handler,
        )
    if callable(register_hook):
        register_hook("pre_llm_call", _pre_llm_call)
    logger.info(
        "self-review-mode: registered (suffix_injection=%s, mode_file=%s)",
        _config()["suffix_on"], _config()["mode_file"],
    )
