"""Self-assessment: Hermes's strength/limit model + task classifier.

The honest model: Hermes wins on memory/continuity, autonomy, breadth, ops,
self-improvement. Specialist / stateless harnesses win on narrow single-repo
coding, diff-centric IDE loops, cloud-sandboxed execution, and fresh-perspective
novelty. The classifier scores a task against these to decide who should do it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Hermes's own strengths (0-1). These are the honest self-assessment.
STRENGTHS: Dict[str, float] = {
    "memory_and_continuity": 0.95,
    "autonomy_and_scheduling": 0.95,
    "breadth_and_platforms": 0.90,
    "ops_and_infrastructure": 0.90,
    "self_improvement": 0.90,
    "narrow_single_repo_coding": 0.55,  # Claude Code / Codex win here
    "diff_centric_ide_loop": 0.50,      # Claude Code wins here
    "cloud_sandboxed_execution": 0.30,  # Codex cloud wins (has sandbox)
    "stateless_fresh_perspective": 0.40,  # stateless harness wins for novelty
}

# Specialist harness strength profiles (independent of Hermes's own).
SPECIALIST_STRENGTHS: Dict[str, Dict[str, float]] = {
    "claude_code": {
        "narrow_single_repo_coding": 0.92,
        "diff_centric_ide_loop": 0.90,
        "reasoning": 0.85,
        "memory_and_continuity": 0.65,   # codebase-aware but not cross-session
        "ops_and_infrastructure": 0.55,
        "stateless_fresh_perspective": 0.60,
    },
    "codex": {
        "narrow_single_repo_coding": 0.88,
        "cloud_sandboxed_execution": 0.95,
        "security_review": 0.85,
        "stateless_fresh_perspective": 0.85,  # no anchor by design
        "memory_and_continuity": 0.50,
        "ops_and_infrastructure": 0.50,
    },
    "opencode": {
        "narrow_single_repo_coding": 0.80,
        "second_opinion": 0.85,
        "diff_centric_ide_loop": 0.80,
        "stateless_fresh_perspective": 0.75,
        "memory_and_continuity": 0.55,
    },
}

# Task-type -> which strengths matter (weighted). Sum of weights per type = 1.
# "keep_self" weights emphasize Hermes's high-scoring strengths; "specialist"
# weights emphasize the areas where specialists win.
TASK_PROFILES: Dict[str, Dict[str, Any]] = {
    "narrow_coding": {
        "keep_self": {
            "narrow_single_repo_coding": 0.6,
            "diff_centric_ide_loop": 0.3,
            "memory_and_continuity": 0.1,
        },
        "specialist": {
            "narrow_single_repo_coding": 0.6,
            "diff_centric_ide_loop": 0.3,
            "reasoning": 0.1,
        },
        "specialist_name": "claude_code",
    },
    "security_review": {
        "keep_self": {
            "security_review": 0.5,
            "ops_and_infrastructure": 0.3,
            "memory_and_continuity": 0.2,
        },
        "specialist": {
            "security_review": 0.7,
            "cloud_sandboxed_execution": 0.3,
        },
        "specialist_name": "codex",
    },
    "sandboxed_execution": {
        "keep_self": {
            "cloud_sandboxed_execution": 0.7,
            "ops_and_infrastructure": 0.3,
        },
        "specialist": {
            "cloud_sandboxed_execution": 0.8,
            "stateless_fresh_perspective": 0.2,
        },
        "specialist_name": "codex",
    },
    "second_opinion": {
        "keep_self": {
            "stateless_fresh_perspective": 0.6,
            "memory_and_continuity": 0.2,
            "breadth_and_platforms": 0.2,
        },
        "specialist": {
            "second_opinion": 0.7,
            "stateless_fresh_perspective": 0.3,
        },
        "specialist_name": "opencode",
    },
    "ops_research": {
        "keep_self": {
            "ops_and_infrastructure": 0.5,
            "breadth_and_platforms": 0.3,
            "memory_and_continuity": 0.2,
        },
        "specialist": {
            "narrow_single_repo_coding": 0.5,
            "reasoning": 0.5,
        },
        "specialist_name": "claude_code",
    },
    "default": {
        "keep_self": {
            "ops_and_infrastructure": 0.4,
            "breadth_and_platforms": 0.3,
            "memory_and_continuity": 0.3,
        },
        "specialist": {
            "narrow_single_repo_coding": 0.5,
            "reasoning": 0.5,
        },
        "specialist_name": "claude_code",
    },
}

# Task-type keywords -> profile. First match wins (order matters).
_TASK_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("narrow_coding", ("function", "method", "refactor", "bug", "implement", "test", "pr", "commit", "diff")),
    ("security_review", ("security", "vulnerabil", "audit", "sandbox", "exploit", "cve", "solidity-auditor", "mythril", "slither")),
    ("sandboxed_execution", ("sandbox", "isolated", "cloud exec", "untrusted", "run in container")),
    ("second_opinion", ("opinion", "second set of eyes", "review this", "independent", "cross-check", "double-check")),
    ("ops_research", ("research", "deploy", "configure", "monitor", "backup", "incident", "oncall", "migrate", "upgrade", "crypto", "analysis")),
)


def classify_task(task: str) -> str:
    """Classify a task description into a task type."""
    t = (task or "").lower()
    for task_type, keywords in _TASK_KEYWORDS:
        if any(kw in t for kw in keywords):
            return task_type
    return "default"


def _score(task: str, profile: str, target: str) -> float:
    """Score a task against a strength profile using the task-type weights.

    ``target`` = "keep_self" (Hermes strengths) or "specialist" (harness strengths).
    Returns a weighted sum in [0, 1].
    """
    if target == "keep_self":
        return _score_self(task)
    task_type = classify_task(task)
    weights = TASK_PROFILES.get(task_type, TASK_PROFILES["default"]).get(target, {})
    src = SPECIALIST_STRENGTHS.get(profile, {})
    total = 0.0
    for strength, w in weights.items():
        total += w * src.get(strength, 0.0)
    return total


def _score_self(task: str) -> float:
    """Score a task against Hermes's own strengths only."""
    task_type = classify_task(task)
    weights = TASK_PROFILES.get(task_type, TASK_PROFILES["default"]).get("keep_self", {})
    total = 0.0
    for strength, w in weights.items():
        total += w * STRENGTHS.get(strength, 0.0)
    return total


def assess(task: str) -> Dict[str, Any]:
    """Return the full assessment for a task.

    Includes keep_self score, specialist-fit scores, the best specialist,
    and novelty/anchoring flags (filled by anchoring.py at the router layer).
    """
    task_type = classify_task(task)
    keep_self = _score_self(task)
    specialist_fit: Dict[str, float] = {}
    for name in SPECIALIST_STRENGTHS:
        specialist_fit[name] = _score(task, name, "specialist")

    best_specialist = max(specialist_fit, key=specialist_fit.get)
    return {
        "task": task,
        "task_type": task_type,
        "keep_self": round(keep_self, 3),
        "specialist_fit": {k: round(v, 3) for k, v in specialist_fit.items()},
        "best_specialist": best_specialist,
        "best_specialist_score": round(specialist_fit[best_specialist], 3),
        "specialist_name": TASK_PROFILES.get(task_type, TASK_PROFILES["default"])["specialist_name"],
    }


def should_route_to_specialist(task: str, win_margin: float = 0.15) -> Tuple[bool, str, Dict[str, Any]]:
    """Decide whether a specialist beats Hermes for this task.

    Returns (route, specialist_name, assessment). A specialist wins only if
    its best score exceeds keep_self by ``win_margin``.
    """
    a = assess(task)
    specialist_name = a["specialist_name"]
    spec_score = a["specialist_fit"].get(specialist_name, 0.0)
    route = spec_score > a["keep_self"] + win_margin
    return route, specialist_name, a