"""Compositional Skill Router — Hermes plugin entry point.

Routes a user query to an *ordered set* of skills via:
  1. Decompose (heuristic + SAD feedback)
  2. Retrieve (per-sub-task, top-k from SRA embedding cache)
  3. Compose (DAG, dependency-respecting)

Tools exposed (via register(ctx)):
  - skill_route_compositional: route a query to a multi-skill plan
  - skill_route_eval:          run 30-query CompSkillBench-style eval
  - skill_route_cache_stats:   inspect the SRA cache we reuse

No core files modified. Zero-cost fallback: if SRA cache missing,
returns helpful error pointing to `sra_skill_router`.

Import pattern: each handler uses _import() so it works whether the
plugin is loaded flat (via sys.path injection during tests) OR
packaged (via hermes_plugins.compositional_skill_router namespace).
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_NAME = "compositional_skill_router"
PLUGIN_VERSION = "0.1.0"


def _import(name: str):
    """Import a sibling module from this plugin directory.

    Works under both flat layout (tests: `sys.path.insert(0, PLUGIN_DIR)`)
    and packaged layout (Hermes loader: `hermes_plugins.compositional_skill_router.X`).
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        # Packaged layout — same package as this __init__.py
        package_name = __package__ or "hermes_plugins.compositional_skill_router"
        return importlib.import_module(f"{package_name}.{name}")


# ─── Tool handlers ──────────────────────────────────────────────────


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _plan_to_dict(plan) -> dict:
    return {
        "score": plan.score,
        "alpha": plan.alpha,
        "notes": plan.notes,
        "steps": [
            {
                "index": i,
                "sub_task": s.sub_task,
                "depends_on": s.depends_on,
                "chosen": {
                    "name": s.chosen.name,
                    "category": s.chosen.category,
                    "score": round(s.chosen.score, 4),
                },
                "alternates": [
                    {
                        "name": c.name,
                        "category": c.category,
                        "score": round(c.score, 4),
                    }
                    for c in s.candidates[1:]
                ],
            }
            for i, s in enumerate(plan.steps)
        ],
    }


def _sad_to_dict(result) -> dict:
    return {
        "query": result.query,
        "iterations": result.iterations,
        "converged": result.converged,
        "n_initial_subtasks": len(result.initial_subtasks),
        "n_final_subtasks": len(result.final_subtasks),
        "plan": _plan_to_dict(result.plan),
    }


def _route_handler(args: dict, **_) -> str:
    route_fn = _import("sad").route

    query = (args.get("query") or "").strip()
    if not query:
        return _json({"success": False, "error": "query is required"})
    try:
        top_k = int(args.get("top_k") or 3)
        max_iterations = int(args.get("max_iterations") or 2)
        threshold = float(args.get("threshold") or 0.6)
        alpha = float(args.get("alpha") or 0.5)
    except (TypeError, ValueError) as e:
        return _json({"success": False, "error": f"invalid parameter: {e}"})

    if top_k < 1 or top_k > 50:
        return _json({"success": False, "error": "top_k must be 1..50"})
    if not 0.0 <= alpha <= 1.0:
        return _json({"success": False, "error": "alpha must be in [0, 1]"})

    t0 = time.time()
    try:
        result = route_fn(
            query,
            top_k=top_k,
            max_iterations=max_iterations,
            threshold=threshold,
            alpha=alpha,
        )
    except FileNotFoundError as e:
        return _json({
            "success": False,
            "error": str(e),
            "hint": "Run sra_skill_router to build the embedding cache first.",
        })
    except (RuntimeError, ValueError, KeyError) as e:
        # Model load failure, malformed cache, missing skill metadata
        logger.exception("[CSR] route failed")
        return _json({"success": False, "error": f"routing failed: {e}"})
    elapsed_ms = int((time.time() - t0) * 1000)
    return _json({
        "success": True,
        "elapsed_ms": elapsed_ms,
        "result": _sad_to_dict(result),
        "hint": "Use skill_view(name) on each chosen skill before acting. The plan respects dependencies.",
    })


def _eval_handler(args: dict, **_) -> str:
    sad_mod = _import("sad")
    evalset_mod = _import("evalset")
    route = sad_mod.route
    EVAL_SET = evalset_mod.EVAL_SET
    by_difficulty = evalset_mod.by_difficulty

    difficulty = (args.get("difficulty") or "all").lower()
    try:
        limit = int(args.get("limit") or 30)
    except (TypeError, ValueError):
        limit = 30
    if difficulty == "all":
        cases = EVAL_SET[:limit]
    else:
        cases = by_difficulty().get(difficulty, [])[:limit]

    if not cases:
        return _json({"success": False, "error": f"no cases for difficulty={difficulty}"})

    metrics = {
        "DA": 0,
        "DA_pm1": 0,
        "SR_at_1": 0,
        "SR_at_3": 0,
        "CatR_at_3": 0,
        "PlanEM": 0,
        "n_cases": len(cases),
    }
    case_results = []

    for case in cases:
        try:
            result = route(case["query"], top_k=3, max_iterations=2)
        except Exception as e:  # noqa: BLE001 — eval should keep going
            case_results.append({"query": case["query"][:60], "error": str(e)})
            continue

        n_pred = len(result.final_subtasks)
        n_true = case["n_subtasks"]
        if n_pred == n_true:
            metrics["DA"] += 1
        if abs(n_pred - n_true) <= 1:
            metrics["DA_pm1"] += 1

        all_retrieved_names = {c.name for cands in result.final_candidates for c in cands}
        all_retrieved_cats = {c.category for cands in result.final_candidates for c in cands}
        gt_skills = set(case["ground_truth_skills"])
        gt_cats = set(case["ground_truth_categories"])

        if gt_skills & all_retrieved_names:
            metrics["SR_at_1"] += 1
        if gt_skills and gt_skills.issubset(all_retrieved_names):
            metrics["SR_at_3"] += 1
        if gt_cats and gt_cats.issubset(all_retrieved_cats):
            metrics["CatR_at_3"] += 1

        chosen_names = [s.chosen.name for s in result.plan.steps]
        if chosen_names[: len(gt_skills)] == list(case["ground_truth_skills"])[: len(chosen_names)]:
            metrics["PlanEM"] += 1

        case_results.append({
            "query": case["query"][:60],
            "n_subtasks_pred": n_pred,
            "n_subtasks_true": n_true,
            "chosen": chosen_names,
            "expected": case["ground_truth_skills"],
        })

    n = max(metrics["n_cases"], 1)
    summary = {k: round(v / n, 3) if k != "n_cases" else v for k, v in metrics.items()}
    return _json({
        "success": True,
        "difficulty": difficulty,
        "summary": summary,
        "case_results": case_results,
    })


def _stats_handler(args: dict, **_) -> str:
    return _json({"success": True, "cache": _import("retrieve").cache_stats()})


# ─── Lifecycle hooks ────────────────────────────────────────────────


def _on_session_start(session_id: str = "", **_) -> None:
    logger.info("[CSR] session=%s", (session_id or "")[:10])


def _on_session_end(session_data: dict = None, **_) -> None:
    logger.info("[CSR] session end")


# ─── Plugin entry point (Hermes loader contract) ───────────────────


def register(ctx) -> None:
    """Register hooks, tools, and slash commands with Hermes.

    This is the contract Hermes' plugin loader expects:
      - plugin.yaml manifest declares provides_tools / provides_hooks
      - __init__.py exposes `register(ctx)` (NOT `init()`)
      - tools register via ctx.register_tool(...)
      - hooks register via ctx.register_hook(...)
      - opt-in via `plugins.enabled` in ~/.hermes/config.yaml
    """
    logger.info("Registering %s v%s", PLUGIN_NAME, PLUGIN_VERSION)

    # Session lifecycle hooks (optional, mostly observability)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)

    # Warm the embedding model at register time so first route() call
    # doesn't pay the ~1-2s model-load latency. Non-fatal if it fails
    # (e.g. retrieve module not on sys.path when loaded via the
    # hermes_plugins namespace — first tool call will warm lazily).
    try:
        _import("retrieve")._get_model()
        logger.info("[CSR] embedding model warmed")
    except Exception as e:  # noqa: BLE001
        logger.debug("[CSR] model warm-up skipped (lazy on first call): %s", e)

    # Tools
    ctx.register_tool(
        name="skill_route_compositional",
        toolset="csr",
        schema={
            "name": "skill_route_compositional",
            "description": (
                "Decompose a complex query into atomic sub-tasks, retrieve "
                "top-k skills per sub-task from the SRA embedding cache, and "
                "compose an ordered DAG plan with dependencies. Returns the "
                "plan with chosen skills and alternates. Use this BEFORE "
                "calling skill_view() when the request chains multiple skills "
                "('build X then deploy Y then monitor Z', numbered steps, "
                "comma-separated actions)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Complex multi-skill request to route.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Candidates per sub-task (1-50).",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 5,
                        "description": "SAD feedback iterations.",
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.6,
                        "description": "SAD Jaccard convergence threshold (0-1).",
                    },
                    "alpha": {
                        "type": "number",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Relevance vs compatibility weight (0=compat, 1=relevance).",
                    },
                },
                "required": ["query"],
            },
        },
        handler=_route_handler,
        description="Route complex query to an ordered multi-skill DAG plan",
        emoji="🧭",
    )

    ctx.register_tool(
        name="skill_route_eval",
        toolset="csr",
        schema={
            "name": "skill_route_eval",
            "description": (
                "Run the 30-query CompSkillBench-style eval set and report "
                "DA, DA±1, SR@1, SR@3, CatR@3, PlanEM metrics. Useful for "
                "verifying the router after skill-library changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard", "all"],
                        "default": "all",
                        "description": "Eval difficulty bucket to run.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Max cases to run.",
                    },
                },
            },
        },
        handler=_eval_handler,
        description="Run 30-query compositional skill routing benchmark",
        emoji="📊",
    )

    ctx.register_tool(
        name="skill_route_cache_stats",
        toolset="csr",
        schema={
            "name": "skill_route_cache_stats",
            "description": (
                "Inspect the SRA embedding cache this plugin reuses "
                "(~/.hermes/sra_cache/). Reports skill count, embedding "
                "dimension, model name, and cache path."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_stats_handler,
        description="Show SRA embedding cache stats",
        emoji="📦",
    )

    logger.info(
        "[CSR] registered 3 tools, 2 hooks (skill_route_compositional, "
        "skill_route_eval, skill_route_cache_stats)"
    )