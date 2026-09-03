"""SAD: Skill-Aware Decomposition iterative refinement.

Per SkillWeaver Algorithm 1: feed retrieved skills back into the decomposer
to align decomposition vocabulary with the actual skill library.

Convergence criterion: Jaccard similarity of top-H skill sets between
iterations exceeds threshold τ (default 0.6).

Without an LLM decomposer (we use heuristics), SAD manifests as:
    - Re-decompose using retrieved skills' top categories as additional
      splitting hints.
    - If first-pass candidates all share the same top category, split the
      query on that category's vocabulary to surface different skills.
    - If decomposition converged (same skills returned), stop.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from .decompose import SubTask, _split_on_markers as _dec_split, decompose
from .retrieve import RetrievedSkill, retrieve_batch
from .compose import Plan, compose

logger = logging.getLogger(__name__)


@dataclass
class SADResult:
    query: str
    iterations: int
    converged: bool
    initial_subtasks: list[SubTask]
    final_subtasks: list[SubTask]
    initial_candidates: list[list[RetrievedSkill]]
    final_candidates: list[list[RetrievedSkill]]
    plan: Plan


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _subtask_texts_equal(a: list[SubTask], b: list[SubTask]) -> bool:
    """Length-and-text equality. Avoids the zip-truncation pitfall
    where different-length lists compare 'equal' on their prefix."""
    return len(a) == len(b) and all(x.text == y.text for x, y in zip(a, b))


def _refine_with_hints(query: str, candidates: list[list[RetrievedSkill]]) -> list[SubTask]:
    """Second-pass decomposition that uses retrieval signal as a hint.

    Refinement rules (in priority order):
      1. If first pass decomposed into 1 sub-task AND candidates cluster
         into multiple categories → force-splits the query on commas /
         " and " to surface skills from the missing categories.
      2. If first pass decomposed into N≥2 sub-tasks AND candidates all
         cluster into one category → trigger an "add comma-split" attempt
         to find cross-category skills.
      3. Otherwise → return the original decomposition (converged).
    """
    cats = Counter()
    for cands in candidates:
        for c in cands:
            cats[c.category] += 1
    if not cats:
        return decompose(query)

    initial = decompose(query)
    top_cat, top_count = cats.most_common(1)[0]
    total = sum(cats.values())
    single_category_dominance = top_count >= total * 0.6 and len(cats) >= 2

    # Rule 1: under-decomposed + multi-category signal → aggressive split.
    if len(initial) <= 1 and len(cats) >= 2:
        forced = _dec_split(query, r",\s+")
        if forced and len(forced) >= 2:
            return [SubTask(p, i, "sad-refined") for i, p in enumerate(forced)]
        forced = _dec_split(query, r"\s+and\s+")
        if forced and len(forced) >= 2:
            return [SubTask(p, i, "sad-refined") for i, p in enumerate(forced)]

    # Rule 2: already decomposed but all candidates cluster in one category.
    if len(initial) >= 2 and single_category_dominance:
        forced = _dec_split(query, r",\s+")
        if forced and len(forced) > len(initial):
            return [SubTask(p, i, "sad-refined") for i, p in enumerate(forced)]

    return initial


def route(
    query: str,
    top_k: int = 3,
    max_iterations: int = 2,
    threshold: float = 0.6,
    alpha: float = 0.5,
) -> SADResult:
    """End-to-end compositional routing with SAD feedback.

    1. Decompose query (heuristic pass 0)
    2. Retrieve top-k per sub-task
    3. Compose plan
    4. Check convergence (Jaccard on top-H skill sets)
    5. If not converged and iter < max: refine decomposition with
       category hints from retrieval, loop.
    6. Return final plan.
    """
    if not query or not query.strip():
        return SADResult(
            query=query,
            iterations=0,
            converged=True,
            initial_subtasks=[],
            final_subtasks=[],
            initial_candidates=[],
            final_candidates=[],
            plan=Plan(steps=[], score=0.0, alpha=alpha, notes="empty query"),
        )

    initial_subs = decompose(query)
    if not initial_subs:
        return SADResult(
            query=query,
            iterations=0,
            converged=True,
            initial_subtasks=[],
            final_subtasks=[],
            initial_candidates=[],
            final_candidates=[],
            plan=Plan(steps=[], score=0.0, alpha=alpha, notes="decompose returned empty"),
        )

    # Iteration 0
    cand_lists = retrieve_batch([s.text for s in initial_subs], top_k=top_k)
    skill_set_prev = {c.name for cands in cand_lists for c in cands}

    final_subs = initial_subs
    final_cands = cand_lists
    converged = False
    iters = 1

    for i in range(1, max_iterations):
        refined = _refine_with_hints(query, cand_lists)
        if _subtask_texts_equal(refined, initial_subs):
            # Refiner couldn't find a better decomposition → converged.
            converged = True
            break
        new_cands = retrieve_batch([s.text for s in refined], top_k=top_k)
        skill_set_new = {c.name for cands in new_cands for c in cands}
        j = _jaccard(skill_set_prev, skill_set_new)
        if j > threshold:
            final_subs = refined
            final_cands = new_cands
            converged = True
            iters = i + 1
            break
        skill_set_prev = skill_set_new
        final_subs = refined
        final_cands = new_cands
        iters = i + 1

    plan = compose(final_subs, final_cands, alpha=alpha)
    return SADResult(
        query=query,
        iterations=iters,
        converged=converged,
        initial_subtasks=initial_subs,
        final_subtasks=final_subs,
        initial_candidates=cand_lists,
        final_candidates=final_cands,
        plan=plan,
    )