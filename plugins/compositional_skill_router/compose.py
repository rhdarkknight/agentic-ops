"""DAG composition: order retrieved skills by dependencies.

Each sub-task → top-k candidates. We must choose ONE skill per sub-task,
then order all chosen skills respecting dependencies.

Selection scoring (from SkillWeaver Eq. 3):
    α · sim(subtask, skill) + (1-α) · mean_compatibility(neighbors)

α = 0.5 by default. Compatibility = average cosine similarity between
this skill's embedding and its predecessors' embeddings (output→input
overlap proxy — assumes semantically related skills share vocabulary).

Dependency detection:
    1. Explicit sequence markers from decompose.SubTask.marker ("then",
       "and", "numbered") → topological order = sub-task order
    2. Fallback: greedy chain by pairwise cosine compatibility

Result: ordered list of (subtask, chosen_skill, alt_candidates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retrieve import RetrievedSkill, _get_model, _load_cache


@dataclass
class PlanStep:
    sub_task: str
    sub_task_idx: int
    chosen: RetrievedSkill
    candidates: list[RetrievedSkill]
    depends_on: list[int] = field(default_factory=list)


@dataclass
class Plan:
    steps: list[PlanStep]
    score: float
    alpha: float
    notes: str = ""


# Markers that indicate the user already specified an ordering — when any
# sub-task carries one, we preserve sub-task order and build a chain.
_EXPLICIT_ORDER_MARKERS = {"then", "numbered", "and", "comma", "sad-refined"}


def _topological_order(sub_tasks: list, chosen: list[RetrievedSkill]) -> list[int]:
    """Return plan indices (output positions) in dependency-respecting order.

    If any sub-task carries an explicit sequence marker, sub-task order is
    the order — we simply return ``range(len(chosen))``.

    Otherwise fall back to greedy chain by pairwise cosine compatibility:
    start with the first (highest retrieval score), then pick the next
    candidate whose mean cosine with already-selected is highest.
    """
    if not chosen:
        return []
    markers = {st.marker for st in sub_tasks}
    if markers & _EXPLICIT_ORDER_MARKERS:
        return list(range(len(chosen)))

    # Greedy chain — encode chosen skills' embeddings once.
    model = _get_model()
    texts = [f"{c.name}: {c.description}" for c in chosen]
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    compat = vecs @ vecs.T  # (n, n) cosine since L2-normalized

    order = [0]
    remaining = set(range(1, len(chosen)))
    while remaining:
        avg_scores = {
            cand: float(np.mean([compat[cand][o] for o in order]))
            for cand in remaining
        }
        best = max(avg_scores, key=avg_scores.get)
        order.append(best)
        remaining.discard(best)
    return order


def compose(
    sub_tasks: list,
    candidates_per_subtask: list[list[RetrievedSkill]],
    alpha: float = 0.5,
) -> Plan:
    """Pick one skill per sub-task, order into a DAG, return plan.

    Defensive: if sub_tasks and candidates_per_subtask have mismatched
    lengths, we zip to the shorter length to avoid IndexError. Empty
    candidates for a sub-task are skipped (no step emitted).
    """
    if not sub_tasks or not candidates_per_subtask:
        return Plan(steps=[], score=0.0, alpha=alpha, notes="empty input")

    # Defensive: align lengths so we never index out of range.
    n = min(len(sub_tasks), len(candidates_per_subtask))
    sub_tasks = sub_tasks[:n]
    candidates_per_subtask = candidates_per_subtask[:n]

    # 1. Pick one skill per sub-task. For step 0 use pure retrieval sim.
    #    For step i>0 use α·sim + (1-α)·mean_cosine_to_prior_chosen.
    #    Pre-encode ALL candidate vectors in one batch (O(N+K) not O(N²)).
    model = _get_model()
    chosen: list[RetrievedSkill] = []
    chosen_vecs: list[np.ndarray] = []  # parallel to chosen

    for i, cands in enumerate(candidates_per_subtask):
        if not cands:
            continue
        cand_texts = [f"{c.name}: {c.description}" for c in cands]
        cand_vecs = model.encode(
            cand_texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        if not chosen:
            # First step: pure retrieval similarity
            best_idx = max(range(len(cands)), key=lambda j: cands[j].score)
        else:
            # α·sim + (1-α)·mean cosine to all already-chosen
            prior = np.stack(chosen_vecs, axis=0)  # (n_prior, d)
            mean_compat = (cand_vecs @ prior.T).mean(axis=1)  # (n_cands,)
            scores = alpha * np.array([c.score for c in cands]) + (1 - alpha) * mean_compat
            best_idx = int(np.argmax(scores))

        chosen.append(cands[best_idx])
        chosen_vecs.append(cand_vecs[best_idx])

    if not chosen:
        return Plan(steps=[], score=0.0, alpha=alpha, notes="no candidates")

    # 2. Order chosen skills into a DAG.
    order = _topological_order(sub_tasks, chosen)

    # 3. Emit PlanSteps. depends_on refers to PLAN positions (output idx),
    #    not sub-task indices — so when we reorder, deps track the plan.
    steps: list[PlanStep] = []
    for out_idx, src_idx in enumerate(order):
        steps.append(
            PlanStep(
                sub_task=sub_tasks[src_idx].text,
                sub_task_idx=sub_tasks[src_idx].index,
                chosen=chosen[src_idx],
                candidates=candidates_per_subtask[src_idx],
                depends_on=[] if out_idx == 0 else [out_idx - 1],
            )
        )

    plan_score = float(np.mean([s.chosen.score for s in steps]))
    notes = (
        f"decomposed into {len(sub_tasks)} sub-task(s), "
        f"retrieved {sum(len(c) for c in candidates_per_subtask)} candidates, "
        f"composed into {len(steps)}-step plan"
    )
    return Plan(steps=steps, score=plan_score, alpha=alpha, notes=notes)