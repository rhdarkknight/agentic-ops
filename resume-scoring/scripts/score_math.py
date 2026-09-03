#!/usr/bin/env python3
"""Deterministic caps for a role evaluation JSON."""
from __future__ import annotations

from typing import Any


class ScoreError(ValueError):
    pass


def finalize(evaluation: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    scores = evaluation.get("scores")
    if not isinstance(scores, dict):
        raise ScoreError("evaluation.scores must be an object")
    category_total = 0.0
    category_max = 0
    capped: list[str] = []
    missing: list[str] = []
    normalized: dict[str, Any] = {}
    for cat in role["categories"]:
        key = cat["key"]
        cap = int(cat["max"])
        category_max += cap
        raw_entry = scores.get(key)
        if not isinstance(raw_entry, dict) or "score" not in raw_entry:
            missing.append(key)
            normalized[key] = {"score": 0.0, "max": cap, "evidence": "missing"}
            continue
        try:
            raw = float(raw_entry["score"])
        except (TypeError, ValueError) as exc:
            raise ScoreError(f"{key}.score is not a number") from exc
        used = min(max(raw, 0.0), float(cap))
        if used != raw:
            capped.append(f"{key}:{raw}->{used}")
        evidence = str(raw_entry.get("evidence") or "").strip()
        if not evidence:
            raise ScoreError(f"{key}.evidence is empty")
        normalized[key] = {"score": used, "max": cap, "evidence": evidence}
        category_total += used

    extra = sorted(set(scores) - {c["key"] for c in role["categories"]})
    bonus_in = evaluation.get("bonus_points") or {}
    try:
        bonus_raw = float(bonus_in.get("total") or 0)
    except (TypeError, ValueError) as exc:
        raise ScoreError("bonus_points.total is not a number") from exc
    bonus_max = int(role.get("bonus_max") or 0)
    bonus = min(max(bonus_raw, 0.0), float(bonus_max))
    deductions_in = evaluation.get("deductions") or {}
    try:
        deductions = max(float(deductions_in.get("total") or 0), 0.0)
    except (TypeError, ValueError) as exc:
        raise ScoreError("deductions.total is not a number") from exc
    final = category_total + bonus - deductions
    min_final = float(role.get("min_final_score", -20))
    max_final = float(role.get("max_final_score", category_max + bonus_max))
    clamped = min(max(final, min_final), max_final)
    return {
        "scores": normalized,
        "bonus_points": {
            "total": bonus,
            "breakdown": str(bonus_in.get("breakdown") or ""),
        },
        "deductions": {
            "total": deductions,
            "reasons": str(deductions_in.get("reasons") or ""),
        },
        "key_strengths": list(evaluation.get("key_strengths") or []),
        "areas_for_improvement": list(evaluation.get("areas_for_improvement") or []),
        "category_total": category_total,
        "category_max": category_max,
        "final": clamped,
        "capped": capped,
        "missing_categories": missing,
        "extra_categories": extra,
        "cutoff_note": "Rank only. Do not auto-reject.",
    }
