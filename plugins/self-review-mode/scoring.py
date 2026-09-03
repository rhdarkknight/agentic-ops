"""Self-review-mode scoring — the paper's reward-shaping mechanics, made pure.

Transfers from "Speculate While You Reason" (arXiv:2607.25816):

1. **Hard gate (tool-name gate analog).** The paper only reuses a speculated
   tool call when the *name* matches exactly; a wrong name gets zero. Here the
   hard gate is **severity classification**: a predicted finding whose severity
   does not exactly match the real finding earns zero credit — it would not have
   been "raised correctly" and therefore cannot be reused. This is anti-Goodhart:
   a self-review that launders a blocker down to a minor scores exactly as if it
   had said nothing.

2. **Partial-credit token-F1 (argument-F1 analog).** Once the hard gate passes,
   we score the *specificity* of the finding via token-F1 over its evidence
   tokens. A close-but-not-identical characterization earns partial credit.

3. **Miss penalty (name-conditioned reward's zero for missing keys).** Gold
   findings that no prediction matched (or matched with the wrong severity) are
   *misses*. Misses are penalized in proportion to their severity weight, so an
   over-lenient self-review (predicts nothing, or predicts everything as minor)
   scores negative — the exact failure the paper's reward design discourages.

4. **Thrash detector (the "penalty for repeated meaningless tool calls" analog).**
   Diminishing-return review passes — N consecutive passes each producing fewer
   than ``min_new`` *new* findings — are flagged for early termination instead of
   letting the loop grind.

All functions are pure and side-effect free; nothing here touches disk or the
message history.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

# Severity ladder must match the adversarial-work-review skill's taxonomy.
SEVERITIES = ("nit", "minor", "major", "blocker")
_SEVERITY_WEIGHT = {"nit": 0.25, "minor": 0.5, "major": 1.0, "blocker": 2.0}


class InvalidSeverity(ValueError):
    """Raised when a severity string is not in the known ladder."""


def validate_severity(severity: Optional[str]) -> str:
    """Normalize + validate a severity string against the ladder.

    Raises InvalidSeverity on unknown or missing values so callers can't
    silently inject a custom tier or pass a finding with no severity key —
    either would break the hard gate's exact-match semantics.
    """
    if not isinstance(severity, str):
        raise InvalidSeverity(
            f"severity must be a str, got {type(severity).__name__!r}"
        )
    s = severity.strip().lower()
    if s not in SEVERITIES:
        raise InvalidSeverity(f"unknown severity {severity!r}; expected one of {SEVERITIES}")
    return s


def severity_weight(severity: Optional[str]) -> float:
    """Return the weight used for miss-penalty and normalization.

    A blocker miss is worth 8x a nit miss (2.0 vs 0.25), so over-lenient
    self-reviews that drop blockers are punished far harder than ones that
    drop nits.
    """
    return _SEVERITY_WEIGHT[validate_severity(severity)]


def _tokens(text: str) -> List[str]:
    """Lowercase alphanumeric token list for F1 comparison.

    This is intentionally crude (no stemming/stopword removal) so the metric is
    deterministic and testable. It matches how the paper normalizes argument
    values to strings before computing token-F1.
    """
    if text is None:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_f1(gold: str, pred: str) -> float:
    """Harmonic-mean F1 over token multisets between gold and pred evidence.

    Returns 0.0 when either side is empty (a finding with no evidence carries no
    specificity to credit). Handles duplicates via Counter so repeating an
    important token isn't free.
    """
    from collections import Counter

    g = Counter(_tokens(gold))
    p = Counter(_tokens(pred))
    if not g or not p:
        return 0.0
    overlap = sum((g & p).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_match_for_gold(
    gold: Dict[str, str],
    preds: List[Dict[str, str]],
) -> Optional[float]:
    """Score the best admissible prediction for one gold finding.

    Admissibility = hard gate: predicted severity must exactly equal gold
    severity. Among admissible predictions, take the max token-F1 over evidence
    **that is strictly positive**. If every admissible prediction has F1 == 0
    (empty or disjoint evidence), the finding is a miss — naming the right
    severity with no real specificity conveys no usable information and must not
    escape the miss penalty (anti-Goodhart: don't let guessing severities score
    near-zero instead of negative).

    Returns the best positive F1, or None if the finding is a miss.
    """
    best: float | None = None
    for p in preds:
        if validate_severity(p.get("severity")) != validate_severity(gold.get("severity")):
            continue  # hard gate: wrong severity = not admissible, no credit
        f1 = token_f1(gold.get("evidence", ""), p.get("evidence", ""))
        if f1 > 0 and (best is None or f1 > best):
            best = f1
    return best


def alignment_score(
    gold_findings: Sequence[Dict[str, str]],
    predicted_findings: Sequence[Dict[str, str]],
) -> float:
    """Score how well a self-review predicted the actual findings.

    Each gold finding contributes ``+weight * best_admissible_f1`` if it has at
    least one severity-matching prediction, or ``-weight`` if it is a miss.
    The total is normalized by the sum of gold weights, giving a score in
    ``[-1.0, 1.0]``:

    - ``1.0``  — every gold finding predicted at the right severity with F1=1.
    - ``0.0``  — perfect indifference: matched findings' credit exactly offsets
      the misses (e.g. half predicted perfectly, half missed).
    - ``-1.0`` — nothing matched (over-lenient / empty / all-wrong-severity).

    Raises InvalidSeverity on a bad severity string in either sequence.
    """
    golds = list(gold_findings)
    preds = list(predicted_findings)
    if not golds:
        # Nothing to predict against: a self-review of an empty gold set is
        # trivially aligned only if it also predicted nothing. Predicting
        # findings for work that has none is harmless but not credit-worthy.
        return 1.0 if not preds else 0.0

    total_weight = 0.0
    acc = 0.0
    for g in golds:
        w = severity_weight(g.get("severity"))
        total_weight += w
        best = _best_match_for_gold(g, preds)
        if best is None:
            acc -= w  # miss penalty
        else:
            acc += w * best
    if total_weight == 0.0:
        return 0.0
    return max(-1.0, min(1.0, acc / total_weight))


def miss_breakdown(
    gold_findings: Sequence[Dict[str, str]],
    predicted_findings: Sequence[Dict[str, str]],
) -> Dict[str, int]:
    """Count gold findings by miss-vs-hit, for diagnostics.

    Returns ``{"hits": int, "misses": int, "blocker_misses": int,
    "major_misses": int}`` so the caller can report *why* an alignment score is
    low (anti-Goodhart transparency).
    """
    golds = list(gold_findings)
    preds = list(predicted_findings)
    hits = misses = blocker_misses = major_misses = 0
    for g in golds:
        best = _best_match_for_gold(g, preds)
        if best is None:
            misses += 1
            sev = validate_severity(g.get("severity"))
            if sev == "blocker":
                blocker_misses += 1
            elif sev == "major":
                major_misses += 1
        else:
            hits += 1
    return {
        "hits": hits,
        "misses": misses,
        "blocker_misses": blocker_misses,
        "major_misses": major_misses,
    }


def detect_thrash(
    pass_new_finding_counts: Sequence[int],
    min_new: int = 1,
    window: int = 2,
) -> bool:
    """True when the last ``window`` review passes each found < ``min_new`` NEW
    findings — i.e., diminishing returns / meaningless repetition.

    Mirrors the paper's penalty for repeated tool calls that make no progress:
    if the loop keeps re-running but stops surfacing anything new, it is
    thrashing and should terminate. Requires at least ``window`` passes to be
    present to avoid flagging a fresh loop.
    """
    if min_new < 0:
        raise ValueError("min_new must be >= 0")
    if window < 1:
        raise ValueError("window must be >= 1")
    counts = list(pass_new_finding_counts)
    if len(counts) < window:
        return False
    return all(c < min_new for c in counts[-window:])


def classify_from_text(text: str) -> str:
    """Heuristic severity classification from free-text (auxiliary).

    Returns the highest severity mentioned, defaulting to ``minor`` when none
    is found. Useful for normalizing dicts that forgot a ``severity`` key.
    NOT the authority — the authority is the hard gate on the explicit field.
    """
    t = (text or "").lower()
    for sev in ("blocker", "major"):
        if sev in t or f"{sev}s" in t:
            return sev
    if "nit" in t or "style" in t:
        return "nit"
    return "minor"
