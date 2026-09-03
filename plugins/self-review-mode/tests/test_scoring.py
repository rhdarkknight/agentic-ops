"""Tests for self-review-mode scoring — the hard-gate alignment + thrash logic.

Run from anywhere: resolves the plugin dir from this file's location so the
suite works both inside the agent tree and standalone under
~/.hermes/plugins/self-review-mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def scoring():
    spec = importlib.util.spec_from_file_location(
        "plugins.self_review_scoring", _PLUGIN_DIR / "scoring.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- validate_severity ------------------------------------------------------

def test_validate_severity_normalizes(scoring):
    assert scoring.validate_severity(" BLOCKER ") == "blocker"
    assert scoring.validate_severity("Major") == "major"


def test_validate_severity_rejects_unknown(scoring):
    with pytest.raises(scoring.InvalidSeverity):
        scoring.validate_severity("critical")  # 'critical' is NOT in the ladder
    with pytest.raises(scoring.InvalidSeverity):
        scoring.validate_severity("")
    with pytest.raises(scoring.InvalidSeverity):
        scoring.validate_severity(None)
    with pytest.raises(scoring.InvalidSeverity):
        scoring.validate_severity(42)


def test_severity_weight_blocker_greater_than_nit(scoring):
    assert scoring.severity_weight("blocker") > scoring.severity_weight("major") > \
        scoring.severity_weight("minor") > scoring.severity_weight("nit")


# --- token_f1 ---------------------------------------------------------------

def test_token_f1_identical(scoring):
    assert scoring.token_f1("null deref in parse", "null deref in parse") == 1.0


def test_token_f1_disjoint(scoring):
    assert scoring.token_f1("a b c", "x y z") == 0.0


def test_token_f1_partial(scoring):
    # overlap {null,deref}=2 ; p={null,deref,check}=3 ; g={null,deref,parse}=3
    f1 = scoring.token_f1("null deref parse", "null deref check")
    assert f1 == pytest.approx(2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3), abs=1e-9)
    assert f1 == pytest.approx(0.6666666, abs=1e-6)


def test_token_f1_empty_side_returns_zero(scoring):
    assert scoring.token_f1("", "foo") == 0.0
    assert scoring.token_f1("foo", None) == 0.0
    assert scoring.token_f1("", "") == 0.0


def test_token_f1_handles_duplicates(scoring):
    # g has 'x' twice, p has 'x' once -> recall 0.5
    assert scoring.token_f1("x x", "x") == pytest.approx(2 * 1 * 0.5 / (1 + 0.5), abs=1e-9)


def test_token_f1_case_and_punct_insensitive(scoring):
    assert scoring.token_f1("Null-Deref!", "null deref") == 1.0


# --- alignment_score --------------------------------------------------------

PERFECT_GOLD = [
    {"severity": "blocker", "evidence": "null deref in parse"},
    {"severity": "major", "evidence": "race on counter"},
]
PERFECT_PRED = [
    {"severity": "blocker", "evidence": "null deref in parse"},
    {"severity": "major", "evidence": "race on counter"},
]


def test_alignment_perfect(scoring):
    assert scoring.alignment_score(PERFECT_GOLD, PERFECT_PRED) == 1.0


def test_alignment_empty_pred_all_miss(scoring):
    # predicting nothing = every gold is a miss = -1.0 (over-lenient)
    assert scoring.alignment_score(PERFECT_GOLD, []) == -1.0


def test_alignment_wrong_severity_all_miss(scoring):
    # hard gate: downgrading blocker->minor & major->minor = no admissible match
    pred = [
        {"severity": "minor", "evidence": "null deref in parse"},
        {"severity": "minor", "evidence": "race on counter"},
    ]
    assert scoring.alignment_score(PERFECT_GOLD, pred) == -1.0


def test_alignment_right_severity_no_evidence_is_a_miss(scoring):
    # Anti-Goodhart: predicting the right severity with EMPTY evidence must NOT
    # dodge the miss penalty (token-F1 = 0 -> not a positive match). Guessing
    # every severity with no specificity should score negative, not ~0.
    gold = [{"severity": "blocker", "evidence": "null deref in parse"}]
    pred_empty_evidence = [{"severity": "blocker", "evidence": ""}]
    assert scoring.alignment_score(gold, pred_empty_evidence) == -1.0

    # Same idea: right severity but totally disjoint evidence = miss too.
    pred_disjoint = [{"severity": "blocker", "evidence": "zzz qqq"}]
    assert scoring.alignment_score(gold, pred_disjoint) == -1.0


def test_alignment_miss_blocker_is_worse_than_miss_nit(scoring):
    # missing a blocker (w=2.0) costs far more than a nit; the matched minor
    # (w=0.5) partially offsets. acc = -2.0 + 0.5 = -1.5 / total 2.5 = -0.6.
    gold_blocker = [{"severity": "blocker", "evidence": "oops"}, {"severity": "minor", "evidence": "style"}]
    pred_hit_nit_only = [
        {"severity": "minor", "evidence": "style"},
    ]
    score_nit = scoring.alignment_score(gold_blocker, pred_hit_nit_only)
    assert score_nit == pytest.approx(-1.5 / 2.5, abs=1e-9)


def test_alignment_partial_credit_on_match(scoring):
    gold = [{"severity": "blocker", "evidence": "null deref parse"}]
    pred = [{"severity": "blocker", "evidence": "null deref check"}]  # F1 = 2/3
    assert scoring.alignment_score(gold, pred) == pytest.approx(2 / 3, abs=1e-9)


def test_alignment_empty_gold(scoring):
    assert scoring.alignment_score([], []) == 1.0
    assert scoring.alignment_score([], [{"severity": "minor", "evidence": "x"}]) == 0.0


def test_alignment_invalid_severity_raises(scoring):
    with pytest.raises(scoring.InvalidSeverity):
        scoring.alignment_score(
            [{"severity": "critical", "evidence": "x"}],
            [{"severity": "blocker", "evidence": "x"}],
        )


# --- miss_breakdown ---------------------------------------------------------

def test_miss_breakdown_counts(scoring):
    breakdown = scoring.miss_breakdown(PERFECT_GOLD, PERFECT_PRED)
    assert breakdown == {"hits": 2, "misses": 0, "blocker_misses": 0, "major_misses": 0}

    breakdown2 = scoring.miss_breakdown(PERFECT_GOLD, [])
    assert breakdown2 == {"hits": 0, "misses": 2, "blocker_misses": 1, "major_misses": 1}


# --- detect_thrash ----------------------------------------------------------

def test_thrash_true_on_dwindling_tail(scoring):
    assert scoring.detect_thrash([3, 1, 0, 0], min_new=1, window=2) is True


def test_thrash_false_when_new_findings_present(scoring):
    assert scoring.detect_thrash([3, 1, 1], min_new=1, window=2) is False
    assert scoring.detect_thrash([1, 2, 3], min_new=1, window=2) is False


def test_thrash_requires_window_passes(scoring):
    assert scoring.detect_thrash([0, 0], min_new=1, window=3) is False
    assert scoring.detect_thrash([], min_new=1, window=2) is False


def test_thrash_min_new_zero_is_never_thrash(scoring):
    # min_new=0 means "no expectation of new findings" -> never thrash
    assert scoring.detect_thrash([1, 1, 1], min_new=0, window=2) is False


def test_thrash_invalid_args_raise(scoring):
    with pytest.raises(ValueError):
        scoring.detect_thrash([0, 0], min_new=-1)
    with pytest.raises(ValueError):
        scoring.detect_thrash([0, 0], window=0)


# --- classify_from_text -----------------------------------------------------

def test_classify_from_text(scoring):
    assert scoring.classify_from_text("this is a blocker") == "blocker"
    assert scoring.classify_from_text("major issue here") == "major"
    assert scoring.classify_from_text("just a nit") == "nit"
    assert scoring.classify_from_text("missing severity text") == "minor"
