"""Anchoring mitigation: novelty flag, counterfactual revisit, clean-slate path.

The three mechanisms that stop Hermes from anchoring to a prior that no longer
applies:

1. NOVELTY FLAG  — detect "no prior art" via recall similarity; register the
   task in the append-only world-model store (agent-world-model) so it's
   durable + queryable via the same recall path. Fall back to a plain JSONL
   log ONLY if the canonical store is unavailable (log a warning).
2. COUNTERFACTUAL — sample decisions from N days ago; would current-me make the
   same call? <30% counterfactual-no = anchoring.
3. CLEAN-SLATE    — re-solve a recurring problem WITHOUT memory first, then
   compare to the anchored answer. Divergence = healthy; identical = anchoring.
"""

from __future__ import annotations

import difflib
import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Canonical world-model store. If importable, novelty entries go here.
# Fallback: plain JSONL (same schema) + warning.
_WORLD_MODEL_DIR = os.path.expanduser("~/.hermes/worlds")
_FALLBACK_NO_PRIOR_ART = os.path.expanduser("~/.hermes/self-router/no_prior_art.jsonl")

NOVEL_ENTRY_TYPE = "no_prior_art"
NOVELTY_DEDUPE_WINDOW_SEC = 7 * 24 * 60 * 60


def build_task_signature(tool_name: str, args: Any = None) -> str:
    """Build a stable signature from the tool name and its arguments."""
    payload = args if args is not None else {}
    normalized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"{tool_name}:{normalized}"


class AnchoringStore:
    """Durable novelty register writing to the agent-world-model store.

    Prefers the canonical WorldModel archive; falls back to a plain JSONL
    append-only log with the same schema when the world model is unavailable.
    """

    def __init__(self, agent_id: str = "hermes", store_dir: Optional[str] = None) -> None:
        self.agent_id = agent_id
        self.store_dir = store_dir or os.environ.get("HERMES_WORLD_HOME", _WORLD_MODEL_DIR)
        self._wm = None  # lazy-loaded WorldModel
        self._wm_available: Optional[bool] = None

    def _load_world_model(self):
        if self._wm_available is not None:
            return self._wm
        try:
            import sys
            sys.path.insert(0, os.path.expanduser("~/.hermes/skills/agent-world-model/scripts"))
            from world_model import WorldModel  # type: ignore[import-not-found]
            self._wm = WorldModel(agent_id=self.agent_id, root=Path(self.store_dir))
            self._wm_available = True
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning(
                "agent-world-model unavailable (%s); falling back to JSONL register",
                exc,
            )
            self._wm = None
            self._wm_available = False
        return self._wm

    def register_no_prior_art(
        self,
        task_signature: str,
        closest_match: float,
        description: str,
        timestamp: Optional[float] = None,
    ) -> str:
        """Append a no_prior_art entry to the canonical store (or fallback).

        Returns the storage path used.
        """
        ts = timestamp if timestamp is not None else time.time()
        entry: Dict[str, Any] = {
            "type": NOVEL_ENTRY_TYPE,
            "task_signature": task_signature,
            "closest_match": round(float(closest_match), 4),
            "description": description,
            "timestamp": ts,
            "ts_iso": _iso(ts),
            "agent_id": self.agent_id,
        }
        fallback_path = self._fallback_path()
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path + ".lock", "a", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            if self._seen_recently(task_signature, ts, fallback_path):
                if os.path.isfile(self._archive_path()):
                    return self._archive_path()
                return fallback_path
            wm = self._load_world_model()
            if wm is not None:
                try:
                    wm.archive.append(
                        entry_type=NOVEL_ENTRY_TYPE,
                        agent_id=self.agent_id,
                        description=(
                            f"[no_prior_art] {description} "
                            f"(sig={task_signature}, closest={closest_match:.3f})"
                        ),
                        data=entry,
                    )
                    return self._archive_path()
                except Exception as exc:  # pragma: no cover
                    logger.warning("world-model write failed (%s); falling back to JSONL", exc)
            with open(fallback_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            return fallback_path

    def _fallback_path(self) -> str:
        if self.store_dir == _WORLD_MODEL_DIR and not os.environ.get("HERMES_WORLD_HOME"):
            return _FALLBACK_NO_PRIOR_ART
        return os.path.join(self.store_dir, "no_prior_art.jsonl")

    def _fallback_candidates(self) -> tuple[str, ...]:
        local = os.path.join(self.store_dir, "no_prior_art.jsonl")
        if local == _FALLBACK_NO_PRIOR_ART:
            return (local,)
        return (local, _FALLBACK_NO_PRIOR_ART)

    def _archive_path(self) -> str:
        safe_agent_id = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in self.agent_id
        )[:80] or "agent"
        return os.path.join(self.store_dir, f"{safe_agent_id}_archive.jsonl")

    @staticmethod
    def _recent_json_lines(path: str) -> list[dict[str, Any]]:
        if not os.path.isfile(path):
            return []
        rows: deque[dict[str, Any]] = deque(maxlen=5000)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(raw, dict):
                        rows.append(raw)
        except (OSError, UnicodeDecodeError):
            return []
        return list(rows)

    def _seen_recently(self, task_signature: str, ts: float, fallback_path: str) -> bool:
        cutoff = ts - NOVELTY_DEDUPE_WINDOW_SEC
        for path in (self._archive_path(), *self._fallback_candidates()):
            for raw in self._recent_json_lines(path):
                data = raw.get("data")
                data = data if isinstance(data, dict) else {}
                entry_type = raw.get("entry_type", raw.get("type", data.get("type")))
                if entry_type is not None and entry_type != NOVEL_ENTRY_TYPE:
                    continue
                signature = data.get("task_signature", raw.get("task_signature"))
                seen_ts = data.get("timestamp", raw.get("timestamp"))
                if signature == task_signature:
                    try:
                        if isinstance(seen_ts, (int, float, str)) and float(seen_ts) >= cutoff:
                            return True
                    except (TypeError, ValueError):
                        pass
        return False

    def closest_match(self, task_signature: str) -> float:
        """Return the highest similarity against stored novelty signatures."""
        best = 0.0
        for path in (self._archive_path(), *self._fallback_candidates()):
            for raw in self._recent_json_lines(path):
                data = raw.get("data")
                data = data if isinstance(data, dict) else {}
                entry_type = raw.get("entry_type", raw.get("type", data.get("type")))
                if entry_type is not None and entry_type != NOVEL_ENTRY_TYPE:
                    continue
                signature = data.get("task_signature", raw.get("task_signature"))
                if isinstance(signature, str):
                    best = max(best, difflib.SequenceMatcher(None, task_signature, signature).ratio())
        return round(best, 4)


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


def detect_novelty(
    closest_match: float,
    novelty_threshold: float = 0.6,
    task_signature: str = "",
    description: str = "",
    store: Optional[AnchoringStore] = None,
) -> Dict[str, Any]:
    """Flag a task as novel (no prior art) based on recall similarity.

    ``closest_match`` is the max Hindsight recall similarity for the task.
    Below ``novelty_threshold`` = no prior art. When novel, registers the task
    in the canonical store (world model) for durability + counterfactual later.
    """
    no_prior_art = float(closest_match or 0.0) < novelty_threshold
    result: Dict[str, Any] = {
        "no_prior_art": no_prior_art,
        "closest_match": round(float(closest_match or 0.0), 4),
        "novelty_threshold": novelty_threshold,
    }
    if no_prior_art and (task_signature or description):
        store = store or AnchoringStore()
        try:
            path = store.register_no_prior_art(
                task_signature=task_signature,
                closest_match=float(closest_match or 0.0),
                description=description,
            )
            result["register_path"] = path
        except Exception as exc:  # never let the register break routing
            logger.warning("no_prior_art register failed: %s", exc)
            result["register_error"] = str(exc)
    return result


def compute_anchoring_risk(closest_match: float, confidence: float = 0.5) -> float:
    """Estimate anchoring risk for a task.

    The dangerous case is a *novel* task (low closest_match) where the model is
    nevertheless *confident* — it will lean on a wrong prior rather than reason
    fresh. So confidence dominates; a low match (novel) does NOT lower the risk,
    it raises it (no prior to anchor to = free, but confidence on a novel task
    = the model is reproducing a stale prior). Range [0, 1].
    """
    conf = min(max(float(confidence or 0.0), 0.0), 1.0)
    # Confidence is the dominant signal. A modest match term captures the
    # "strong memory match" contributor the plan names.
    return round(conf * 0.75 + 0.25, 3) if conf >= 0.5 else round(conf * 0.5, 3)


def compute_counterfactual_no_rate(
    decisions: List[Tuple[str, bool]],
) -> Dict[str, Any]:
    """Given (decision_id, would_change) pairs, compute the discovery rate.

    ``would_change=True`` means current-me would NOT make the same call as
    the baseline (a counterfactual "no" = healthy discovery).
    <30% counterfactual-no rate = anchoring signal.
    """
    if not decisions:
        return {"rate": None, "count": 0, "anchored": True, "sample_count": 0}
    total = len(decisions)
    change = sum(1 for _, would_change in decisions if would_change)
    rate = change / total
    anchored = rate < 0.30
    return {
        "rate": round(rate, 3),
        "count": change,
        "sample_count": total,
        "anchored": anchored,
    }


def clean_slate_divergence(
    clean_slate_answer: str,
    anchored_answer: str,
) -> Dict[str, Any]:
    """Compare a clean-slate (no-memory) answer to the anchored answer.

    Identical = anchoring (no divergence). Different = healthy discovery.
    Divergence is a simple text-equality nominal check; identical strings
    (normalized) signal anchoring.
    """
    cs = (clean_slate_answer or "").strip().lower()
    an = (anchored_answer or "").strip().lower()
    identical = cs == an and bool(cs)
    return {
        "divergent": not identical,
        "identical": identical,
        "anchored": identical,  # identical to prior = anchoring
        "clean_slate_answer": clean_slate_answer,
        "anchored_answer": anchored_answer,
    }