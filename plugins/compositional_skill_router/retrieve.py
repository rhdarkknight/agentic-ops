"""Per-sub-task skill retrieval using SRA's embedding cache.

Reuses ~/.hermes/sra_cache/skill_embeddings.npy and skill_embeddings_meta.json.
L2-normalized vectors → inner product == cosine similarity (per SkillWeaver).
Returns top-k skills per sub-task.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except ImportError:
    HAVE_ST = False

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.path.expanduser("~/.hermes/sra_cache"))
EMB_FILE = CACHE_DIR / "skill_embeddings.npy"
META_FILE = CACHE_DIR / "skill_embeddings_meta.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RetrievedSkill:
    name: str
    description: str
    category: str
    score: float
    rank: int


_model_singleton: SentenceTransformer | None = None
_cache: tuple[np.ndarray, list[dict]] | None = None


def _get_model() -> SentenceTransformer:
    global _model_singleton
    if _model_singleton is None:
        if not HAVE_ST:
            raise RuntimeError("sentence-transformers not installed")
        _model_singleton = SentenceTransformer(MODEL_NAME)
    return _model_singleton


def _load_cache() -> tuple[np.ndarray, list[dict]]:
    """Load embeddings + metadata. Cached per-process."""
    global _cache
    if _cache is not None:
        return _cache
    if not EMB_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            f"SRA cache not found at {CACHE_DIR}. "
            "Run `hermes sra refresh` or trigger sra_skill_router to build embeddings."
        )
    embs = np.load(EMB_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)
    skills = meta["skills"]
    assert embs.shape[0] == len(skills), (
        f"Embedding count {embs.shape[0]} != skill count {len(skills)}"
    )
    _cache = (embs, skills)
    return _cache


def encode_query(text: str) -> np.ndarray:
    """Encode a query string → 384-dim L2-normalized vector."""
    model = _get_model()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def retrieve(subtask_text: str, top_k: int = 3) -> list[RetrievedSkill]:
    """Retrieve top-k skills for a single sub-task."""
    embs, skills = _load_cache()
    q = encode_query(subtask_text)
    scores = embs @ q  # inner product == cosine for L2-normalized vectors
    top_idx = np.argsort(-scores)[:top_k]
    out = []
    for rank, i in enumerate(top_idx):
        s = skills[int(i)]
        out.append(
            RetrievedSkill(
                name=s["name"],
                description=s.get("description", ""),
                category=s.get("category", ""),
                score=float(scores[int(i)]),
                rank=rank,
            )
        )
    return out


def retrieve_batch(subtask_texts: list[str], top_k: int = 3) -> list[list[RetrievedSkill]]:
    """Retrieve top-k skills for each sub-task. Encodes queries in one batch."""
    if not subtask_texts:
        return []
    embs, skills = _load_cache()
    model = _get_model()
    qs = model.encode(subtask_texts, convert_to_numpy=True, normalize_embeddings=True)
    qs = qs.astype(np.float32)
    out = []
    for q in qs:
        scores = embs @ q
        top_idx = np.argsort(-scores)[:top_k]
        out.append(
            [
                RetrievedSkill(
                    name=skills[int(i)]["name"],
                    description=skills[int(i)].get("description", ""),
                    category=skills[int(i)].get("category", ""),
                    score=float(scores[int(i)]),
                    rank=rank,
                )
                for rank, i in enumerate(top_idx)
            ]
        )
    return out


def cache_stats() -> dict:
    """Stats about the SRA embedding cache this plugin reuses."""
    if not EMB_FILE.exists():
        return {"available": False, "path": str(EMB_FILE)}
    embs, skills = _load_cache()
    return {
        "available": True,
        "n_skills": len(skills),
        "emb_dim": int(embs.shape[1]),
        "model": MODEL_NAME,
        "cache_path": str(EMB_FILE),
    }