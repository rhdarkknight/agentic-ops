"""Hindsight memory governance — FORGET primitive + steward approval queue.

Two primitives fill the gaps left by the standard Hindsight client:

1.  ``forget(memory_id, kind, reason)`` — physically removes a directive or
    mental_model from the bank. Idempotent (already-gone is success). Logs
    to an append-only audit file.

2.  ``propose(content, tags, ...)`` — queues a fact for steward review. The
    fact lands in the SQLite-backed pending_facts table, NOT in Hindsight.
    A human (or a bypass policy) calls ``approve(pending_id)`` to push
    the fact into Hindsight as a directive, or ``reject(pending_id)`` to
    drop it. Every op is audited.

Bypass tags (``HINDSIGHT_GOVERNANCE_BYPASS_TAGS``) and a queue kill
switch (``HINDSIGHT_GOVERNANCE_QUEUE=0``) let the user opt out of the
queue without changing code. ``HINDSIGHT_GOVERNANCE_FORGET=0`` disables
the FORGET primitive.

Reverse path for safety:
  * HINDSIGHT_GOVERNANCE_FORGET=0  → forget() is no-op, returns error
  * HINDSIGHT_GOVERNANCE_QUEUE=0   → propose() auto-approves
  * HINDSIGHT_GOVERNANCE_PENDING_DB=...  → override DB path (tests)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Local imports
from hermes_constants import get_hermes_home

_PLUGIN_DIR = Path(__file__).resolve().parent
_SHARED_DIR = get_hermes_home() / "plugins" / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

try:
    from hindsight_bank import resolve_active_bank_id as _resolve_bank_id
except Exception:  # pragma: no cover
    def _resolve_bank_id() -> str:
        return "hermes"

import pending_store as ps

logger = logging.getLogger(__name__)

AUDIT_PATH = Path(os.path.expanduser("~/.hermes/state/hindsight_governance_audit.jsonl"))
AUDIT_FAILURES_PATH = Path(
    os.path.expanduser("~/.hermes/state/hindsight_governance_audit_failures")
)

# Counter for audit-log write failures. Read by governance_status().
# A non-zero value means some ops were silent — the audit log is
# best-effort by design (we never want the audit failure to break the real
# op), but the user must be able to detect silent loss. The counter is
# persisted to AUDIT_FAILURES_PATH so it survives process restarts.
def _load_audit_write_failures() -> int:
    try:
        return int(AUDIT_FAILURES_PATH.read_text(encoding="utf-8").strip() or "0")
    except (FileNotFoundError, ValueError, OSError):
        return 0


_audit_write_failures: int = _load_audit_write_failures()

# Bypass-tag CAS: in-memory lock keyed on a content-fingerprint, value is
# a dict with `expires` (unix time) and optionally `winner_hid` (the
# hindsight_id of the winning create_directive call). The bypass path
# (_direct_approve) goes straight to Hindsight without a SQLite row, so
# there is no DB row to CAS against. We use a short-TTL in-process lock
# to coalesce concurrent bypass retains of the same content. Window: 5s
# — long enough to defeat the thundering-herd from a busy pre-compaction
# hook, short enough that intentional re-runs of the same fact still
# work. Coalesced callers see `winner_hid` once the winner writes it.
import threading as _threading
_BYPASS_LOCK_TTL_SEC = 5.0
_bypass_lock: dict[str, dict] = {}
_bypass_lock_mu = _threading.Lock()


def _bypass_cas(fingerprint: str) -> bool:
    """Attempt to claim a bypass-tag retain slot. Returns True on success
    (caller may proceed to create the directive), False on contention
    (a concurrent caller already holds the slot for the same content).
    """
    now = time.time()
    with _bypass_lock_mu:
        # Prune expired entries lazily
        expired = [k for k, slot in _bypass_lock.items() if slot["expires"] < now]
        for k in expired:
            _bypass_lock.pop(k, None)
        if fingerprint in _bypass_lock:
            return False
        _bypass_lock[fingerprint] = {"expires": now + _BYPASS_LOCK_TTL_SEC, "winner_hid": ""}
        return True


def _bypass_set_winner(fingerprint: str, hindsight_id: str) -> None:
    """Record the winner's hindsight_id so coalesced callers can see it."""
    with _bypass_lock_mu:
        slot = _bypass_lock.get(fingerprint)
        if slot is not None:
            slot["winner_hid"] = hindsight_id


def _bypass_winner_hid(fingerprint: str) -> str:
    """Return the winner's hindsight_id if available, else empty string."""
    with _bypass_lock_mu:
        slot = _bypass_lock.get(fingerprint)
        if slot is None:
            return ""
        return slot.get("winner_hid", "")


def _bypass_release(fingerprint: str) -> None:
    """Release a bypass slot — called after directive creation (success
    OR failure) so a quick retry isn't artificially blocked."""
    with _bypass_lock_mu:
        _bypass_lock.pop(fingerprint, None)


def _bypass_lock_held(fingerprint: str) -> bool:
    """True if a slot for this fingerprint is currently held (not expired)."""
    now = time.time()
    with _bypass_lock_mu:
        slot = _bypass_lock.get(fingerprint)
        if slot is None:
            return False
        return slot["expires"] >= now

# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

def _truthy(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default)
    return val not in ("0", "false", "False", "")


def _skip_sources() -> set[str]:
    """Sources to silently drop from the review queue and the digest.

    Default: the test-fixture source tags used by the governance test suite
    (v3_test, real_sdk_test). Comma-separated override via
    HINDSIGHT_GOVERNANCE_SKIP_SOURCES.
    """
    raw = os.environ.get(
        "HINDSIGHT_GOVERNANCE_SKIP_SOURCES", "v3_test,real_sdk_test"
    )
    return {s.strip() for s in raw.split(",") if s.strip()}


def _bypass_tags() -> list[str]:
    # Default: only `_health` (operational/observability facts that should
    # never block on a steward). `_provenance` and `_trace` were in the
    # original defaults but are footguns — a buggy provenance-emitter
    # would silently pollute Hindsight. Require explicit opt-in via the
    # HINDSIGHT_GOVERNANCE_BYPASS_TAGS env var if you want them.
    raw = os.environ.get("HINDSIGHT_GOVERNANCE_BYPASS_TAGS", "_health")
    return [t.strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Lazy Hindsight client (shared with hindsight-memory)
# ---------------------------------------------------------------------------

class _ClientHolder:
    """Lazy singleton matching the pattern in plugins/hindsight_memory/__init__.py"""

    def __init__(self) -> None:
        self._client = None
        self._bank_id: Optional[str] = None
        self._initialized = False
        self._error: Optional[str] = None

    def _ensure(self) -> Optional[object]:
        if self._initialized:
            return self._client
        self._initialized = True
        try:
            from hindsight import HindsightEmbedded

            cfg_path = get_hermes_home() / "hindsight" / "config.json"
            if not cfg_path.exists():
                self._error = "hindsight config.json not found"
                return None

            cfg = json.loads(cfg_path.read_text())
            api_key = (
                os.environ.get("HINDSIGHT_API_LLM_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or ""
            )
            if not api_key:
                self._error = "LLM API key not set"
                return None

            embed = HindsightEmbedded(
                profile=cfg.get("profile", "hermes"),
                llm_provider=cfg.get("llm_provider", "openrouter"),
                llm_api_key=api_key,
                llm_model=cfg.get("llm_model", "anthropic/claude-opus-4.6"),
                log_level="warning",
                idle_timeout=0,
            )
            self._client = embed.client
            self._bank_id = _resolve_bank_id()
            return self._client
        except Exception as e:
            self._error = str(e)
            logger.debug("hindsight-governance: client init failed: %s", e)
            return None

    def client(self) -> Optional[object]:
        return self._ensure()

    def bank_id(self) -> str:
        if not self._initialized:
            self._ensure()
        return self._bank_id or _resolve_bank_id() or "hermes"

    def error(self) -> Optional[str]:
        return self._error


_holder = _ClientHolder()


def _get_client():
    return _holder.client(), _holder.bank_id(), _holder.error()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _audit(op: str, **fields: Any) -> dict[str, Any]:
    global _audit_write_failures
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op": op,
        "by": os.environ.get("USER", "unknown"),
        **fields,
    }
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        _audit_write_failures += 1
        # Best-effort persist so the count survives process restarts.
        # We never raise from here — audit failure must not break the
        # real op.
        try:
            AUDIT_FAILURES_PATH.write_text(
                str(_audit_write_failures), encoding="utf-8"
            )
        except Exception:
            pass
        logger.debug("audit write failed: %s", e)
    return record


def read_audit(limit: int = 100, op: Optional[str] = None) -> list[dict[str, Any]]:
    if not AUDIT_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if op and r.get("op") != op:
                    continue
                out.append(r)
    except Exception:
        return out
    out.reverse()
    return out[:limit]


# ---------------------------------------------------------------------------
# FORGET primitive
# ---------------------------------------------------------------------------

VALID_KINDS = {"directive", "mental_model"}


def forget(memory_id: str, kind: str, reason: str = "") -> dict[str, Any]:
    """Physically remove a directive or mental_model from the active bank.

    Idempotent: deleting a non-existent id is treated as success (already
    gone). The op is logged to the audit file. Kill switch:
    HINDSIGHT_GOVERNANCE_FORGET=0 returns error without side-effects.
    """
    if not _truthy("HINDSIGHT_GOVERNANCE_FORGET"):
        return {"success": False, "error": "FORGET disabled by HINDSIGHT_GOVERNANCE_FORGET=0"}

    if kind not in VALID_KINDS:
        return {"success": False, "error": f"kind must be one of {sorted(VALID_KINDS)}"}
    if not memory_id or not isinstance(memory_id, str):
        return {"success": False, "error": "memory_id required"}

    client, bank_id, err = _get_client()
    if client is None:
        return {"success": False, "error": f"Hindsight client unavailable: {err}"}

    try:
        if kind == "directive":
            client.delete_directive(bank_id=bank_id, directive_id=memory_id)
        else:
            client.delete_mental_model(bank_id=bank_id, mental_model_id=memory_id)
    except Exception as e:
        msg = str(e).lower()
        # 404 / not found → already gone, still success
        if "not found" in msg or "404" in msg or "does not exist" in msg:
            audit = _audit(
                "forget", memory_id=memory_id, kind=kind, reason=reason,
                bank_id=bank_id, already_gone=True,
            )
            return {"success": True, "already_gone": True, "audit": audit}
        return {"success": False, "error": str(e)}

    audit = _audit(
        "forget", memory_id=memory_id, kind=kind, reason=reason,
        bank_id=bank_id, already_gone=False,
    )
    return {"success": True, "audit": audit}


# ---------------------------------------------------------------------------
# Memory-unit invalidation (PATCH state=invalidated) — Gap 5
# ---------------------------------------------------------------------------
# The daemon 0.9.1 has NO physical DELETE for memory_units. The only removal
# primitive is PATCH /v1/default/banks/{bank_id}/memories/{memory_id} with
# state=invalidated (soft-retire, reversible, moved to archive). The SDK does
# NOT wrap this endpoint, so we call it via the low-level escape hatch:
#   client.memory().api_client.call_api("PATCH", url, body=...)
# call_api is confirmed present on hindsight_client_api/api_client.py:253.

def _memory_unit_url(bank_id: str, memory_id: str) -> str:
    return f"/v1/default/banks/{bank_id}/memories/{memory_id}"


def _patch_memory_unit(memory_id: str, state: str, reason: str = "") -> dict[str, Any]:
    """PATCH a memory unit's curation state. Returns (success, error)."""
    if not _truthy("HINDSIGHT_GOVERNANCE_INVALIDATE_UNIT"):
        return {"success": False, "error": "disabled by HINDSIGHT_GOVERNANCE_INVALIDATE_UNIT=0"}
    if not memory_id or not isinstance(memory_id, str):
        return {"success": False, "error": "memory_id required"}
    client, bank_id, err = _get_client()
    if client is None:
        return {"success": False, "error": f"Hindsight client unavailable: {err}"}
    try:
        memory_api = client.memory  # property -> MemoryApi
        if callable(memory_api):
            memory_api = memory_api()
        api_client = memory_api.api_client
        # call_api needs the FULL url (host + path), not a relative path.
        host = getattr(api_client.configuration, "host", "") or ""
        path = _memory_unit_url(bank_id, memory_id)
        url = host.rstrip("/") + path
        body = {"state": state}
        if reason:
            body["reason"] = reason
        # call_api is async; run it through the provider's sync bridge.
        resp = _run_async(api_client.call_api("PATCH", url, body=body))
        return {"success": True, "state": state, "memory_id": memory_id, "bank_id": bank_id}
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "404" in msg or "does not exist" in msg:
            return {"success": True, "already_gone": True, "state": state,
                    "memory_id": memory_id, "bank_id": bank_id}
        return {"success": False, "error": str(e)}


def _run_async(coro):
    """Run an async coroutine to completion (sync context).

    Uses a persistent module-level event loop so the aiohttp client session
    survives across calls (asyncio.run() closes the loop each time, breaking
    the session on the 2nd call).
    """
    import asyncio
    global _GOV_LOOP
    try:
        asyncio.get_running_loop()
        # Already in a loop — run in a fresh thread with its own loop.
        import threading
        result = {}
        def _runner():
            result["value"] = asyncio.run(coro)
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        return result.get("value")
    except RuntimeError:
        pass
    if _GOV_LOOP is None or _GOV_LOOP.is_closed():
        _GOV_LOOP = asyncio.new_event_loop()
    return _GOV_LOOP.run_until_complete(coro)


_GOV_LOOP = None


def invalidate_unit(memory_id: str, reason: str = "") -> dict[str, Any]:
    """Soft-retire a memory unit (PATCH state=invalidated). Reversible."""
    res = _patch_memory_unit(memory_id, "invalidated", reason)
    if res.get("success"):
        _audit("invalidate_unit", memory_id=memory_id, reason=reason,
               bank_id=res.get("bank_id", ""), already_gone=res.get("already_gone", False))
    return res


def restore_unit(memory_id: str, reason: str = "") -> dict[str, Any]:
    """Revert a memory unit to valid (PATCH state=valid)."""
    res = _patch_memory_unit(memory_id, "valid", reason)
    if res.get("success"):
        _audit("restore_unit", memory_id=memory_id, reason=reason,
               bank_id=res.get("bank_id", ""), already_gone=res.get("already_gone", False))
    return res


def invalidate_units_batch(memory_ids: list[str], reason: str = "") -> dict[str, Any]:
    """Bulk invalidate multiple memory units. Per-id results."""
    if not memory_ids:
        return {"success": False, "error": "memory_ids required"}
    results = []
    ok = 0
    for mid in memory_ids:
        r = invalidate_unit(mid, reason)
        results.append({"memory_id": mid, "success": r.get("success", False),
                        "error": r.get("error"), "already_gone": r.get("already_gone", False)})
        if r.get("success"):
            ok += 1
    return {"success": True, "invalidated": ok, "total": len(memory_ids), "results": results}


# ---------------------------------------------------------------------------
# Search-to-forget
# ---------------------------------------------------------------------------

def _substring_score(needle: str, haystack: str) -> float:
    if not needle:
        return 0.0
    n = needle.lower()
    h = haystack.lower()
    if n in h:
        return 1.0 - (h.find(n) / max(len(h), 1)) * 0.5
    # token-level match
    ntok = set(re.findall(r"\w+", n))
    htok = set(re.findall(r"\w+", h))
    if not ntok or not htok:
        return 0.0
    overlap = len(ntok & htok) / len(ntok)
    return overlap * 0.5


def search_to_forget(query: str, top_k: int = 5, kind: str | None = None) -> dict[str, Any]:
    """Return candidate directives + mental_models matching the query.

    Search is deterministic substring + token overlap (no LLM call). The
    human reads the candidates and decides which to forget.
    """
    client, bank_id, err = _get_client()
    if client is None:
        return {"success": False, "error": f"Hindsight client unavailable: {err}", "candidates": []}

    candidates: list[dict[str, Any]] = []
    kinds = [kind] if kind in VALID_KINDS else list(VALID_KINDS)

    try:
        if "directive" in kinds:
            try:
                directives = client.list_directives(bank_id=bank_id)
                for d in directives or []:
                    name = getattr(d, "name", "") or ""
                    content = getattr(d, "content", "") or ""
                    score = max(_substring_score(query, name), _substring_score(query, content))
                    if score > 0.0:
                        candidates.append({
                            "id": getattr(d, "id", ""),
                            "kind": "directive",
                            "name": name,
                            "content": content,
                            "score": round(score, 3),
                            "tags": list(getattr(d, "tags", []) or []),
                        })
            except Exception as e:
                logger.debug("list_directives failed: %s", e)

        if "mental_model" in kinds:
            try:
                mms = client.list_mental_models(bank_id=bank_id)
                for m in mms or []:
                    name = getattr(m, "name", "") or ""
                    content = getattr(m, "source_query", "") or ""
                    score = max(_substring_score(query, name), _substring_score(query, content))
                    if score > 0.0:
                        candidates.append({
                            "id": getattr(m, "id", ""),
                            "kind": "mental_model",
                            "name": name,
                            "content": content,
                            "score": round(score, 3),
                            "tags": list(getattr(m, "tags", []) or []),
                        })
            except Exception as e:
                logger.debug("list_mental_models failed: %s", e)
    except Exception as e:
        return {"success": False, "error": str(e), "candidates": []}

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {"success": True, "candidates": candidates[:max(1, int(top_k))]}


def list_recent(limit: int = 20, kind: str | None = None) -> dict[str, Any]:
    """Browse the most recent directives + mental_models for manual selection."""
    client, bank_id, err = _get_client()
    if client is None:
        return {"success": False, "error": f"Hindsight client unavailable: {err}", "items": []}

    items: list[dict[str, Any]] = []
    kinds = [kind] if kind in VALID_KINDS else list(VALID_KINDS)
    try:
        if "directive" in kinds:
            for d in client.list_directives(bank_id=bank_id) or []:
                items.append({
                    "id": getattr(d, "id", ""),
                    "kind": "directive",
                    "name": getattr(d, "name", "") or "",
                    "content": getattr(d, "content", "") or "",
                    "created_at": getattr(d, "created_at", "") or "",
                    "tags": list(getattr(d, "tags", []) or []),
                })
        if "mental_model" in kinds:
            for m in client.list_mental_models(bank_id=bank_id) or []:
                items.append({
                    "id": getattr(m, "id", ""),
                    "kind": "mental_model",
                    "name": getattr(m, "name", "") or "",
                    "content": getattr(m, "source_query", "") or "",
                    "created_at": getattr(m, "created_at", "") or "",
                    "tags": list(getattr(m, "tags", []) or []),
                })
    except Exception as e:
        return {"success": False, "error": str(e), "items": []}

    items.sort(key=lambda i: i.get("created_at", ""), reverse=True)
    return {"success": True, "items": items[:max(1, int(limit))]}


# ---------------------------------------------------------------------------
# Steward approval queue
# ---------------------------------------------------------------------------

def _content_hash(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def propose(
    content: str,
    tags: Optional[list[str]] = None,
    *,
    source: str = "manual",
    salience: str = "medium",
    entities: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    proposed_by: Optional[str] = None,
) -> dict[str, Any]:
    """Queue a fact for steward review.

    Bypass: if any tag in ``HINDSIGHT_GOVERNANCE_BYPASS_TAGS`` is in
    ``tags``, the fact is auto-approved (calls ``approve`` internally
    with the bypass reason). If the queue kill switch is off, also
    auto-approve (legacy behavior).
    """
    if not content or not isinstance(content, str):
        return {"success": False, "error": "content required"}
    tags = tags or []

    # Silently drop test-fixture sources so they never reach the digest,
    # never block a steward, and never cost a retain call.
    if source in _skip_sources():
        return {
            "success": True,
            "queued": False,
            "skipped": True,
            "reason": f"source={source!r} in HINDSIGHT_GOVERNANCE_SKIP_SOURCES",
        }

    bypassed = False
    if not _truthy("HINDSIGHT_GOVERNANCE_QUEUE"):
        bypassed = True
    elif any(t in _bypass_tags() for t in tags):
        bypassed = True

    if bypassed:
        # Direct path: call retain through the existing hindsight_retain
        return _direct_approve(
            content=content, tags=tags, source=source, salience=salience,
            session_id=session_id, proposed_by=proposed_by,
            note=f"bypass: tag={','.join(t for t in tags if t in _bypass_tags()) or 'queue_disabled'}",
        )

    fact = ps.insert_pending(
        content=content, source=source, tags=tags,
        entities=entities or [], salience=salience,
        session_id=session_id, proposed_by=proposed_by,
    )
    _audit(
        "propose", pending_id=fact.pending_id,
        content_hash=_content_hash(content), source=source,
        tags=tags, proposed_by=proposed_by,
    )
    return {
        "success": True,
        "queued": True,
        "pending_id": fact.pending_id,
        "expires_at": fact.expires_at,
    }


def _direct_approve(
    *, content: str, tags: list[str], source: str, salience: str,
    session_id: Optional[str], proposed_by: Optional[str], note: str,
) -> dict[str, Any]:
    """Bypass path — push directly into Hindsight as a directive, no queue.

    Concurrency: uses an in-memory CAS keyed on a content-fingerprint to
    coalesce concurrent bypass retains of the same content. The lock has
    a 5s TTL; a slow re-run of the same fact after 5s will pass through
    and create a new directive (acceptable: bypass is for ops/observability
    data where dupes are tolerable; the audit log + tags let the user
    detect them).
    """
    import hashlib
    fingerprint = hashlib.sha1(
        (content[:200] + "|" + ",".join(sorted(tags))).encode("utf-8")
    ).hexdigest()
    if not _bypass_cas(fingerprint):
        # Coalesced: return the winner's hindsight_id if it has been
        # recorded yet. If the winner hasn't finished creating the
        # directive, the lock is still held; we spin briefly to wait.
        winner_hid = ""
        for _ in range(20):  # up to ~1s
            winner_hid = _bypass_winner_hid(fingerprint)
            if winner_hid or not _bypass_lock_held(fingerprint):
                break
            time.sleep(0.05)
        return {
            "success": True, "queued": False, "bypassed": True,
            "coalesced": True,
            "hindsight_id": winner_hid,
            "note": "coalesced with concurrent bypass retain",
        }
    client, bank_id, err = _get_client()
    if client is None:
        _bypass_release(fingerprint)
        return {"success": False, "error": f"Hindsight client unavailable: {err}"}
    try:
        result = client.create_directive(
            bank_id=bank_id,
            name=(content[:60] + "…") if len(content) > 60 else content,
            content=content,
            tags=list(set(tags + ["governance_bypass"])),
        )
        hid = getattr(result, "id", None) or getattr(result, "directive_id", None) or ""
        _bypass_set_winner(fingerprint, hid)
        _audit(
            "approve", pending_id=None, hindsight_id=hid, source=source,
            by=proposed_by or os.environ.get("USER", "unknown"), note=note,
            bypassed=True,
        )
        return {"success": True, "queued": False, "bypassed": True, "hindsight_id": hid}
    except Exception as e:
        _bypass_release(fingerprint)
        return {"success": False, "error": str(e)}


def review_pending(
    limit: int = 50,
    source: Optional[str] = None,
    status: str = "pending",
) -> dict[str, Any]:
    """List facts in the queue. Default returns oldest pending first."""
    rows = ps.list_pending(status=status, source=source, limit=limit)
    return {
        "success": True,
        "items": [
            {
                "pending_id": r.pending_id,
                "proposed_at": r.proposed_at,
                "source": r.source,
                "content": r.content,
                "tags": r.tags,
                "salience": r.salience,
                "proposed_by": r.proposed_by,
                "session_id": r.session_id,
                "expires_at": r.expires_at,
            }
            for r in rows
        ],
    }


def approve(pending_id: str, note: str = "", reviewed_by: str = "human") -> dict[str, Any]:
    """Push a pending fact into Hindsight as a directive. Idempotent.

    TOCTOU-safe: uses SQLite compare-and-set to ensure only one concurrent
    caller can win the right to create a Hindsight directive. Loser gets
    an ``error: "approve race: ..."`` response and the orphan directive
    is best-effort cleaned up.
    """
    if not pending_id:
        return {"success": False, "error": "pending_id required"}
    row = ps.get_pending(pending_id)
    if row is None:
        return {"success": False, "error": f"pending_id {pending_id} not found"}
    if row.status == "approved" and row.hindsight_id:
        return {
            "success": True, "already_approved": True,
            "hindsight_id": row.hindsight_id,
            "pending_id": pending_id,
        }
    if row.status != "pending":
        return {
            "success": False, "error": f"cannot approve: status is {row.status}",
        }

    client, bank_id, err = _get_client()
    if client is None:
        return {"success": False, "error": f"Hindsight client unavailable: {err}"}

    # Step 1: claim the row via CAS — flip status to 'approving' so a
    # racing caller sees the transition and bails before creating.
    claim = ps.update_status(
        pending_id,
        status="approving",  # intermediate state; not in valid filter set
        require_prior_status="pending",
    )
    if claim is None:
        # Another caller won the claim (or row already changed)
        current = ps.get_pending(pending_id)
        if current and current.status == "approved" and current.hindsight_id:
            return {
                "success": True, "already_approved": True,
                "hindsight_id": current.hindsight_id, "pending_id": pending_id,
            }
        return {
            "success": False, "error": "approve race: row already transitioning",
            "pending_id": pending_id,
            "current_status": current.status if current else None,
        }

    # Step 2: create the directive. If this fails, roll the claim back.
    try:
        result = client.create_directive(
            bank_id=bank_id,
            name=(row.content[:60] + "…") if len(row.content) > 60 else row.content,
            content=row.content,
            tags=list(set((row.tags or []) + ["steward_approved"])),
        )
    except Exception as e:
        ps.update_status(
            pending_id, status="pending",
            require_prior_status="approving",
        )
        return {"success": False, "error": f"Hindsight create failed: {e}"}

    hid = getattr(result, "id", None) or getattr(result, "directive_id", None) or ""

    # Step 3: finalize — flip 'approving' to 'approved'. Another caller
    # could theoretically have already done this, in which case we
    # detected the race and have an orphan to clean up.
    final = ps.update_status(
        pending_id, status="approved",
        reviewed_by=reviewed_by, review_note=note, hindsight_id=hid,
        require_prior_status="approving",
    )
    if final is None:
        # Rollback the orphan directive
        try:
            client.delete_directive(bank_id=bank_id, directive_id=hid)
        except Exception as cleanup_err:
            logger.warning("approve CAS lost at finalize: orphaned %s not cleaned: %s", hid, cleanup_err)
        current = ps.get_pending(pending_id)
        return {
            "success": False,
            "error": "approve race: row transitioned before finalize; another caller won",
            "pending_id": pending_id,
            "winner_hindsight_id": current.hindsight_id if current else None,
        }

    _audit(
        "approve", pending_id=pending_id, hindsight_id=hid,
        by=reviewed_by, note=note, content_hash=_content_hash(row.content),
    )
    return {
        "success": True, "hindsight_id": hid, "pending_id": pending_id,
    }


def reject(pending_id: str, note: str = "", reviewed_by: str = "human") -> dict[str, Any]:
    """Mark a pending fact rejected. No SDK call. Idempotent on already-rejected."""
    if not pending_id:
        return {"success": False, "error": "pending_id required"}
    row = ps.get_pending(pending_id)
    if row is None:
        return {"success": False, "error": f"pending_id {pending_id} not found"}
    if row.status == "rejected":
        return {"success": True, "already_rejected": True, "pending_id": pending_id}
    if row.status == "approved":
        return {"success": False, "error": "cannot reject: already approved"}
    ps.update_status(
        pending_id, status="rejected",
        reviewed_by=reviewed_by, review_note=note,
        require_prior_status="pending",  # same CAS pattern
    )
    _audit(
        "reject", pending_id=pending_id,
        by=reviewed_by, note=note, content_hash=_content_hash(row.content),
    )
    return {"success": True, "pending_id": pending_id}


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------

def approve_batch(
    pending_ids: list[str],
    note: str = "",
    reviewed_by: str = "human",
) -> dict[str, Any]:
    """Bulk-approve multiple pending facts. Each id is processed by
    :func:`approve` independently — one failure does not block the rest.

    Returns a per-id result list plus aggregate counts.
    """
    if not pending_ids:
        return {"success": True, "results": [], "approved": 0, "skipped": 0, "failed": 0}
    results: list[dict[str, Any]] = []
    approved = skipped = failed = 0
    for pid in pending_ids:
        r = approve(pid, note=note, reviewed_by=reviewed_by)
        results.append({"pending_id": pid, **r})
        if r.get("success"):
            approved += 1
        elif r.get("already_approved"):
            skipped += 1
        else:
            failed += 1
    return {
        "success": failed == 0,
        "results": results,
        "approved": approved,
        "skipped": skipped,
        "failed": failed,
    }


def reject_batch(
    pending_ids: list[str],
    note: str = "",
    reviewed_by: str = "human",
) -> dict[str, Any]:
    """Bulk-reject multiple pending facts. Each id is processed by
    :func:`reject` independently.
    """
    if not pending_ids:
        return {"success": True, "results": [], "rejected": 0, "skipped": 0, "failed": 0}
    results: list[dict[str, Any]] = []
    rejected = skipped = failed = 0
    for pid in pending_ids:
        r = reject(pid, note=note, reviewed_by=reviewed_by)
        results.append({"pending_id": pid, **r})
        if r.get("success"):
            rejected += 1
        elif r.get("already_rejected"):
            skipped += 1
        else:
            failed += 1
    return {
        "success": failed == 0,
        "results": results,
        "rejected": rejected,
        "skipped": skipped,
        "failed": failed,
    }


def review_pending_compact(limit: int = 20, source: Optional[str] = None) -> dict[str, Any]:
    """Compact one-line-per-fact summary for Telegram / mobile review.

    Each line is short enough to fit a Telegram message: first 80 chars
    of content, salience badge, age, pending_id suffix.
    """
    rows = ps.list_pending(status="pending", source=source, limit=limit)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    lines: list[dict[str, Any]] = []
    for r in rows:
        try:
            age_s = (now - datetime.fromisoformat(r.proposed_at)).total_seconds()
        except Exception:
            age_s = 0
        snippet = r.content[:80] + ("…" if len(r.content) > 80 else "")
        sal = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(r.salience, "🟡")
        if age_s < 3600:
            age_str = f"{int(age_s/60)}m"
        elif age_s < 86400:
            age_str = f"{int(age_s/3600)}h"
        else:
            age_str = f"{int(age_s/86400)}d"
        lines.append({
            "pending_id": r.pending_id,
            "summary": f"{sal} [{age_str}] {snippet}",
            "source": r.source,
            "salience": r.salience,
        })
    return {
        "success": True,
        "count": len(lines),
        "lines": lines,
        "hint": "hindsight_approve_batch(pending_ids=[...]) or hindsight_reject_batch(pending_ids=[...])",
    }


def governance_status() -> dict[str, Any]:
    """Snapshot of governance state — useful for cron health checks."""
    pending = len(ps.list_pending(status="pending"))
    approved = len(ps.list_pending(status="approved", limit=1000))
    rejected = len(ps.list_pending(status="rejected", limit=1000))
    expired = len(ps.list_pending(status="expired", limit=1000))
    return {
        "success": True,
        "bank_id": _holder.bank_id(),
        "client_error": _holder.error(),
        "kill_switches": {
            "FORGET": _truthy("HINDSIGHT_GOVERNANCE_FORGET", "1"),
            "QUEUE": _truthy("HINDSIGHT_GOVERNANCE_QUEUE", "1"),
        },
        "bypass_tags": _bypass_tags(),
        "audit_path": AUDIT_PATH,
        "audit_write_failures": _audit_write_failures,
        "counts": {
            "pending": ps.count_pending(),
            "approved": ps.count_by_status("approved"),
            "rejected": ps.count_by_status("rejected"),
            "expired": ps.count_by_status("expired"),
        },
        "action": (
            "queue is empty"
            if ps.count_pending() == 0
            else f"consider draining the queue ({ps.count_pending()} pending)"
        ),
    }
# Pre-compaction integration
# ---------------------------------------------------------------------------

def make_propose_retain_caller(
    *, source: str = "pre_compaction", session_id: Optional[str] = None,
) -> Callable[[dict], None]:
    """Return a ``retain_caller(payload) -> None`` suitable for passing into
    :func:`pre_compaction.extract_and_dedupe`.

    Routes each extracted fact through :func:`propose` so it lands in the
    pending queue instead of being written to Hindsight directly. The
    ``source`` and ``session_id`` are stamped on every proposed fact for
    audit traceability.

    Kill switch: ``HINDSIGHT_GOVERNANCE_QUEUE=0`` or bypass tags still
    short-circuit to direct retain inside ``propose`` — no change in
    behavior for ops/observability facts.

    Usage from the in-repo HindsightMemoryProvider.on_pre_compress:

        from hindsight_governance import make_propose_retain_caller
        caller = make_propose_retain_caller(source="pre_compaction", session_id=sid)
        pre_compaction.extract_and_dedupe(messages, retain_caller=caller, ...)
    """
    def _caller(payload: dict) -> None:
        try:
            # pre_compaction payload shape: {entity, relation, value, salience, source, session_id, tags}
            # Manually-built payloads may use {content, ...} — handle both.
            if "content" in payload:
                content = payload["content"]
                entities = list(payload.get("entities") or [])
            else:
                entity = payload.get("entity") or ""
                relation = payload.get("relation") or ""
                value = payload.get("value") or ""
                content = f"{entity} {relation} {value}".strip()
                entities = [entity] if entity else []

            if not content:
                return

            tags = list(payload.get("tags") or [])
            if "pre_compaction" not in tags:
                tags.append("pre_compaction")

            propose(
                content=content,
                tags=tags,
                source=source,
                salience=payload.get("salience", "medium"),
                entities=entities,
                session_id=session_id or payload.get("session_id"),
            )
        except Exception as e:
            logger.debug("propose_retain_caller failed: %s", e)
    return _caller


__all__ = [
    "forget", "search_to_forget", "list_recent",
    "propose", "review_pending", "review_pending_compact", "approve", "reject",
    "approve_batch", "reject_batch",
    "read_audit", "governance_status",
    "make_propose_retain_caller",
    "VALID_KINDS",
]
