"""Hindsight memory governance plugin — FORGET + steward approval queue.

Tool surface exposed to the agent:

  hindsight_forget(memory_id, kind, reason) -> str (JSON)
  hindsight_search_to_forget(query, top_k, kind) -> str (JSON)
  hindsight_list_recent(limit, kind) -> str (JSON)
  hindsight_propose(content, tags, source, salience, entities, session_id) -> str (JSON)
  hindsight_review_pending(limit, source, status) -> str (JSON)
  hindsight_approve(pending_id, note, reviewed_by) -> str (JSON)
  hindsight_reject(pending_id, note, reviewed_by) -> str (JSON)
  hindsight_audit_log(limit, op) -> str (JSON)
  hindsight_governance_status() -> str (JSON)

All tools return JSON strings; the agent runtime unwraps them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# The plugin loader does NOT add the plugin dir to sys.path, so bare sibling
# imports (`import governance`, `import pending_store`) fail. Add the plugin
# dir to sys.path so all sibling modules resolve.
_GOV_DIR = Path(__file__).parent
if str(_GOV_DIR) not in sys.path:
    sys.path.insert(0, str(_GOV_DIR))

import governance as _gov  # noqa: E402


def _j(obj) -> str:
    return json.dumps(obj, default=str)


# --- tool handlers ---------------------------------------------------------

def hindsight_forget(memory_id: str, kind: str, reason: str = "", **_kw) -> str:
    """Physically remove a directive or mental_model from Hindsight. Idempotent."""
    return _j(_gov.forget(memory_id, kind, reason))


def hindsight_search_to_forget(query: str, top_k: int = 5, kind: str = "", **_kw) -> str:
    """Return directive + mental_model candidates matching a substring query."""
    return _j(_gov.search_to_forget(query, top_k=top_k, kind=kind or None))


def hindsight_list_recent(limit: int = 20, kind: str = "", **_kw) -> str:
    """List the most recent directives + mental_models for manual selection."""
    return _j(_gov.list_recent(limit=limit, kind=kind or None))


def hindsight_propose(
    content: str,
    tags: list[str] | None = None,
    source: str = "manual",
    salience: str = "medium",
    entities: list[str] | None = None,
    session_id: str = "",
    **_kw,
) -> str:
    """Queue a fact for steward review. Returns pending_id + expires_at."""
    return _j(_gov.propose(
        content=content, tags=tags, source=source, salience=salience,
        entities=entities, session_id=session_id or None,
    ))


def hindsight_review_pending(limit: int = 50, source: str = "", status: str = "pending", **_kw) -> str:
    """List facts awaiting steward review."""
    return _j(_gov.review_pending(limit=limit, source=source or None, status=status))


def hindsight_approve(pending_id: str, note: str = "", reviewed_by: str = "human", **_kw) -> str:
    """Push a pending fact into Hindsight as a directive. Idempotent."""
    return _j(_gov.approve(pending_id, note=note, reviewed_by=reviewed_by))


def hindsight_reject(pending_id: str, note: str = "", reviewed_by: str = "human", **_kw) -> str:
    """Mark a pending fact rejected. No SDK call."""
    return _j(_gov.reject(pending_id, note=note, reviewed_by=reviewed_by))


def hindsight_audit_log(limit: int = 100, op: str = "", **_kw) -> str:
    """Read the append-only audit log."""
    return _j({"items": _gov.read_audit(limit=limit, op=op or None)})


def hindsight_governance_status(**_kw) -> str:
    """Snapshot of governance state — counts, kill switches, bank."""
    return _j(_gov.governance_status())


def hindsight_approve_batch(pending_ids: list[str], note: str = "", reviewed_by: str = "human", **_kw) -> str:
    """Bulk-approve multiple pending facts. Per-id results + aggregate counts."""
    return _j(_gov.approve_batch(pending_ids, note=note, reviewed_by=reviewed_by))


def hindsight_reject_batch(pending_ids: list[str], note: str = "", reviewed_by: str = "human", **_kw) -> str:
    """Bulk-reject multiple pending facts. Per-id results + aggregate counts."""
    return _j(_gov.reject_batch(pending_ids, note=note, reviewed_by=reviewed_by))


def hindsight_review_pending_compact(limit: int = 20, source: str = "", **_kw) -> str:
    """Compact one-line-per-fact summary for Telegram / mobile review."""
    return _j(_gov.review_pending_compact(limit=limit, source=source or None))


def hindsight_invalidate_unit(memory_id: str, reason: str = "", **_kw) -> str:
    """Soft-retire a memory unit (PATCH state=invalidated). Reversible via restore."""
    return _j(_gov.invalidate_unit(memory_id, reason))


def hindsight_restore_unit(memory_id: str, reason: str = "", **_kw) -> str:
    """Revert a memory unit to valid (PATCH state=valid)."""
    return _j(_gov.restore_unit(memory_id, reason))


def hindsight_invalidate_units_batch(memory_ids: list[str], reason: str = "", **_kw) -> str:
    """Bulk invalidate multiple memory units. Per-id results."""
    return _j(_gov.invalidate_units_batch(memory_ids, reason))


# --- plugin metadata -------------------------------------------------------

__plugin__ = {
    "name": "hindsight-governance",
    "version": "1.0.0",
    "description": "Hindsight memory governance — FORGET primitive + steward approval queue for new facts.",
    "tools": [
        {
            "name": "hindsight_forget",
            "description": "Physically remove a directive or mental_model from the active Hindsight bank. Idempotent: deleting a non-existent id is treated as success. Audit-logged. Kill switch: HINDSIGHT_GOVERNANCE_FORGET=0.",
            "func": hindsight_forget,
            "parameters": {
                "memory_id": {"type": "string", "description": "The id of the directive or mental_model to remove."},
                "kind": {"type": "string", "enum": ["directive", "mental_model"], "description": "Type of memory to forget."},
                "reason": {"type": "string", "description": "Audit note explaining why this is being removed."},
            },
            "required": ["memory_id", "kind"],
        },
        {
            "name": "hindsight_search_to_forget",
            "description": "Find directive + mental_model candidates matching a substring query. Read-only; no side-effects. Use the returned ids with hindsight_forget to remove a fact.",
            "func": hindsight_search_to_forget,
            "parameters": {
                "query": {"type": "string", "description": "Substring or token search."},
                "top_k": {"type": "integer", "default": 5, "description": "Max candidates to return."},
                "kind": {"type": "string", "enum": ["directive", "mental_model", ""], "description": "Restrict to one type. Empty = both."},
            },
            "required": ["query"],
        },
        {
            "name": "hindsight_list_recent",
            "description": "List the most recent directives + mental_models for manual browsing. Use with hindsight_forget to clean up after a bad batch.",
            "func": hindsight_list_recent,
            "parameters": {
                "limit": {"type": "integer", "default": 20},
                "kind": {"type": "string", "enum": ["directive", "mental_model", ""], "description": "Restrict to one type."},
            },
            "required": [],
        },
        {
            "name": "hindsight_propose",
            "description": "Queue a fact for steward review. The fact lands in the pending_facts SQLite table, NOT in Hindsight, until a human calls hindsight_approve. Bypass tags (_health, _provenance, _trace) auto-approve. Kill switch: HINDSIGHT_GOVERNANCE_QUEUE=0.",
            "func": hindsight_propose,
            "parameters": {
                "content": {"type": "string", "description": "The fact text."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
                "source": {"type": "string", "default": "manual", "description": "Source label (pre_compaction, manual, subagent, cron)."},
                "salience": {"type": "string", "enum": ["high", "medium", "low"], "default": "medium"},
                "entities": {"type": "array", "items": {"type": "string"}, "description": "Optional extracted entities."},
                "session_id": {"type": "string", "description": "Session this came from."},
            },
            "required": ["content"],
        },
        {
            "name": "hindsight_review_pending",
            "description": "List facts awaiting steward review. Default returns oldest pending first.",
            "func": hindsight_review_pending,
            "parameters": {
                "limit": {"type": "integer", "default": 50},
                "source": {"type": "string", "description": "Filter by source (pre_compaction, manual, subagent, cron)."},
                "status": {"type": "string", "enum": ["pending", "approved", "rejected", "expired"], "default": "pending"},
            },
            "required": [],
        },
        {
            "name": "hindsight_approve",
            "description": "Push a pending fact into Hindsight as a directive. Idempotent on already-approved (returns existing hindsight_id).",
            "func": hindsight_approve,
            "parameters": {
                "pending_id": {"type": "string"},
                "note": {"type": "string", "description": "Audit note (e.g. 'verified against spec v2')."},
                "reviewed_by": {"type": "string", "default": "human"},
            },
            "required": ["pending_id"],
        },
        {
            "name": "hindsight_reject",
            "description": "Mark a pending fact rejected. No SDK call. Cannot reject an already-approved fact.",
            "func": hindsight_reject,
            "parameters": {
                "pending_id": {"type": "string"},
                "note": {"type": "string"},
                "reviewed_by": {"type": "string", "default": "human"},
            },
            "required": ["pending_id"],
        },
        {
            "name": "hindsight_audit_log",
            "description": "Read the append-only audit log of all forget/propose/approve/reject operations.",
            "func": hindsight_audit_log,
            "parameters": {
                "limit": {"type": "integer", "default": 100},
                "op": {"type": "string", "enum": ["forget", "propose", "approve", "reject", ""], "description": "Filter by op type."},
            },
            "required": [],
        },
        {
            "name": "hindsight_governance_status",
            "description": "Snapshot of governance state — pending/approved/rejected/expired counts, kill-switch state, active bank id, and an action hint.",
            "func": hindsight_governance_status,
            "parameters": {},
            "required": [],
        },
        {
            "name": "hindsight_approve_batch",
            "description": "Bulk-approve multiple pending facts in one call. Each id is processed independently — one failure does not block the rest. Returns per-id results plus approved/skipped/failed counts. Use hindsight_review_pending first to get the pending_ids.",
            "func": hindsight_approve_batch,
            "parameters": {
                "pending_ids": {"type": "array", "items": {"type": "string"}, "description": "List of pending_ids to approve."},
                "note": {"type": "string", "description": "Audit note applied to all approvals."},
                "reviewed_by": {"type": "string", "default": "human"},
            },
            "required": ["pending_ids"],
        },
        {
            "name": "hindsight_reject_batch",
            "description": "Bulk-reject multiple pending facts in one call. Each id is processed independently.",
            "func": hindsight_reject_batch,
            "parameters": {
                "pending_ids": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
                "reviewed_by": {"type": "string", "default": "human"},
            },
            "required": ["pending_ids"],
        },
        {
            "name": "hindsight_review_pending_compact",
            "description": "Compact one-line-per-fact summary for Telegram / mobile review. Each line: salience badge + age + first 80 chars of content + pending_id. Use the pending_ids with hindsight_approve_batch / hindsight_reject_batch.",
            "func": hindsight_review_pending_compact,
            "parameters": {
                "limit": {"type": "integer", "default": 20},
                "source": {"type": "string", "description": "Filter by source (pre_compaction, manual, subagent, cron)."},
            },
            "required": [],
        },
        {
            "name": "hindsight_invalidate_unit",
            "description": "Soft-retire a memory unit (PATCH state=invalidated). Excluded from recall/consolidation, moved to archive. Reversible via hindsight_restore_unit. Audit-logged. Kill switch: HINDSIGHT_GOVERNANCE_INVALIDATE_UNIT=0.",
            "func": hindsight_invalidate_unit,
            "parameters": {
                "memory_id": {"type": "string", "description": "The memory unit id to invalidate."},
                "reason": {"type": "string", "description": "Audit note explaining why."},
            },
            "required": ["memory_id"],
        },
        {
            "name": "hindsight_restore_unit",
            "description": "Revert a memory unit to valid (PATCH state=valid). Recovery path for an invalidated unit. Audit-logged.",
            "func": hindsight_restore_unit,
            "parameters": {
                "memory_id": {"type": "string", "description": "The memory unit id to restore."},
                "reason": {"type": "string", "description": "Audit note."},
            },
            "required": ["memory_id"],
        },
        {
            "name": "hindsight_invalidate_units_batch",
            "description": "Bulk invalidate multiple memory units. Per-id results; one failure does not block the rest.",
            "func": hindsight_invalidate_units_batch,
            "parameters": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "List of memory unit ids to invalidate."},
                "reason": {"type": "string", "description": "Audit note applied to all."},
            },
            "required": ["memory_ids"],
        },
    ],
}


def register(ctx):
    """Register hindsight governance tools with Hermes."""
    for tool_def in __plugin__["tools"]:
        ctx.register_tool(
            name=tool_def["name"],
            toolset="memory",
            schema=tool_def["parameters"],
            handler=tool_def["func"],
            description=tool_def["description"],
        )
