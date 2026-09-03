# Hindsight Governance

Forgetting facts and steward review of new knowledge writes.

## Why

Standard Hindsight vector search returns BOTH a deprecated fact and its
replacement when they have high semantic similarity. The LLM then compromises
and hallucinate. Standard Hindsight retain is also unconditional — any
subagent, cron job, or pre-compaction hook can permanently write to the
long-term bank with no human gate.

This plugin adds two primitives on top of the standard Hindsight client:

1. **FORGET** — physically remove a directive or mental_model from the bank.
2. **Steward approval queue** — high-volume automated retains (e.g. the
   pre-compaction hook) get queued in SQLite; a human reviews and decides.

## Tools

| Tool | Purpose |
|---|---|
| `hindsight_forget(memory_id, kind, reason)` | Remove a directive or mental_model. Idempotent. |
| `hindsight_search_to_forget(query, top_k, kind)` | Find candidates by substring/token search. |
| `hindsight_list_recent(limit, kind)` | Browse recent directives + mental_models. |
| `hindsight_propose(content, tags, source, salience, ...)` | Queue a fact for review. |
| `hindsight_review_pending(limit, source, status)` | List facts awaiting review. |
| `hindsight_approve(pending_id, note, reviewed_by)` | Push a pending fact into Hindsight. |
| `hindsight_reject(pending_id, note, reviewed_by)` | Mark a fact rejected. No SDK call. |
| `hindsight_audit_log(limit, op)` | Read the append-only audit log. |
| `hindsight_governance_status()` | Snapshot of counts + kill-switch state. |

## Steward workflow

1. A fact is proposed (manually via `hindsight_propose`, or automatically by
   the pre-compaction hook).
2. A human (or a Telegram bot wired to `hindsight_review_pending`) reads
   the queue.
3. They call `hindsight_approve(pending_id, note="...")` to push the fact
   into Hindsight, or `hindsight_reject(pending_id, note="...")` to drop it.
4. Every op is written to `~/.hermes/state/hindsight_governance_audit.jsonl`.

## Bypass tags

Tags in `HINDSIGHT_GOVERNANCE_BYPASS_TAGS` (default: `_health, _provenance, _trace`)
auto-approve retains without human review. These are operational/observability
facts that should never block on a steward.

## Kill switches

| Env var | Default | Effect when `0` |
|---|---|---|
| `HINDSIGHT_GOVERNANCE_FORGET` | 1 | `hindsight_forget` returns error, no SDK call |
| `HINDSIGHT_GOVERNANCE_QUEUE` | 1 | `hindsight_propose` auto-approves (bypass the queue) |
| `HINDSIGHT_GOVERNANCE_PENDING_DB` | `~/.hermes/state/hindsight_governance.db` | Override DB path |
| `HINDSIGHT_GOVERNANCE_BYPASS_TAGS` | `_health,_provenance,_trace` | Comma-separated auto-approve tags |

## Integration

The pre-compaction hook in `hindsight-hardening` routes its retain calls
through `hindsight_propose` instead of `client.retain` directly. To opt out,
set `HINDSIGHT_GOVERNANCE_QUEUE=0`.

The standard `hindsight_retain` tool is unchanged. High-trust retains (manual,
human-initiated) use the direct tool; low-trust / automated retains should
use `hindsight_propose`.

## Storage

- Pending facts: `~/.hermes/state/hindsight_governance.db` (SQLite, WAL mode)
- Audit log: `~/.hermes/state/hindsight_governance_audit.jsonl` (append-only)
- Auto-expire: pending rows > 7 days old are marked `expired` by sweep
- Auto-prune: `approved > 90d`, `rejected > 30d`, `expired > 30d` are hard-deleted
