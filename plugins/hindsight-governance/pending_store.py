"""SQLite-backed pending-facts queue for Hindsight memory governance.

The approval queue backs the steward workflow: any `hindsight_propose` call
inserts a row here; the row sits until a human (or an automated steward
under the bypass policy) calls `approve` or `reject`.

Schema is intentionally minimal — a single table with a status index. The
governance layer adds the audit log (append-only JSONL) on top of this.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

DEFAULT_DB_PATH = Path(os.path.expanduser("~/.hermes/state/hindsight_governance.db"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pending_id    TEXT UNIQUE NOT NULL,
    proposed_at   TEXT NOT NULL,
    source        TEXT NOT NULL,
    session_id    TEXT,
    proposed_by   TEXT,
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '[]',
    entities      TEXT NOT NULL DEFAULT '[]',
    salience      TEXT NOT NULL DEFAULT 'medium',
    status        TEXT NOT NULL DEFAULT 'pending',
    reviewed_at   TEXT,
    reviewed_by   TEXT,
    review_note   TEXT,
    hindsight_id  TEXT,
    expires_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_facts(status, proposed_at);
CREATE INDEX IF NOT EXISTS idx_pending_expires ON pending_facts(expires_at);
"""


def _db_path() -> Path:
    override = os.environ.get("HINDSIGHT_GOVERNANCE_PENDING_DB")
    if override:
        return Path(override)
    return DEFAULT_DB_PATH


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    """Create tables/indexes if missing. Idempotent."""
    with _connect() as conn:
        conn.executescript(SCHEMA)


@dataclass
class PendingFact:
    pending_id: str
    proposed_at: str
    source: str
    content: str
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    salience: str = "medium"
    status: str = "pending"
    session_id: Optional[str] = None
    proposed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None
    hindsight_id: Optional[str] = None
    expires_at: str = ""
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PendingFact":
        d = dict(row)
        for k in ("tags", "entities"):
            v = d.get(k) or "[]"
            try:
                d[k] = json.loads(v) if isinstance(v, str) else (v or [])
            except json.JSONDecodeError:
                d[k] = []
        return cls(**d)


def _new_pending_id() -> str:
    return f"pf_{uuid.uuid4().hex[:16]}"


def insert_pending(
    *,
    content: str,
    source: str,
    tags: list[str] | None = None,
    entities: list[str] | None = None,
    salience: str = "medium",
    session_id: Optional[str] = None,
    proposed_by: Optional[str] = None,
    ttl_days: int = 7,
) -> PendingFact:
    """Insert a new pending fact. Returns the stored row (with pending_id and expires_at)."""
    init_schema()
    if salience not in ("high", "medium", "low"):
        salience = "medium"
    now = _now_utc()
    expires = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
    pid = _new_pending_id()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pending_facts
                (pending_id, proposed_at, source, session_id, proposed_by,
                 content, tags, entities, salience, status, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                pid, now, source, session_id, proposed_by,
                content, json.dumps(tags or []), json.dumps(entities or []),
                salience, expires,
            ),
        )
    return PendingFact(
        pending_id=pid, proposed_at=now, source=source, content=content,
        tags=tags or [], entities=entities or [], salience=salience,
        session_id=session_id, proposed_by=proposed_by, expires_at=expires,
    )


def get_pending(pending_id: str) -> Optional[PendingFact]:
    init_schema()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pending_facts WHERE pending_id = ?", (pending_id,)
        ).fetchone()
    return PendingFact.from_row(row) if row else None


def count_pending() -> int:
    """Return the count of rows in 'pending' status.

    Used by governance_status() to surface queue depth without loading
    the full row bodies. Cheap (indexed COUNT).
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM pending_facts WHERE status = 'pending'"
        ).fetchone()[0]


def count_by_status(status: str) -> int:
    """Return the count of rows in the given status."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM pending_facts WHERE status = ?", (status,)
        ).fetchone()[0]


def list_pending(
    *,
    status: str = "pending",
    source: Optional[str] = None,
    limit: int = 50,
) -> list[PendingFact]:
    init_schema()
    sql = "SELECT * FROM pending_facts WHERE status = ?"
    args: list[Any] = [status]
    if source:
        sql += " AND source = ?"
        args.append(source)
    sql += " ORDER BY proposed_at ASC LIMIT ?"
    args.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [PendingFact.from_row(r) for r in rows]


def update_status(
    pending_id: str,
    *,
    status: str,
    reviewed_by: Optional[str] = None,
    review_note: Optional[str] = None,
    hindsight_id: Optional[str] = None,
    require_prior_status: Optional[str] = None,
) -> Optional[PendingFact]:
    """Update a pending row.

    ``require_prior_status`` enforces a compare-and-set: the UPDATE only
    fires if the row's current status matches. Returns the row on
    success, or None if the row was not found / the prior-status check
    failed. The caller can detect the latter by ``get_pending()`` showing
    a different status than expected.

    Used by ``approve()`` and ``reject()`` to prevent TOCTOU races
    (e.g. two concurrent approves both passing the status=='pending'
    guard and both calling ``directives().create()``).
    """
    init_schema()
    now = _now_utc()
    sql = (
        "UPDATE pending_facts "
        "SET status = ?, reviewed_at = ?, reviewed_by = ?, review_note = ?, "
        "    hindsight_id = COALESCE(?, hindsight_id)"
    )
    args: list[Any] = [status, now, reviewed_by, review_note, hindsight_id]
    if require_prior_status is not None:
        sql += " WHERE pending_id = ? AND status = ?"
        args.extend([pending_id, require_prior_status])
    else:
        sql += " WHERE pending_id = ?"
        args.append(pending_id)
    with _connect() as conn:
        cur = conn.execute(sql, args)
        if cur.rowcount == 0 and require_prior_status is not None:
            # CAS failed — row either gone or in a different state
            return None
        row = conn.execute(
            "SELECT * FROM pending_facts WHERE pending_id = ?", (pending_id,)
        ).fetchone()
    return PendingFact.from_row(row) if row else None


def sweep_expired(now: Optional[str] = None) -> int:
    """Mark all pending rows whose expires_at < now as 'expired'.

    Returns the count of rows transitioned.
    """
    init_schema()
    now = now or _now_utc()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE pending_facts SET status = 'expired', reviewed_at = ?, "
            "review_note = 'expired without review' "
            "WHERE status = 'pending' AND expires_at < ?",
            (now, now),
        )
    return cur.rowcount or 0


def prune_old(
    *,
    approved_days: int = 90,
    rejected_days: int = 30,
    expired_days: int = 30,
) -> int:
    """Hard-delete old terminal rows. Returns the count deleted."""
    init_schema()
    now = datetime.now(timezone.utc)
    with _connect() as conn:
        total = 0
        for status, days in [
            ("approved", approved_days),
            ("rejected", rejected_days),
            ("expired", expired_days),
        ]:
            cutoff = (now - timedelta(days=days)).isoformat()
            cur = conn.execute(
                "DELETE FROM pending_facts WHERE status = ? AND reviewed_at < ?",
                (status, cutoff),
            )
            total += cur.rowcount or 0
    return total


__all__ = [
    "PendingFact",
    "init_schema",
    "insert_pending",
    "get_pending",
    "list_pending",
    "update_status",
    "sweep_expired",
    "prune_old",
    "DEFAULT_DB_PATH",
]
