"""Tests for the Hindsight governance pending store."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Plugin lives in ~/.hermes/plugins/hindsight-governance — add it to path
HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR))

import pending_store as ps


class PendingStoreTests(unittest.TestCase):
    def setUp(self):
        # Isolate each test in its own tempfile DB
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False, prefix="pending_"
        )
        self.tmp.close()
        os.environ["HINDSIGHT_GOVERNANCE_PENDING_DB"] = self.tmp.name
        ps.init_schema()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass
        os.environ.pop("HINDSIGHT_GOVERNANCE_PENDING_DB", None)

    def test_schema_creates_tables(self):
        # init_schema ran in setUp; verify columns exist
        with ps._connect() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(pending_facts)").fetchall()]
        for c in ("id", "pending_id", "content", "status", "expires_at", "tags"):
            self.assertIn(c, cols)

    def test_insert_returns_full_row(self):
        f = ps.insert_pending(content="Ryan prefers caveman", source="manual", salience="high")
        self.assertTrue(f.pending_id.startswith("pf_"))
        self.assertEqual(f.status, "pending")
        self.assertEqual(f.content, "Ryan prefers caveman")
        self.assertEqual(f.salience, "high")
        self.assertTrue(f.expires_at)

    def test_insert_persists_round_trip(self):
        f = ps.insert_pending(
            content="API endpoint is /v2/foo", source="pre_compaction",
            tags=["api", "endpoint"], entities=["api", "endpoint"], salience="medium",
            session_id="sess_xyz", proposed_by="hermes",
        )
        got = ps.get_pending(f.pending_id)
        assert got is not None
        self.assertEqual(got.content, "API endpoint is /v2/foo")
        self.assertEqual(got.tags, ["api", "endpoint"])
        self.assertEqual(got.entities, ["api", "endpoint"])
        self.assertEqual(got.session_id, "sess_xyz")
        self.assertEqual(got.proposed_by, "hermes")

    def test_get_pending_missing_returns_none(self):
        self.assertIsNone(ps.get_pending("pf_doesnotexist"))

    def test_list_pending_filters_by_status(self):
        f1 = ps.insert_pending(content="a", source="manual")
        f2 = ps.insert_pending(content="b", source="manual")
        ps.update_status(f2.pending_id, status="approved", reviewed_by="human")
        pending = ps.list_pending(status="pending")
        approved = ps.list_pending(status="approved")
        self.assertEqual([p.pending_id for p in pending], [f1.pending_id])
        self.assertEqual([p.pending_id for p in approved], [f2.pending_id])

    def test_list_pending_filters_by_source(self):
        ps.insert_pending(content="x", source="pre_compaction")
        ps.insert_pending(content="y", source="manual")
        self.assertEqual(
            len(ps.list_pending(source="pre_compaction")), 1
        )
        self.assertEqual(
            len(ps.list_pending(source="manual")), 1
        )

    def test_update_status_records_metadata(self):
        f = ps.insert_pending(content="c", source="manual")
        got = ps.update_status(
            f.pending_id, status="rejected",
            reviewed_by="ryan", review_note="not a fact", hindsight_id=None,
        )
        assert got is not None
        self.assertEqual(got.status, "rejected")
        self.assertEqual(got.reviewed_by, "ryan")
        self.assertEqual(got.review_note, "not a fact")
        self.assertIsNotNone(got.reviewed_at)

    def test_sweep_expired_marks_old_rows(self):
        f = ps.insert_pending(content="old", source="manual", ttl_days=0)
        # Backdate expires_at to the past
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with ps._connect() as conn:
            conn.execute(
                "UPDATE pending_facts SET expires_at = ? WHERE pending_id = ?",
                (past, f.pending_id),
            )
        n = ps.sweep_expired()
        self.assertEqual(n, 1)
        got = ps.get_pending(f.pending_id)
        assert got is not None
        self.assertEqual(got.status, "expired")

    def test_sweep_expired_does_not_touch_fresh_rows(self):
        ps.insert_pending(content="fresh", source="manual", ttl_days=7)
        n = ps.sweep_expired()
        self.assertEqual(n, 0)

    def test_prune_old_deletes_aged_terminal_rows(self):
        f = ps.insert_pending(content="c", source="manual")
        ps.update_status(f.pending_id, status="approved", reviewed_by="r")
        # Backdate reviewed_at by 100 days
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        with ps._connect() as conn:
            conn.execute(
                "UPDATE pending_facts SET reviewed_at = ? WHERE pending_id = ?",
                (old, f.pending_id),
            )
        n = ps.prune_old(approved_days=90)
        self.assertEqual(n, 1)
        self.assertIsNone(ps.get_pending(f.pending_id))

    def test_salience_coerced_to_valid(self):
        f = ps.insert_pending(content="x", source="manual", salience="banana")
        self.assertEqual(f.salience, "medium")


if __name__ == "__main__":
    unittest.main()
