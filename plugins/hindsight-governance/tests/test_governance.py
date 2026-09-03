"""Tests for the Hindsight governance core (FORGET + approval queue).

All SDK calls are mocked — these tests run offline. The point is to
verify the governance logic (kill switches, idempotency, audit log,
queue→approve flow) without a live Hindsight daemon.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR))

# Isolate DB before importing governance
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="gov_")
_TMP_DB.close()
os.environ["HINDSIGHT_GOVERNANCE_PENDING_DB"] = _TMP_DB.name
os.environ.pop("HINDSIGHT_GOVERNANCE_FORGET", None)
os.environ.pop("HINDSIGHT_GOVERNANCE_QUEUE", None)

import governance as gov
import pending_store as ps


# --- mock helpers ----------------------------------------------------------

class _MockClient:
    """Mimics the subset of Hindsight client surface we use."""
    def __init__(self):
        self._directives = []  # list of attribute-bearing instances
        self._mental_models = []
        self._id_seq = 0
        self.delete_directive_calls = []
        self.delete_mental_model_calls = []
        self.create_directive_calls = []
        self._raise_on_delete: Optional[str] = None

    def _next_id(self):
        self._id_seq += 1
        return f"mock_{self._id_seq}"

    def delete_directive(self, bank_id, directive_id):
        self.delete_directive_calls.append((bank_id, directive_id))
        if self._raise_on_delete:
            raise Exception(self._raise_on_delete)
        before = len(self._directives)
        self._directives = [d for d in self._directives if d.id != directive_id]
        if len(self._directives) == before:
            raise Exception("not found: directive does not exist")

    def delete_mental_model(self, bank_id, mental_model_id):
        self.delete_mental_model_calls.append((bank_id, mental_model_id))
        if self._raise_on_delete:
            raise Exception(self._raise_on_delete)
        before = len(self._mental_models)
        self._mental_models = [m for m in self._mental_models if m.id != mental_model_id]
        if len(self._mental_models) == before:
            raise Exception("not found: mental_model does not exist")

    def list_directives(self, bank_id):
        return list(self._directives)

    def list_mental_models(self, bank_id):
        return list(self._mental_models)

    def create_directive(self, bank_id, name, content, priority=0, is_active=True, tags=None):
        d = _Directive(self._next_id(), name, content, tags or [],
                       "2026-01-02T00:00:00+00:00")
        self._directives.append(d)
        self.create_directive_calls.append({
            "bank_id": bank_id, "name": name, "content": content,
            "priority": priority, "is_active": is_active, "tags": tags,
        })
        return _MockResult(d.id)

    def seed_directive(self, name, content, tags=None):
        d = _Directive(self._next_id(), name, content, tags or [], "2026-01-01T00:00:00+00:00")
        self._directives.append(d)
        return d

    def seed_mental_model(self, name, source_query, tags=None):
        m = _MentalModel(self._next_id(), name, source_query, tags or [], "2026-01-01T00:00:00+00:00")
        self._mental_models.append(m)
        return m


class _Directive:
    def __init__(self, id, name, content, tags, created_at):
        self.id = id
        self.name = name
        self.content = content
        self.tags = tags
        self.created_at = created_at

    # Compat shim: existing test code still does d["id"] in some places
    def __getitem__(self, key):
        return getattr(self, key)


class _MentalModel:
    def __init__(self, id, name, source_query, tags, created_at):
        self.id = id
        self.name = name
        self.source_query = source_query
        self.tags = tags
        self.created_at = created_at

    def __getitem__(self, key):
        return getattr(self, key)


class _MockResult:
    def __init__(self, id_):
        self.id = id_


# --- fixtures --------------------------------------------------------------

class GovernanceTestBase(unittest.TestCase):
    def setUp(self):
        # Wipe the audit log between tests
        if gov.AUDIT_PATH.exists():
            gov.AUDIT_PATH.unlink()
        # Reset the audit-failure counter
        gov._audit_write_failures = 0
        # Reset the singleton so each test gets a fresh client
        gov._holder._initialized = False
        gov._holder._client = None
        gov._holder._bank_id = "test_bank"
        # Reset the bypass lock for tests
        gov._bypass_lock.clear()
        self.mock = _MockClient()

    def tearDown(self):
        # Clean DB rows between tests; init_schema so the table exists
        ps.init_schema()
        with ps._connect() as conn:
            conn.execute("DELETE FROM pending_facts")


# --- FORGET primitive ------------------------------------------------------

class ForgetTests(GovernanceTestBase):
    @patch.object(gov, "_get_client")
    def test_forget_directive_success(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        d = self.mock.seed_directive("old rule", "deprecated")
        res = gov.forget(d["id"], "directive", reason="stale")
        self.assertTrue(res["success"])
        self.assertFalse(res.get("already_gone"))
        self.assertEqual(self.mock.delete_directive_calls, [("test_bank", d["id"])])

    @patch.object(gov, "_get_client")
    def test_forget_mental_model_success(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        m = self.mock.seed_mental_model("user pattern", "user likes X")
        res = gov.forget(m["id"], "mental_model", reason="wrong")
        self.assertTrue(res["success"])
        self.assertEqual(self.mock.delete_mental_model_calls, [("test_bank", m["id"])])

    @patch.object(gov, "_get_client")
    def test_forget_already_gone_is_success(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        # don't seed — id doesn't exist
        res = gov.forget("ghost_id", "directive", reason="cleanup")
        self.assertTrue(res["success"])
        self.assertTrue(res.get("already_gone"))

    @patch.object(gov, "_get_client")
    def test_forget_unexpected_error_returns_error(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        self.mock._raise_on_delete = "connection timeout"
        d = self.mock.seed_directive("a", "b")
        res = gov.forget(d["id"], "directive", reason="x")
        self.assertFalse(res["success"])
        self.assertIn("connection timeout", res["error"])

    def test_forget_invalid_kind(self):
        res = gov.forget("any", "wrong_kind")
        self.assertFalse(res["success"])
        self.assertIn("kind", res["error"])

    def test_forget_kill_switch(self):
        os.environ["HINDSIGHT_GOVERNANCE_FORGET"] = "0"
        try:
            res = gov.forget("any", "directive")
            self.assertFalse(res["success"])
            self.assertIn("disabled", res["error"])
        finally:
            os.environ.pop("HINDSIGHT_GOVERNANCE_FORGET")

    def test_forget_audit_log_written(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            d = self.mock.seed_directive("a", "b")
            gov.forget(d["id"], "directive", reason="stale")
        self.assertTrue(gov.AUDIT_PATH.exists())
        lines = [json.loads(l) for l in gov.AUDIT_PATH.read_text().splitlines() if l]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["op"], "forget")
        self.assertEqual(lines[0]["memory_id"], d["id"])
        self.assertEqual(lines[0]["reason"], "stale")


# --- search-to-forget ------------------------------------------------------

class SearchToForgetTests(GovernanceTestBase):
    @patch.object(gov, "_get_client")
    def test_search_substring_match(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        self.mock.seed_directive("API pricing", "rates as of 2025: $10/mo")
        self.mock.seed_directive("color theme", "use blue")
        res = gov.search_to_forget("pricing", top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["candidates"]), 1)
        self.assertEqual(res["candidates"][0]["name"], "API pricing")
        self.assertEqual(res["candidates"][0]["kind"], "directive")

    @patch.object(gov, "_get_client")
    def test_search_returns_empty_for_no_match(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        self.mock.seed_directive("a", "b")
        res = gov.search_to_forget("xyzzy")
        self.assertTrue(res["success"])
        self.assertEqual(res["candidates"], [])

    @patch.object(gov, "_get_client")
    def test_search_filters_by_kind(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        self.mock.seed_directive("pricing", "rates")
        self.mock.seed_mental_model("pricing pattern", "user asks about pricing")
        res = gov.search_to_forget("pricing", kind="mental_model")
        self.assertEqual(len(res["candidates"]), 1)
        self.assertEqual(res["candidates"][0]["kind"], "mental_model")


# --- propose / approve / reject -------------------------------------------

class QueueTests(GovernanceTestBase):
    def test_propose_inserts_pending_row(self):
        res = gov.propose("API is at /v2/foo", tags=["api"], source="pre_compaction", salience="high")
        self.assertTrue(res["success"])
        self.assertTrue(res["queued"])
        self.assertTrue(res["pending_id"].startswith("pf_"))
        row = ps.get_pending(res["pending_id"])
        assert row is not None
        self.assertEqual(row.content, "API is at /v2/foo")
        self.assertEqual(row.salience, "high")

    def test_propose_bypass_tag_auto_approves(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("trace event", tags=["_health", "noise"], source="auto")
        self.assertTrue(res["success"])
        self.assertTrue(res.get("bypassed"))
        self.assertFalse(res.get("queued"))
        # audit shows approve
        audit = gov.read_audit(limit=10)
        self.assertTrue(any(a["op"] == "approve" and a.get("bypassed") for a in audit))

    def test_propose_queue_kill_switch_auto_approves(self):
        os.environ["HINDSIGHT_GOVERNANCE_QUEUE"] = "0"
        try:
            with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
                res = gov.propose("anything", tags=["normal"], source="manual")
            self.assertTrue(res["success"])
            self.assertTrue(res.get("bypassed"))
        finally:
            os.environ.pop("HINDSIGHT_GOVERNANCE_QUEUE")

    def test_review_pending_filters_status(self):
        f1 = gov.propose("a", source="manual")
        f2 = gov.propose("b", source="manual")
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            gov.approve(f2["pending_id"])
        pending = gov.review_pending(status="pending")
        approved = gov.review_pending(status="approved")
        self.assertEqual([p["pending_id"] for p in pending["items"]], [f1["pending_id"]])
        self.assertEqual([p["pending_id"] for p in approved["items"]], [f2["pending_id"]])

    def test_approve_calls_directives_create(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("the API is at /v2/foo", tags=["api"], salience="high")
            self.assertTrue(res["queued"])
            appr = gov.approve(res["pending_id"], note="verified")
        self.assertTrue(appr["success"])
        self.assertIn(appr["hindsight_id"], [d["id"] for d in self.mock._directives])
        # create was called with the right content + tags
        cc = self.mock.create_directive_calls[0]
        self.assertEqual(cc["content"], "the API is at /v2/foo")
        self.assertIn("steward_approved", cc["tags"])
        self.assertIn("api", cc["tags"])

    def test_approve_idempotent_double_approve(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("fact")
            gov.approve(res["pending_id"])
            second = gov.approve(res["pending_id"])
        self.assertTrue(second["already_approved"])
        self.assertEqual(len(self.mock.create_directive_calls), 1)

    def test_approve_concurrent_race_atomic(self):
        """Two concurrent approves — the Hindsight bank ends up with exactly
        1 directive and the row is in 'approved' state.

        Simulates the TOCTOU race: two threads both pass the initial
        ``status=='pending'`` guard. Whichever loses the SQLite compare-and-set
        on 'approving' (or 'approved') gets ``already_approved: true`` with
        the winner's hindsight_id — no orphan directive.
        """
        import threading
        results: list[dict] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def race_approve(pid):
            with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
                # Both threads wait until they are simultaneously at the
                # critical section, maximizing the chance of a real race.
                barrier.wait(timeout=2)
                r = gov.approve(pid, note="racing")
                with lock:
                    results.append(r)

        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("race target")
            pid = res["pending_id"]

            t1 = threading.Thread(target=race_approve, args=(pid,))
            t2 = threading.Thread(target=race_approve, args=(pid,))
            t1.start(); t2.start()
            t1.join(); t2.join()

        # The bank ended up with exactly 1 directive for this fact
        # (counted by content match — both threads target the same content).
        bank_directives = [d for d in self.mock._directives
                          if d.content == "race target"]
        self.assertEqual(len(bank_directives), 1,
            f"expected 1 directive in bank, got {len(bank_directives)}: "
            f"{[d.id for d in bank_directives]}")
        # Pending row is in 'approved' state with the same hindsight_id
        row = ps.get_pending(pid)
        assert row is not None
        self.assertEqual(row.status, "approved")
        self.assertEqual(row.hindsight_id, bank_directives[0].id)
        # create_directive_calls == 1 (no orphan)
        self.assertEqual(len(self.mock.create_directive_calls), 1)
        # At least one of the two results succeeded
        successes = [r for r in results if r.get("success")]
        self.assertGreaterEqual(len(successes), 1)

    def test_approve_missing_id(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.approve("pf_nope")
        self.assertFalse(res["success"])
        self.assertIn("not found", res["error"])

    def test_reject_no_sdk_call(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)) as m_get:
            res = gov.propose("bad fact")
            gov.reject(res["pending_id"], note="not durable")
        row = ps.get_pending(res["pending_id"])
        assert row is not None
        self.assertEqual(row.status, "rejected")
        self.assertEqual(row.review_note, "not durable")
        # no create_directive call
        self.assertEqual(self.mock.create_directive_calls, [])

    def test_reject_already_rejected_idempotent(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("x")
            gov.reject(res["pending_id"])
            second = gov.reject(res["pending_id"])
        self.assertTrue(second["already_rejected"])

    def test_reject_after_approve_blocked(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            res = gov.propose("y")
            gov.approve(res["pending_id"])
            rej = gov.reject(res["pending_id"])
        self.assertFalse(rej["success"])

    def test_approve_batch_all_success(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            pids = [gov.propose(f"fact {i}")["pending_id"] for i in range(3)]
            r = gov.approve_batch(pids, note="batch verify")
        self.assertTrue(r["success"])
        self.assertEqual(r["approved"], 3)
        self.assertEqual(r["skipped"], 0)
        self.assertEqual(r["failed"], 0)
        self.assertEqual(len(r["results"]), 3)
        self.assertEqual(len(self.mock.create_directive_calls), 3)

    def test_approve_batch_partial_failure(self):
        with patch.object(gov, "_get_client", return_value=(self.mock, "test_bank", None)):
            good = gov.propose("good fact")["pending_id"]
            # One valid, one nonexistent
            r = gov.approve_batch([good, "pf_nope"], note="mixed")
        self.assertFalse(r["success"])
        self.assertEqual(r["approved"], 1)
        self.assertEqual(r["failed"], 1)

    def test_approve_batch_empty(self):
        r = gov.approve_batch([], note="x")
        self.assertTrue(r["success"])
        self.assertEqual(r["approved"], 0)

    def test_reject_batch_all_success(self):
        pids = [gov.propose(f"r {i}")["pending_id"] for i in range(2)]
        r = gov.reject_batch(pids, note="not relevant")
        self.assertTrue(r["success"])
        self.assertEqual(r["rejected"], 2)

    def test_review_pending_compact_format(self):
        # Insert a known-pending row
        ps.insert_pending(content="a very long fact " * 20, source="manual", salience="high")
        r = gov.review_pending_compact(limit=5)
        self.assertTrue(r["success"])
        self.assertGreater(r["count"], 0)
        first = r["lines"][0]
        self.assertIn("🔴", first["summary"])  # high salience badge
        self.assertIn("…", first["summary"])    # truncated
        self.assertTrue(first["pending_id"].startswith("pf_"))
        self.assertIn("hindsight_approve_batch", r["hint"])

    def test_audit_propose_written(self):
        gov.propose("test", source="manual", tags=["x"])
        audit = gov.read_audit(op="propose", limit=10)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["source"], "manual")
        self.assertIn("content_hash", audit[0])

    def test_concurrent_propose_no_collision(self):
        """N concurrent propose() calls each get a unique pending_id.

        Race: SQLite PRIMARY KEY on pending_id could collide if UUID
        generation is racy. Also tests that no row is dropped.
        """
        import threading
        N = 20
        barrier = threading.Barrier(N)
        results: list[dict] = [None] * N  # type: ignore

        def worker(i: int) -> None:
            barrier.wait()  # release all threads simultaneously
            r = gov.propose(
                f"concurrent fact {i}",
                tags=["concurrent"],
                source="race_test",
            )
            results[i] = r

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All N calls succeeded
        self.assertEqual(len(results), N)
        self.assertTrue(
            all(r is not None and r.get("success") for r in results),
            f"some proposes failed: {results}",
        )
        self.assertTrue(
            all(r is not None and r.get("queued") for r in results),
            f"some proposes not queued: {results}",
        )
        # Every pending_id is unique (UUID4 PK no-collision)
        ids = [r["pending_id"] for r in results if r]
        self.assertEqual(len(set(ids)), N, f"collision in pending_ids: {ids}")
        # All N rows landed in SQLite
        for pid in ids:
            row = ps.get_pending(pid)
            self.assertIsNotNone(row, f"row missing for {pid}")
            self.assertEqual(row.status, "pending")


# --- audit-failure counter -----------------------------------------------


class AuditCounterTests(GovernanceTestBase):
    def test_counter_starts_at_zero(self):
        self.assertEqual(gov._audit_write_failures, 0)
        s = gov.governance_status()
        self.assertEqual(s["audit_write_failures"], 0)

    def test_counter_increments_on_write_failure(self):
        import pathlib
        before = gov._audit_write_failures
        real = pathlib.Path.open

        def boom(self, *a, **kw):
            if "hindsight_governance_audit" in str(self):
                raise IOError("simulated disk full")
            return real(self, *a, **kw)

        pathlib.Path.open = boom
        try:
            gov._audit("test_op", note="failure injection")
        finally:
            pathlib.Path.open = real
        self.assertEqual(gov._audit_write_failures, before + 1)
        s = gov.governance_status()
        self.assertEqual(s["audit_write_failures"], before + 1)

    def test_audit_path_exposed_in_status(self):
        s = gov.governance_status()
        self.assertIn("audit_path", s)
        self.assertTrue(str(s["audit_path"]).endswith("hindsight_governance_audit.jsonl"))

    def test_count_helpers_in_status(self):
        s = gov.governance_status()
        self.assertIn("counts", s)
        for k in ("pending", "approved", "rejected", "expired"):
            self.assertIn(k, s["counts"])
            self.assertIsInstance(s["counts"][k], int)


# --- bypass CAS (concurrent bypass retain coalescing) --------------------


class BypassCASTests(GovernanceTestBase):
    """The bypass path (propose with bypass tag) goes straight to Hindsight
    with no SQLite row. Concurrent bypass retains of the same content must
    not double-create. The in-memory _bypass_lock provides the CAS.
    """

    @patch.object(gov, "_get_client")
    def test_bypass_cas_coalesces_concurrent_retains(self, m_get):
        m_get.return_value = (self.mock, "test_bank", None)
        import threading
        N = 10
        barrier = threading.Barrier(N)
        results: dict[int, dict] = {}

        def worker() -> None:
            barrier.wait()
            r = gov.propose(
                "bypass race fact",
                tags=["_health"],  # bypass tag
                source="race_test",
            )
            results[threading.get_ident()] = r

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # At most ONE call should have created a directive.
        self.assertEqual(
            len(self.mock.create_directive_calls), 1,
            f"expected 1 create_directive, got {len(self.mock.create_directive_calls)}",
        )
        # All N calls returned success (the coalesced ones report coalesced=True)
        self.assertEqual(len(results), N, f"some workers didn't report: {results}")
        n_success = sum(1 for r in results.values() if r["success"])
        self.assertEqual(n_success, N, f"some bypass calls failed: {results}")

    @patch.object(gov, "_get_client")
    def test_bypass_cas_coalesced_callers_get_winner_hid(self, m_get):
        """When the bypass CAS coalesces, every caller (winner + losers)
        must see the winner's hindsight_id. Previously losers got "".
        """
        m_get.return_value = (self.mock, "test_bank", None)
        # Add a deliberate small delay inside create_directive so the
        # coalesced callers spin and the winner's id is recorded before
        # the losers check.
        original = self.mock.create_directive
        hid_captured: list[str] = []
        def slow_create(**kwargs):
            import time as _t
            _t.sleep(0.1)  # let losers race in
            res = original(**kwargs)
            hid = getattr(res, "id", None) or ""
            hid_captured.append(hid)
            return res
        self.mock.create_directive = slow_create
        try:
            N = 10
            import threading
            barrier = threading.Barrier(N)
            results: dict[int, dict] = {}
            def worker():
                barrier.wait()
                results[threading.get_ident()] = gov.propose(
                    "hid-share fact", tags=["_health"], source="share_test",
                )
            threads = [threading.Thread(target=worker) for _ in range(N)]
            for t in threads: t.start()
            for t in threads: t.join(timeout=5)
        finally:
            self.mock.create_directive = original

        # All N got success=True
        self.assertEqual(len(results), N)
        for r in results.values():
            self.assertTrue(r["success"])
        # At most 1 directive was created
        self.assertEqual(
            len(self.mock.create_directive_calls), 1,
            f"expected 1 create, got {len(self.mock.create_directive_calls)}",
        )
        # All N callers (winner + 9 losers) got the SAME non-empty hindsight_id
        winner_hid = hid_captured[0]
        self.assertTrue(winner_hid, f"winner hid empty: {hid_captured}")
        hids = [r["hindsight_id"] for r in results.values()]
        self.assertEqual(set(hids), {winner_hid}, f"callers see different hids: {hids}")
        # Exactly one caller reports coalesced=False, the rest True
        n_winner = sum(1 for r in results.values() if r.get("hindsight_id") and not r.get("coalesced"))
        n_coalesced = sum(1 for r in results.values() if r.get("coalesced"))
        self.assertEqual(n_winner + n_coalesced, N)
        self.assertGreaterEqual(n_winner, 1, "no caller was the winner")
        self.assertEqual(n_coalesced, N - 1, "wrong number of coalesced callers")

    @patch.object(gov, "_get_client")
    def test_bypass_cas_releases_on_failure(self, m_get):
        """A failed bypass create should release the lock so a retry works."""
        m_get.return_value = (self.mock, "test_bank", None)
        # Replace create_directive with a function that always raises
        original = self.mock.create_directive

        def boom(**kwargs):
            raise RuntimeError("simulated SDK error")

        self.mock.create_directive = boom
        try:
            r1 = gov.propose("retry fact", tags=["_health"], source="retry_test")
            self.assertFalse(r1["success"])
            r2 = gov.propose("retry fact", tags=["_health"], source="retry_test")
            self.assertFalse(r2["success"])
        finally:
            self.mock.create_directive = original

    def test_audit_failure_counter_persists_across_imports(self):
        """The audit-failure counter survives process restarts via
        AUDIT_FAILURES_PATH. This protects users from silent loss after
        a gateway restart.
        """
        import importlib
        import governance as g2
        # Force a known value
        g2._audit_write_failures = 42
        g2.AUDIT_FAILURES_PATH.write_text("42", encoding="utf-8")
        # Reimport to simulate process restart
        importlib.reload(g2)
        self.assertEqual(g2._audit_write_failures, 42,
                         "counter did not persist across import")
        # Cleanup
        g2.AUDIT_FAILURES_PATH.unlink(missing_ok=True)
        importlib.reload(g2)
        self.assertEqual(g2._audit_write_failures, 0,
                         "counter should reset when file removed")


# --- pre-compaction integration ------------------------------------------

class PreCompactionIntegrationTests(GovernanceTestBase):
    def test_propose_caller_routes_through_queue(self):
        caller = gov.make_propose_retain_caller(source="pre_compaction", session_id="sess_x")
        caller({"content": "API is at /v2", "salience": "high", "tags": ["api"]})
        # Should appear in pending queue
        pending = gov.review_pending(status="pending")
        self.assertEqual(len(pending["items"]), 1)
        self.assertEqual(pending["items"][0]["content"], "API is at /v2")
        self.assertEqual(pending["items"][0]["source"], "pre_compaction")
        self.assertEqual(pending["items"][0]["session_id"], "sess_x")
        self.assertIn("pre_compaction", pending["items"][0]["tags"])

    def test_propose_caller_with_pre_compaction_module(self):
        """End-to-end: pre_compaction.extract_and_dedupe with the propose caller."""
        # Add the pre_compaction module to path
        hc_dir = Path("/tmp/hermes-home/.hermes/plugins/hindsight-hardening")
        if str(hc_dir) not in sys.path:
            sys.path.insert(0, str(hc_dir))
        try:
            import pre_compaction  # type: ignore
        except Exception as e:
            self.skipTest(f"pre_compaction module not importable: {e}")

        caller = gov.make_propose_retain_caller(source="pre_compaction", session_id="sess_e2e")
        # Mock LLM that returns 1 fact
        def fake_llm(system, user):
            return json.dumps({"facts": [{"entity": "API", "relation": "endpoint", "value": "/v2/foo", "salience": "high"}]})
        result = pre_compaction.extract_and_dedupe(
            messages=[{"role": "user", "content": "I think the API is at /v2/foo"}],
            llm_caller=fake_llm,
            retain_caller=caller,
            session_id="sess_e2e",
            source="pre_compaction",
        )
        self.assertEqual(len(result.facts), 1)
        # The fact should be in pending queue, not directly in Hindsight
        pending = gov.review_pending(status="pending", source="pre_compaction")
        self.assertEqual(len(pending["items"]), 1)
        self.assertIn("/v2/foo", pending["items"][0]["content"])

    def test_propose_caller_swallows_exceptions(self):
        caller = gov.make_propose_retain_caller()
        # Should not raise even with bad payload
        try:
            caller({"content": "", "tags": None, "salience": "high"})
        except Exception as e:
            self.fail(f"caller should not raise: {e}")


# --- status snapshot ------------------------------------------------------

class StatusTests(GovernanceTestBase):
    def test_governance_status_shape(self):
        res = gov.governance_status()
        self.assertTrue(res["success"])
        self.assertIn("bank_id", res)
        self.assertIn("kill_switches", res)
        self.assertIn("counts", res)
        self.assertEqual(res["kill_switches"]["FORGET"], True)
        self.assertEqual(res["kill_switches"]["QUEUE"], True)
        # Default bypass tags = just _health
        self.assertEqual(res["bypass_tags"], ["_health"])
        # action hint
        self.assertIn("action", res)
        # queue is empty in this test
        self.assertEqual(res["counts"]["pending"], 0)
        self.assertEqual(res["action"], "queue is empty")

    def test_bypass_default_excludes_provenance(self):
        """Default bypass tags must NOT include _provenance — that's a footgun."""
        # unset env to test the default
        os.environ.pop("HINDSIGHT_GOVERNANCE_BYPASS_TAGS", None)
        # reimport the module to pick up the env
        import importlib
        import governance as g2
        importlib.reload(g2)
        self.assertEqual(g2._bypass_tags(), ["_health"])
        # restore
        os.environ["HINDSIGHT_GOVERNANCE_BYPASS_TAGS"] = "_health"
        importlib.reload(g2)


if __name__ == "__main__":
    unittest.main()
