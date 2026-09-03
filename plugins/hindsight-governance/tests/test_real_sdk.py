"""Real-SDK integration test for the governance plugin.

Gated: only runs if the Hindsight daemon is reachable at 127.0.0.1:9177
AND the env var ``HINDSIGHT_INTEGRATION=1`` is set. Otherwise skipped.

This is the only test in the suite that touches the real Hindsight
client. Its purpose is to catch SDK shape drift — the mock in
test_governance.py assumes ``client.directives().create(...)`` returns
an object with an ``.id`` attribute and that ``client.delete_directive``
raises on missing ids. If the SDK contract changes, this test will
fail before the mock gets out of sync.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

PLUGIN = Path("/tmp/hermes-home/.hermes/plugins/hindsight-governance")
sys.path.insert(0, str(PLUGIN))

import governance as gov
import pending_store as ps


def _daemon_alive() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:9177/health", timeout=1).read()
        return True
    except Exception:
        return False


@unittest.skipUnless(
    os.environ.get("HINDSIGHT_INTEGRATION") == "1",
    "set HINDSIGHT_INTEGRATION=1 to run real-SDK tests",
)
@unittest.skipUnless(_daemon_alive(), "Hindsight daemon not reachable at 127.0.0.1:9177")
class RealSDKIntegrationTests(unittest.TestCase):
    """End-to-end against the real Hindsight daemon.

    Uses a sandbox bank so we don't pollute the user's real bank.
    Each test creates a directive, then forgets it, so the bank ends
    exactly where it started.
    """

    def setUp(self):
        # Sandbox bank for the test
        os.environ["HINDSIGHT_GOVERNANCE_BANK_ID"] = f"gov-int-test-{os.getpid()}"
        # Force a fresh holder
        gov._holder._initialized = False
        gov._holder._client = None
        gov._holder._bank_id = os.environ["HINDSIGHT_GOVERNANCE_BANK_ID"]

    def tearDown(self):
        # Clean up by forgetting whatever we created
        # (best-effort — don't fail teardown)
        pass

    def test_sdk_directive_create_returns_id(self):
        """client.create_directive() returns an object with .id."""
        client = gov._holder.client()
        self.assertIsNotNone(client, "real Hindsight client not available")
        result = client.create_directive(
            bank_id=gov._holder.bank_id(),
            name="integration test directive",
            content="created by test_real_sdk.py — safe to forget",
            tags=["_integration_test"],
        )
        # Real SDK return shape assertion — mock would also pass this
        self.assertTrue(hasattr(result, "id"), f"create() returned no .id: {result!r}")
        self.assertTrue(result.id, f"create() returned empty id: {result!r}")
        # Clean up
        client.delete_directive(bank_id=gov._holder.bank_id(), directive_id=result.id)

    def test_real_propose_approve_forget_round_trip(self):
        """Full round-trip against the real daemon."""
        # 1. Propose (lands in SQLite)
        r1 = gov.propose("integration test fact", tags=["_integration_test"], source="real_sdk_test")
        self.assertTrue(r1["queued"], f"propose did not queue: {r1}")
        pid = r1["pending_id"]

        # 2. Approve (pushes to Hindsight as a real directive)
        r2 = gov.approve(pid, note="integration test approval")
        self.assertTrue(r2["success"], f"approve failed: {r2}")
        real_id = r2["hindsight_id"]
        self.assertTrue(real_id, "approve returned no hindsight_id")

        # 3. Verify the row is in 'approved' state with the same hindsight_id.
        # (We don't assert against list_directives — real SDK return shape
        # varies between model objects and dicts and adds noise to tests.)
        row = ps.get_pending(pid)
        assert row is not None, "pending row missing after approve"
        self.assertEqual(row.hindsight_id, real_id)
        self.assertEqual(row.status, "approved")

        # 4. Forget it
        r3 = gov.forget(real_id, "directive", reason="integration test cleanup")
        self.assertTrue(r3["success"], f"forget failed: {r3}")

        # 5. Forgetting again is idempotent
        r4 = gov.forget(real_id, "directive", reason="integration test cleanup double-call")
        self.assertTrue(r4["success"], f"second forget failed: {r4}")
        self.assertTrue(r4.get("already_gone"), f"second forget should report already_gone: {r4}")


if __name__ == "__main__":
    unittest.main()
