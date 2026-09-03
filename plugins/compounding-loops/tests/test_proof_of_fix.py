"""Tests for #8: proof-of-fix (revert-or-mutate) gate mode."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_HERMES_HOME = _PLUGIN_DIR.parent.parent


@pytest.fixture(scope="module")
def plugin():
    os.environ["HERMES_LOOPS_ENABLED"] = "1"
    os.environ["HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN"] = "1"
    os.environ["HERMES_LOOPS_MAX_PASSES"] = "9"
    os.environ.pop("HERMES_LOOPS_REVIEW_TOOLS", None)
    os.environ.pop("HERMES_LOOPS_PROOF_OF_FIX", None)
    os.environ.pop("HERMES_LOOPS_MAX_TURNS", None)
    os.environ.pop("HERMES_LOOPS_CIRCUIT_BREAKER", None)
    if str(_HERMES_HOME) not in sys.path:
        sys.path.insert(0, str(_HERMES_HOME))
    spec = importlib.util.spec_from_file_location(
        "plugins.cl_pof", _PLUGIN_DIR / "__init__.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["plugins.cl_pof"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def proof_mode(monkeypatch):
    monkeypatch.setenv("HERMES_LOOPS_PROOF_OF_FIX", "1")


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    p = tmp_path / "STATUS.json"
    monkeypatch.setenv("HERMES_LOOP_STATE_FILE", str(p))
    return p


def _pc(cid, name, args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


_PROOF_BLOCK = """Proof-of-fix fix-1:
  test: test_auth_csrf
  red-before: fails with CSRFError
  green-after: passes
  revert-verified: yes"""


def test_proof_of_fix_rejects_clean_pass_without_proofs(plugin, proof_mode, state_file):
    """A clean pass after iterations must include verified proof blocks."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker. Fixed."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify(
        "Review pass 3: 0 blockers, 0 majors. Done.", messages=msgs
    )
    assert decision["approved"] is False
    assert "proof-of-fix" in decision["reason"].lower()


def test_proof_of_fix_approves_with_proofs(plugin, proof_mode, state_file):
    """A clean pass WITH a verified proof block is approved."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": f"Review pass 1: 1 blocker. Fixed.\n{_PROOF_BLOCK}"},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify(
        f"Review pass 3: 0 blockers, 0 majors. Done.\n{_PROOF_BLOCK}",
        messages=msgs,
    )
    assert decision["approved"] is True


def test_proof_of_fix_rejects_unverified_proof(plugin, proof_mode, state_file):
    """A proof block with revert-verified: no is not counted."""
    unverified = """Proof-of-fix fix-1:
  test: test_auth_csrf
  red-before: fails
  green-after: passes
  revert-verified: no"""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": f"Review pass 1: 1 blocker. Fixed.\n{unverified}"},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify(
        f"Review pass 3: 0 blockers, 0 majors. Done.", messages=msgs
    )
    assert decision["approved"] is False
    assert "proof-of-fix" in decision["reason"].lower()


def test_proof_of_fix_disabled_by_default(plugin, state_file):
    """Without HERMES_LOOPS_PROOF_OF_FIX, no proof requirement."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 1 blocker. Fixed."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 3: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify(
        "Review pass 3: 0 blockers, 0 majors. Done.", messages=msgs
    )
    # No proof requirement — approved normally.
    assert decision["approved"] is True


def test_proof_of_fix_not_required_on_first_pass(plugin, proof_mode, state_file):
    """If highest_pass == 1 (no iterations), no proof required even in
    proof mode — there was nothing to fix."""
    msgs = [
        {"role": "user", "content": "Implement X"},
        {"role": "assistant", "content": None,
         "tool_calls": [_pc("w1", "write_file"), _pc("w2", "patch")]},
        {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        {"role": "tool", "tool_call_id": "w2", "content": "ok"},
        {"role": "assistant", "content": "Review pass 1: 0 blockers, 0 majors."},
        {"role": "assistant", "content": "Review pass 2: 0 blockers, 0 majors."},
    ]
    decision = plugin._pre_exit_verify(
        "Review pass 2: 0 blockers, 0 majors. Done.", messages=msgs
    )
    assert decision["approved"] is True


def test_count_verified_proofs_parses_multiple(plugin):
    """Multiple proof blocks are counted correctly."""
    text = """Review pass 2: 0 blockers.

Proof-of-fix fix-1:
  test: test_a
  red-before: fail
  green-after: pass
  revert-verified: yes

Proof-of-fix fix-2:
  test: test_b
  red-before: fail
  green-after: pass
  revert-verified: yes

Proof-of-fix fix-3:
  test: test_c
  red-before: fail
  green-after: pass
  revert-verified: no
"""
    count = plugin._count_verified_proofs(text, [])
    assert count == 2  # fix-3 is unverified