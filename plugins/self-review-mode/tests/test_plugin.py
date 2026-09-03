"""Tests for self-review-mode plugin wiring — tools, hooks, mode isolation.

Resolves the plugin dir from this file's location, so it runs both standalone
under ~/.hermes/plugins/self-review-mode and inside the agent tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture()
def plugin(monkeypatch, tmp_path):
    """Load the plugin module fresh with an isolated mode file."""
    monkeypatch.setenv("SELF_REVIEW_ENABLED", "1")
    monkeypatch.setenv("SELF_REVIEW_MODE_FILE", str(tmp_path / "mode.json"))
    monkeypatch.delenv("SELF_REVIEW_SUFFIX", raising=False)
    spec = importlib.util.spec_from_file_location(
        "plugins.self_review_mode", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ctx():
    """A minimal register()-compatible context recording tools + hooks."""
    class Ctx:
        def __init__(self):
            self.tools = {}
            self.hooks = {}

        def register_tool(self, name, toolset=None, schema=None, handler=None):
            self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}

        def register_hook(self, name, fn):
            self.hooks[name] = fn

    return Ctx()


# --- self_review tool -------------------------------------------------------

def test_self_review_emits_speculator_card(plugin, tmp_path):
    out = json.loads(plugin._self_review_handler({"target": "auth handler"}))
    assert out["mode"] == "SPECULATOR"
    assert "[SELF-REVIEW SPECULATOR MODE]" in out["suffix"]
    assert "BUILDER" in out["mode_isolation_card"]["roles"]
    assert out["target"] == "auth handler"
    # disk marker written
    assert json.loads((tmp_path / "mode.json").read_text())["mode"] == "SPECULATOR"


def test_self_review_uses_configured_mode_file(plugin, tmp_path):
    plugin._self_review_handler({})
    assert (tmp_path / "mode.json").exists()


def test_self_review_disabled(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_ENABLED", "0")
    out = json.loads(plugin._self_review_handler({}))
    assert out["enabled"] is False


# --- self_review_score tool -------------------------------------------------

def test_self_review_score_happy_path(plugin):
    res = json.loads(plugin._self_review_score_handler({
        "gold_findings": [{"severity": "blocker", "evidence": "null deref parse"}],
        "predicted_findings": [{"severity": "blocker", "evidence": "null deref check"}],
        "pass_new_finding_counts": [3, 0, 0],
        "window": 2,
    }))
    # handler rounds alignment_score to 4dp -> 0.6667
    assert res["alignment_score"] == pytest.approx(2 / 3, abs=1e-3)
    assert res["miss_breakdown"]["hits"] == 1
    assert res["thrash_detected"] is True


def test_self_review_score_overlenient(plugin):
    res = json.loads(plugin._self_review_score_handler({
        "gold_findings": [{"severity": "blocker", "evidence": "x"}],
        "predicted_findings": [{"severity": "minor", "evidence": "x"}],
    }))
    assert res["alignment_score"] == -1.0
    assert res["miss_breakdown"]["blocker_misses"] == 1


def test_self_review_score_invalid_finding_returns_error(plugin):
    res = json.loads(plugin._self_review_score_handler({
        "gold_findings": [{"severity": "critical", "evidence": "x"}],
        "predicted_findings": [],
    }))
    assert "error" in res


def test_self_review_score_disabled(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_ENABLED", "0")
    res = json.loads(plugin._self_review_score_handler({"gold_findings": [], "predicted_findings": []}))
    assert "error" in res and "disabled" in res["error"]


def test_self_review_score_rejects_non_list(plugin):
    res = json.loads(plugin._self_review_score_handler({"gold_findings": "x", "predicted_findings": []}))
    assert "error" in res and "must be lists" in res["error"]
    res = json.loads(plugin._self_review_score_handler({"gold_findings": [], "predicted_findings": 42}))
    assert "error" in res and "must be lists" in res["error"]


def test_self_review_score_rejects_non_dict_items(plugin):
    # LLM can emit garbage items inside the lists; must not crash (AttributeError)
    res = json.loads(plugin._self_review_score_handler({
        "gold_findings": ["not a dict"],
        "predicted_findings": [],
    }))
    assert "error" in res and "must be a dict" in res["error"]
    res = json.loads(plugin._self_review_score_handler({
        "gold_findings": [{"severity": "blocker", "evidence": "x"}],
        "predicted_findings": [None],
    }))
    assert "error" in res and "must be a dict" in res["error"]


# --- pre_llm_call suffix hook -----------------------------------------------

def test_suffix_off_by_default_returns_none(plugin):
    assert plugin._pre_llm_call(user_message="build me something") is None


def test_suffix_on_injects_context_dict(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_SUFFIX", "1")
    result = plugin._pre_llm_call(user_message="build me something")
    assert isinstance(result, dict)
    assert "context" in result
    assert "[SELF-REVIEW SPECULATOR MODE]" in result["context"]


def test_suffix_only_on_build_hint(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_SUFFIX", "1")
    # review chatter should NOT trigger injection
    assert plugin._pre_llm_call(user_message="review these findings") is None
    # non-string user_message is ignored
    assert plugin._pre_llm_call(user_message=[]) is None
    assert plugin._pre_llm_call(user_message=None) is None


def test_suffix_respects_master_switch(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_ENABLED", "0")
    monkeypatch.setenv("SELF_REVIEW_SUFFIX", "1")
    assert plugin._pre_llm_call(user_message="build me x") is None


# --- mode isolation helpers -------------------------------------------------

def test_has_mode_reset(plugin):
    assert plugin._has_mode_reset("foo [MODE RESET] bar") is True
    assert plugin._has_mode_reset("foo bar") is False


def test_register_registers_tools_and_hook(plugin):
    ctx = _fake_ctx()
    plugin.register(ctx)
    assert "self_review" in ctx.tools
    assert "self_review_score" in ctx.tools
    assert ctx.tools["self_review"]["toolset"] == "review"
    assert "pre_llm_call" in ctx.hooks
    # schema marks required fields
    assert ctx.tools["self_review_score"]["schema"]["parameters"]["required"] == [
        "gold_findings", "predicted_findings",
    ]


def test_register_disabled_noops(plugin, monkeypatch):
    monkeypatch.setenv("SELF_REVIEW_ENABLED", "0")
    ctx = _fake_ctx()
    plugin.register(ctx)
    assert ctx.tools == {}
    assert ctx.hooks == {}
