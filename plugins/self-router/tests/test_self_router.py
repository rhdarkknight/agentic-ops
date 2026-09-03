"""Tests for the self-router plugin (WS4).

Cover: self_assess classification/scoring, router decision logic (keep_self /
recommend / dispatch gating), anchoring (novelty flag + world-model register,
counterfactual no-rate, clean-slate divergence).

Run from inside the plugin dir:
    cd ~/.hermes/plugins/self-router && python -m pytest tests/ -v --rootdir=tests/
"""

import os
import sys
import json
from pathlib import Path

# Make the plugin package importable under a stable name for tests.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_PLUGIN_DIR)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from self_router import self_assess, anchoring, router, config  # noqa: E402


# ---------------------------------------------------------------------------
# self_assess
# ---------------------------------------------------------------------------

def test_classify_coding_task():
    assert self_assess.classify_task("implement the refund function") == "narrow_coding"


def test_classify_security_task():
    assert self_assess.classify_task("audit this for a security vulnerability") == "security_review"


def test_classify_ops_research():
    assert self_assess.classify_task("deploy the new backup service") == "ops_research"


def test_classify_unknown_defaults():
    assert self_assess.classify_task("what is the weather") == "default"


def test_assess_returns_expected_keys():
    a = self_assess.assess("refactor the auth module")
    for key in ("task", "task_type", "keep_self", "specialist_fit",
                "best_specialist", "specialist_name"):
        assert key in a


def test_keep_self_beats_specialist_on_ops():
    # Ops/infrastructure is where Hermes wins.
    route, spec, a = self_assess.should_route_to_specialist(
        "monitor the fleet and report incidents", win_margin=0.15
    )
    assert route is False


def test_specialist_beats_self_on_narrow_sandbox_coding():
    # Narrow single-repo coding + sandbox -> specialist wins.
    route, spec, a = self_assess.should_route_to_specialist(
        "fix the bug in the function and run it in a sandbox", win_margin=0.15
    )
    assert route is True
    assert spec in ("codex", "claude_code")


# ---------------------------------------------------------------------------
# anchoring
# ---------------------------------------------------------------------------

def test_detect_novelty_low_match(tmp_path):
    cfg = config.load_config()
    thresh = cfg["novelty_threshold"]
    # Force fallback store to tmp so we don't touch the real world model.
    store = anchoring.AnchoringStore(agent_id="test", store_dir=str(tmp_path))
    res = anchoring.detect_novelty(
        closest_match=0.1, novelty_threshold=thresh,
        task_signature="sig", description="novel task", store=store,
    )
    assert res["no_prior_art"] is True
    # Register should have produced a path (canonical or fallback).
    assert res.get("register_path")


def test_detect_novelty_high_match_no_flag(tmp_path):
    store = anchoring.AnchoringStore(agent_id="test", store_dir=str(tmp_path))
    res = anchoring.detect_novelty(
        closest_match=0.95, novelty_threshold=0.6,
        task_signature="sig", description="known task", store=store,
    )
    assert res["no_prior_art"] is False


def test_anchoring_store_falls_back_to_jsonl(tmp_path, monkeypatch):
    # Force the world-model load to fail so the JSONL fallback must kick in.
    from self_router import anchoring as anch
    store = anch.AnchoringStore(agent_id="test", store_dir=str(tmp_path))
    monkeypatch.setattr(store, "_load_world_model", lambda: None)
    path = store.register_no_prior_art("sig", 0.1, "desc")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        entry = fh.read()
    assert "no_prior_art" in entry


def test_anchoring_store_writes_typed_entry_to_world_model(tmp_path, monkeypatch):
    # G3: the canonical path must write a TYPED no_prior_art entry with
    # structured fields (task_signature, closest_match, timestamp), not a
    # free-text landmark. Verify via the real WorldModel archive API.
    sys.path.insert(0, os.path.expanduser("~/.hermes/skills/agent-world-model/scripts"))
    from world_model import WorldModel  # noqa: E402

    from self_router import anchoring as anch
    store = anch.AnchoringStore(agent_id=f"test_{os.getpid()}", store_dir=str(tmp_path))
    # Point the world model at the real archive dir so append() lands there.
    store.store_dir = str(tmp_path)
    store._load_world_model = lambda: WorldModel(
        agent_id=store.agent_id, root=__import__("pathlib").Path(store.store_dir)
    )
    store.register_no_prior_art("sig-x", 0.2, "novel desc")
    archive = tmp_path / f"{store.agent_id}_archive.jsonl"
    assert archive.exists()
    lines = [json.loads(line) for line in archive.read_text().splitlines() if line.strip()]
    typed = [e for e in lines if e.get("entry_type") == "no_prior_art"]
    assert typed, "expected a TYPED no_prior_art entry in the world archive"
    data = typed[-1].get("data", {})
    assert data.get("task_signature") == "sig-x"
    assert data.get("closest_match") == 0.2
    assert data.get("timestamp") is not None


def test_task_signature_normalizes_argument_order():
    first = anchoring.build_task_signature("patch", {"path": "x", "old": "a"})
    second = anchoring.build_task_signature("patch", {"old": "a", "path": "x"})
    assert first == second
    assert first != anchoring.build_task_signature("patch", {"path": "y", "old": "a"})


def test_anchoring_store_deduplicates_signature_within_seven_days(tmp_path, monkeypatch):
    store = anchoring.AnchoringStore(agent_id="dedupe", store_dir=str(tmp_path))
    monkeypatch.setattr(store, "_load_world_model", lambda: None)
    ts = 2_000_000_000.0
    signature = anchoring.build_task_signature("terminal", {"command": "true"})
    first = store.register_no_prior_art(signature, 0.1, "desc", timestamp=ts)
    second = store.register_no_prior_art(signature, 0.1, "desc", timestamp=ts + 60)
    assert first == second
    with open(first, encoding="utf-8") as fh:
        assert len([line for line in fh if line.strip()]) == 1


def test_anchoring_store_allows_same_signature_after_window(tmp_path, monkeypatch):
    store = anchoring.AnchoringStore(agent_id="expiry", store_dir=str(tmp_path))
    monkeypatch.setattr(store, "_load_world_model", lambda: None)
    ts = 2_000_000_000.0
    signature = anchoring.build_task_signature("terminal", {"command": "true"})
    path = store.register_no_prior_art(signature, 0.1, "desc", timestamp=ts)
    store.register_no_prior_art(signature, 0.1, "desc", timestamp=ts + anchoring.NOVELTY_DEDUPE_WINDOW_SEC + 1)
    with open(path, encoding="utf-8") as fh:
        assert len([line for line in fh if line.strip()]) == 2


def test_closest_match_uses_stored_signatures(tmp_path, monkeypatch):
    store = anchoring.AnchoringStore(agent_id="match", store_dir=str(tmp_path))
    monkeypatch.setattr(store, "_load_world_model", lambda: None)
    signature = anchoring.build_task_signature("terminal", {"command": "true"})
    store.register_no_prior_art(signature, 0.1, "desc")
    assert store.closest_match(signature) == 1.0


def test_closest_match_ignores_non_object_json(tmp_path):
    path = tmp_path / "no_prior_art.jsonl"
    path.write_text("[]\n{\"task_signature\": \"terminal:{}\"}\n", encoding="utf-8")
    store = anchoring.AnchoringStore(agent_id="malformed", store_dir=str(tmp_path))
    assert store.closest_match("terminal:{}") == 1.0


def test_anchoring_store_sanitizes_agent_id_path(tmp_path, monkeypatch):
    store = anchoring.AnchoringStore(agent_id="../unsafe/id", store_dir=str(tmp_path))
    monkeypatch.setattr(store, "_load_world_model", lambda: None)
    path = store.register_no_prior_art("sig", 0.1, "desc")
    assert path.startswith(str(tmp_path))
    assert Path(path).parent == tmp_path


def test_counterfactual_no_rate_healthy():
    # 50% change = healthy discovery (>= 30%).
    res = anchoring.compute_counterfactual_no_rate(
        [("d1", True), ("d2", False), ("d3", True), ("d4", True)]
    )
    assert res["rate"] == 0.75
    assert res["anchored"] is False


def test_counterfactual_no_rate_anchored():
    # 10% change = anchoring (< 30%).
    res = anchoring.compute_counterfactual_no_rate(
        [("d1", False), ("d2", False), ("d3", False), ("d4", True)]
    )
    assert res["rate"] == 0.25
    assert res["anchored"] is True


def test_clean_slate_divergence_healthy():
    res = anchoring.clean_slate_divergence("do A", "do B")
    assert res["divergent"] is True
    assert res["anchored"] is False


def test_clean_slate_divergence_anchored():
    res = anchoring.clean_slate_divergence("do A", "do A")
    assert res["divergent"] is False
    assert res["anchored"] is True


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

def test_router_disabled_keeps_self(tmp_path):
    r = router.Router(config=config.load_config({"enabled": False}))
    d = r.maybe_route("refactor the auth module")
    assert d["decision"] == "keep_self"


def test_router_recommends_when_executor_unverified(tmp_path):
    # auto_dispatch on but target NOT in verified executors -> must recommend,
    # never silently dispatch.
    r = router.Router(config=config.load_config({
        "auto_dispatch": True,
        "executors": [],  # nothing verified
        "route_novel_to": "codex",
        "novelty_threshold": 0.6,
        "anchoring_risk_threshold": 0.7,
    }))
    d = r.maybe_route(
        task="fix the bug in the function and run it in a sandbox",
        closest_match=0.1,  # novel
        confidence=0.9,     # high anchoring risk
    )
    assert d["decision"] == "recommend"
    assert d.get("verified_executor") is False


def test_router_dispatch_only_when_verified_and_auto(tmp_path):
    r = router.Router(config=config.load_config({
        "auto_dispatch": True,
        "executors": ["codex"],
        "route_novel_to": "codex",
        "novelty_threshold": 0.6,
        "anchoring_risk_threshold": 0.7,
    }))
    d = r.maybe_route(
        task="fix the bug in the function and run it in a sandbox",
        closest_match=0.1,
        confidence=0.9,
    )
    # Novel + high anchoring risk -> route to stateless harness (codex).
    assert d["decision"] == "dispatch"
    assert d["harness"] == "codex"
    assert d.get("verified_executor") is True


def test_router_keep_self_on_ops_when_no_specialist_win(tmp_path):
    r = router.Router(config=config.load_config({
        "auto_dispatch": True,
        "executors": ["claude_code", "codex", "opencode"],
    }))
    d = r.maybe_route("monitor the fleet and report incidents", closest_match=0.9)
    assert d["decision"] == "keep_self"


def test_should_check_readonly_never_triggers(tmp_path):
    r = router.Router(config=config.load_config())
    assert r.should_check("read_file") is False
    assert r.should_check("web_search") is False


def test_should_check_mutating_outside_cooldown(tmp_path):
    r = router.Router(config=config.load_config())
    # First call: outside cooldown -> True.
    assert r.should_check("write_file") is True
    # Immediately after: inside cooldown -> False.
    assert r.should_check("write_file") is False


def test_load_config_reads_config_yaml(tmp_path, monkeypatch):
    # The config.yaml self_router block must be read (executors/auto_dispatch
    # are the runtime source of truth). Regression for the "config block dead"
    # bug: load_config previously never read config.yaml, so executors stayed [].
    cfg_dir = tmp_path / ".hermes"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "self_router:\n  executors: [kimi]\n  auto_dispatch: true\n"
        "  route_novel_to: kimi\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(cfg_dir))
    cfg = config.load_config()
    assert cfg["executors"] == ["kimi"]
    assert cfg["auto_dispatch"] is True
    assert cfg["route_novel_to"] == "kimi"


# ---------------------------------------------------------------------------
# __init__ registration + hook
# ---------------------------------------------------------------------------

def test_plugin_has_register():
    from self_router import register as _reg
    assert callable(_reg)


class _FakeCtx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def test_register_wires_pre_tool_call_hook():
    from self_router.__init__ import register
    ctx = _FakeCtx()
    register(ctx)
    assert "pre_tool_call" in ctx.hooks


def test_hook_returns_none_for_readonly_tool():
    from self_router.__init__ import _pre_tool_call
    res = _pre_tool_call(tool_name="read_file", args={"path": "/tmp/x"})
    assert res is None


# ---------------------------------------------------------------------------
# model cascade
# ---------------------------------------------------------------------------

def test_cascade_map_known_provider_model():
    from self_router import cascade
    assert cascade.map_to_litellm("streamlake", "kat-coder-pro-v2.5") == "streamlake.kat-coder-pro-v2.5"
    assert cascade.map_to_litellm("alibaba-model-studio", "deepseek-v4-flash-0731") == "alibaba/deepseek-v4-flash-0731"


def test_cascade_map_suffix_fallback():
    # New provider, known model name -> suffix match still cascades.
    from self_router import cascade
    assert cascade.map_to_litellm("some-new-provider", "deepseek-v4-flash-0731") == "alibaba/deepseek-v4-flash-0731"


def test_cascade_map_unknown_returns_none():
    from self_router import cascade
    assert cascade.map_to_litellm("unknown", "totally-unknown-model") is None


def test_cascade_writes_config_toml(tmp_path):
    from self_router import cascade
    toml = tmp_path / "config.toml"
    toml.write_text(
        'default_model = "litellm-kimi"\n'
        '\n'
        '[models.litellm-kimi]\n'
        'capabilities = ["tool_use"]\n'
        'model = "abacus/moonshotai/Kimi-K3"\n'
        'provider = "litellm"\n'
        '\n'
        '[providers.litellm]\n'
        'api_key = "x"\n'
        'base_url = "http://x"\n'
        'type = "openai"\n',
        encoding="utf-8",
    )
    path = cascade.write_cascaded_model("alibaba/deepseek-v4-flash-0731", config_toml=toml)
    assert path == toml
    content = toml.read_text(encoding="utf-8")
    assert 'model = "alibaba/deepseek-v4-flash-0731"' in content
    # Other lines preserved.
    assert 'api_key = "x"' in content
    assert 'default_model = "litellm-kimi"' in content


def test_cascade_full_pipeline(tmp_path, monkeypatch):
    # active streamlake -> litellm streamlake model -> written to kimi toml.
    from self_router import cascade
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[models.litellm-kimi]\nmodel = "abacus/moonshotai/Kimi-K3"\nprovider = "litellm"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cascade, "KIMI_CONFIG_TOML", toml)
    res = cascade.cascade(active={"provider": "streamlake", "model": "kat-coder-pro-v2.5"})
    assert res["cascaded"] is True
    assert res["litellm"] == "streamlake.kat-coder-pro-v2.5"
    assert 'model = "streamlake.kat-coder-pro-v2.5"' in toml.read_text(encoding="utf-8")


def test_cascade_no_mapping_no_write(tmp_path, monkeypatch):
    from self_router import cascade
    toml = tmp_path / "config.toml"
    toml.write_text('[models.litellm-kimi]\nmodel = "x"\n', encoding="utf-8")
    monkeypatch.setattr(cascade, "KIMI_CONFIG_TOML", toml)
    res = cascade.cascade(active={"provider": "unknown", "model": "unknown-model"})
    assert res["cascaded"] is False
    # File untouched (still original).
    assert 'model = "x"' in toml.read_text(encoding="utf-8")


def test_cascade_prefers_active_session_model(tmp_path, monkeypatch):
    # The ACTIVE session model (deepseek) must win over the config default
    # (kat-coder). This is the cost-separation guarantee.
    from self_router import cascade
    toml = tmp_path / "config.toml"
    toml.write_text('[models.litellm-kimi]\nmodel = "x"\n', encoding="utf-8")
    monkeypatch.setattr(cascade, "KIMI_CONFIG_TOML", toml)
    # config default = kat-coder, but active session = deepseek-v4-flash.
    res = cascade.cascade(
        active={"provider": "streamlake", "model": "kat-coder-pro-v2.5"},
        active_model="deepseek-v4-flash-0731",
    )
    assert res["cascaded"] is True
    assert res["litellm"] == "alibaba/deepseek-v4-flash-0731"
    assert 'model = "alibaba/deepseek-v4-flash-0731"' in toml.read_text(encoding="utf-8")


def test_session_model_capture_and_lookup():
    import sys
    # The package itself IS the __init__ module; import it via the package name.
    init = sys.modules["self_router"]
    init._SESSION_MODELS.clear()
    init._record_session_model(session_id="sess-1", model="deepseek-v4-flash-0731")
    assert init.get_active_session_model("sess-1") == "deepseek-v4-flash-0731"
    assert init.get_active_session_model("unknown") == ""
    init._SESSION_MODELS.clear()


def test_pre_llm_call_refreshes_session_model():
    # Mid-session /model switch: pre_llm_call must refresh the capture so a live
    # switch cascades to the executor immediately (not just at session start).
    import sys
    init = sys.modules["self_router"]
    init._SESSION_MODELS.clear()
    init._record_session_model(session_id="sess-2", model="old-model")
    # Simulate a /model switch mid-session -> pre_llm_call fires with new model.
    res = init._refresh_session_model(session_id="sess-2", model="deepseek-v4-flash-0731")
    assert res is None  # observer, no context injection
    assert init.get_active_session_model("sess-2") == "deepseek-v4-flash-0731"
    init._SESSION_MODELS.clear()