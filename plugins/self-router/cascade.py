"""Model cascade: route the executor's model dynamically from the active session.

The self-router dispatches to a specialist harness (kimi). But the harness's
model must NOT be hardcoded — it must follow whatever provider/model the active
session is using, so personal and professional inference costs never overlap.

Mechanism: the kimi ACP child reads `~/.kimi-code/config.toml` at process
launch for its model. The CopilotACPClient does NOT send `session/set_model`,
so the child's model is whatever config.toml says at spawn. Therefore, before a
dispatch, the router writes the ACTIVE session's model (mapped to a litellm
model id) into config.toml. The child then inherits the correct cost bucket.

Cost-separation guarantee: a personal session (cheap/free provider) dispatches
kimi on the cheap model; a professional session (cost-accounted provider)
dispatches on the professional model. No hardcoded model, no cross-billing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KIMI_CONFIG_TOML = Path.home() / ".kimi-code" / "config.toml"

# Default model-alias -> litellm model id mapping. Overridable at runtime via
# config.yaml -> self_router.model_map (the source of truth). This is a
# provider-agnostic fallback: the ACTIVE provider/model from config.yaml is
# mapped through it, so a `/model` swap or new provider cascades without editing
# code. Unknown combos fall back to keeping the existing config.toml model.
DEFAULT_MODEL_MAP: Dict[str, str] = {
    # provider:model -> litellm id
    "streamlake:kat-coder-pro-v2.5": "streamlake.kat-coder-pro-v2.5",
    "streamlake:kat-coder-pro-v2": "streamlake.kat-coder-pro-v2",
    "alibaba:deepseek-v4-flash-0731": "alibaba/deepseek-v4-flash-0731",
    "alibaba-model-studio:deepseek-v4-flash-0731": "alibaba/deepseek-v4-flash-0731",
    "opencode-go:kimi-k2.6": "kimi-k2.6",
    "opencode-go:kimi-k2.7-code": "abacus/moonshotai/Kimi-K2.7-Code",
    "opencode-go:kimi-k2.5": "kimi-k2.6",  # fall back to available kimi
    "ollama:kimi-k2.6": "kimi-k2.6",
    "zai:glm-5.2": "abacus/moonshotai/Kimi-K3",  # coding fallback
    "glm-coding:glm-5.2": "abacus/moonshotai/Kimi-K3",
}


def read_active_model(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Read the active session's model.default / model.provider from config.yaml.

    Returns {"provider": ..., "model": ...} or {} if unreadable.
    """
    cfg_path = config_path or Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "config.yaml"
    try:
        import yaml
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        model_cfg = data.get("model") or {}
        return {
            "provider": str(model_cfg.get("provider") or ""),
            "model": str(model_cfg.get("default") or ""),
        }
    except Exception as exc:
        logger.warning("model cascade: could not read active model: %s", exc)
        return {}


def map_to_litellm(provider: str, model: str, model_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Map (provider, model) to a litellm model id, or None if no mapping.

    Falls back to a suffix match on the model name (e.g. "deepseek-v4-flash-0731"
    matches "alibaba/deepseek-v4-flash-0731") so new providers still cascade.
    """
    mapping = model_map or DEFAULT_MODEL_MAP
    key = f"{provider}:{model}"
    if key in mapping:
        return mapping[key]
    # Exact model-name match across all map values.
    for v in mapping.values():
        if v == model or v.endswith("/" + model):
            return v
    return None


def write_cascaded_model(litellm_model: str, config_toml: Optional[Path] = None) -> Optional[Path]:
    """Write the cascaded litellm model into the kimi config.toml.

    Rewrites only the `[models.litellm-kimi] model = ...` line, preserving the
    rest of the file. Atomic (temp + os.replace) so a crash mid-write can't
    corrupt the user's kimi config. Returns the path written, or None on failure.
    """
    path = config_toml or KIMI_CONFIG_TOML
    import os
    import tempfile
    try:
        if not path.is_file():
            logger.warning("model cascade: %s missing; cannot cascade", path)
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        in_models = False
        rewritten = False
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[models."):
                in_models = stripped.startswith("[models.litellm-kimi]")
                out.append(line)
                continue
            if in_models and stripped.startswith("model "):
                out.append(f'model = "{litellm_model}"')
                rewritten = True
                continue
            out.append(line)
        if not rewritten:
            logger.warning("model cascade: no [models.litellm-kimi] model= line found")
            return None
        new_content = "\n".join(out) + "\n"
        # Atomic write: temp file in the same dir, then os.replace.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config.toml.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        logger.info("model cascade: kimi model -> %s (%s)", litellm_model, path)
        return path
    except Exception as exc:
        logger.warning("model cascade write failed: %s", exc)
        return None


def _model_map_from_config() -> Dict[str, str]:
    """Read self_router.model_map from config.yaml (source of truth). Falls back
    to DEFAULT_MODEL_MAP."""
    try:
        import yaml
        from pathlib import Path as _P
        cfg_path = _P(os.environ.get("HERMES_HOME", str(_P.home() / ".hermes"))) / "config.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mm = (data.get("self_router") or {}).get("model_map") or {}
        return mm if isinstance(mm, dict) else {}
    except Exception:
        return {}


def cascade(active: Optional[Dict[str, str]] = None, model_map: Optional[Dict[str, str]] = None, active_model: str = "") -> Optional[Dict[str, Any]]:
    """Full cascade: read active model -> map to litellm -> write to kimi config.

    ``active_model``: the ACTIVE runtime session model (e.g. "deepseek-v4-flash-0731"),
    highest precedence — this is what the user is actually running, not the config
    default. When absent, falls back to config.yaml model.default. ``model_map``:
    explicit override (highest precedence) -> config.yaml self_router.model_map ->
    DEFAULT_MODEL_MAP. Returns {"provider", "model", "litellm": <id>, "written": path}
    on success, or {"cascaded": False, "reason": ...} when no mapping applies.
    """
    active = active or read_active_model()
    provider = active.get("provider", "")
    model = active.get("model", "") or active_model
    # If an ACTIVE session model was provided, prefer its provider:model key.
    if active_model and active_model != model:
        model = active_model
    if not model:
        return {"cascaded": False, "reason": "no active model"}
    # Resolve model_map: explicit override > config.yaml > default.
    resolved_map = model_map or _model_map_from_config() or DEFAULT_MODEL_MAP
    litellm_model = map_to_litellm(provider, model, resolved_map)
    if not litellm_model:
        return {"cascaded": False, "reason": f"no litellm mapping for {provider}:{model}"}
    written = write_cascaded_model(litellm_model)
    if written is None:
        return {"cascaded": False, "reason": "config.toml write failed"}
    return {"cascaded": True, "provider": provider, "model": model, "litellm": litellm_model, "written": str(written)}