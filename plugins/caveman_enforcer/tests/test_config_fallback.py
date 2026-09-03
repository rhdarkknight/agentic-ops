"""Regression coverage for isolated-home caveman config fallback."""

import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plugin  # type: ignore  # noqa: E402


def test_empty_state_falls_through_to_isolated_home_config(tmp_path, monkeypatch):
    """Config fallback must not require PyYAML or consult the real HOME."""
    state_file = tmp_path / "state.json"
    state_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(plugin, "STATE_FILE", state_file)

    fake_home = tmp_path / "home"
    config_path = fake_home / ".hermes" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("display:\n  caveman_mode: ultra\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    original_import = builtins.__import__

    def without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_yaml)

    assert plugin.get_caveman_mode() == "ultra"
