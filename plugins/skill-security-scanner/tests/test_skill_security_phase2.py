"""Tests for hermes_cli.skill_input, hermes_cli.yara_pass, hermes_cli.osv_client."""
from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from hermes_cli.osv_client import (
    clear_cache,
    parse_requirements_txt,
    query_packages,
)
from hermes_cli.skill_input import clear_cache as input_clear_cache
from hermes_cli.skill_input import resolve_input
from hermes_cli.yara_pass import _load_rules, scan_yara


# ----------------- skill_input -----------------


def test_resolve_local_dir(tmp_path: Path):
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text("# hi\n")
    out = resolve_input(d)
    assert out.is_dir()
    assert (out / "SKILL.md").exists()


def test_resolve_local_file(tmp_path: Path):
    f = tmp_path / "SKILL.md"
    f.write_text("# hi\n")
    out = resolve_input(f)
    assert out.is_dir()
    assert (out / "SKILL.md").exists()


def test_resolve_local_zip(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "SKILL.md").write_text("# zip-skill\n")
    zp = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(src / "SKILL.md", arcname="SKILL.md")
    out = resolve_input(zp)
    assert (out / "SKILL.md").exists()
    for p in out.rglob("*"):
        assert str(p.resolve()).startswith(str(out.resolve()))


def test_resolve_local_zip_blocks_zipslip(tmp_path: Path):
    zp = tmp_path / "evil.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("../etc/passwd", "root:x:0:0::/root:/bin/bash")
    with pytest.raises(ValueError, match="Zip-slip"):
        resolve_input(zp)


def test_resolve_nonexistent_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_input(tmp_path / "missing")


def test_resolve_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        resolve_input("ftp://example.com/skill.zip")


def test_resolve_git_cached(tmp_path: Path, monkeypatch):
    """Don't actually clone — stub subprocess to verify cache reuse."""
    import subprocess
    from hermes_cli import skill_input
    calls: list[list[str]] = []
    fake_dest = tmp_path / "fake-clone"
    fake_dest.mkdir()
    (fake_dest / "SKILL.md").write_text("# from-git\n")
    real_run = subprocess.run
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return real_run(["true"], capture_output=True)
    # Patch the symbol in skill_input's namespace (not the subprocess module)
    monkeypatch.setattr(skill_input.subprocess, "run", fake_run)
    monkeypatch.setattr(skill_input, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(skill_input, "_cache_dir", lambda t: fake_dest)
    out = resolve_input("https://github.com/example/skill.git")
    assert out == fake_dest
    assert calls, f"Expected git command to run, got calls={calls}"


def test_input_clear_cache(tmp_path: Path, monkeypatch):
    from hermes_cli import skill_input
    cache_root = tmp_path / "cache-root"
    cache_root.mkdir()
    (cache_root / "a").mkdir()
    (cache_root / "b").mkdir()
    before = sum(1 for _ in cache_root.iterdir())
    monkeypatch.setattr(skill_input, "_CACHE_ROOT", cache_root)
    n = input_clear_cache()
    assert n == before
    assert not (cache_root / "a").exists()
    assert not (cache_root / "b").exists()


# ----------------- yara_pass -----------------


def test_yara_rules_compile():
    rules = _load_rules()
    assert rules is not None, "yara-python not installed or no rules found"


def test_yara_detects_miner_in_real_file(tmp_path: Path):
    f = tmp_path / "miner.py"
    f.write_text("import xmrig\nimport sys\n")
    findings = scan_yara(f)
    miner_hits = [x for x in findings if "miner" in x.pattern.lower() or "crypto" in x.category.lower()]
    assert any(x.pattern == "crypto_miner_binary" for x in miner_hits), \
        f"Expected crypto_miner_binary match, got: {[f.pattern for f in findings]}"


def test_yara_detects_webshell(tmp_path: Path):
    f = tmp_path / "shell.php"
    f.write_text("<?php eval($_GET['cmd']); ?>")
    findings = scan_yara(f)
    assert any(x.rule_id == "YR2" for x in findings), f"Expected YR2 (webshell), got: {findings}"


def test_yara_clean_file_no_findings(tmp_path: Path):
    f = tmp_path / "clean.py"
    f.write_text("def hello():\n    return 'world'\n")
    findings = scan_yara(f)
    assert findings == []


def test_yara_mimikatz_detected(tmp_path: Path):
    f = tmp_path / "dropper.py"
    f.write_text("# Stage 2: mimikatz sekurlsa::logonpasswords\n")
    findings = scan_yara(f)
    rule_ids = {x.rule_id for x in findings}
    assert "YR4" in rule_ids or "YR1" in rule_ids


# ----------------- osv_client -----------------


def test_parse_requirements_basic():
    content = textwrap.dedent('''
    # comment
    requests==2.31.0
    flask>=2.0
    --index-url https://example.com
    numpy
    ''').strip()
    pkgs = parse_requirements_txt(content)
    names = [p[0] for p in pkgs]
    assert "requests" in names
    assert "flask" in names
    assert "numpy" in names
    by_name = {p[0]: p[1] for p in pkgs}
    assert by_name["requests"] == "2.31.0"
    assert by_name["flask"] == "2.0"
    assert by_name["numpy"] is None


def test_parse_requirements_empty():
    assert parse_requirements_txt("") == []
    assert parse_requirements_txt("# only comments\n# more\n") == []


def test_query_packages_offline_returns_empty():
    clear_cache()
    with mock.patch("hermes_cli.osv_client._http_client") as hc:
        hc.return_value.__enter__.side_effect = ConnectionError("offline")
        out = query_packages([("requests", "2.31.0")])
    assert out == [[]]


def test_query_packages_with_mock_response():
    clear_cache()
    fake_response = mock.MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"vulns": [{"id": "GHSA-test-1", "summary": "test vuln",
                        "database_specific": {"severity": "HIGH"}}]}
        ]
    }
    fake_response.raise_for_status.return_value = None

    fake_vuln_response = mock.MagicMock()
    fake_vuln_response.json.return_value = {
        "id": "GHSA-test-1",
        "summary": "test vuln",
        "database_specific": {"severity": "HIGH"},
        "aliases": ["CVE-2024-9999"],
    }
    fake_vuln_response.raise_for_status.return_value = None

    cm = mock.MagicMock()
    cm.get.return_value = fake_vuln_response
    cm.post.return_value = fake_response
    cm.__enter__.return_value = cm
    cm.__exit__.return_value = False

    with mock.patch("hermes_cli.osv_client._http_client", return_value=cm):
        out = query_packages([("requests", "2.31.0")])
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].vuln_id == "GHSA-test-1"
    assert out[0][0].severity == "HIGH"


def test_query_packages_empty_input():
    assert query_packages([]) == []


def test_clear_cache_idempotent():
    clear_cache()
    clear_cache()


# ----------------- integration: scan_skill with Phase 2 passes -----------------


def test_scan_skill_includes_yara_findings(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# safe\n")
    (tmp_path / "miner.py").write_text("import xmrig\n")
    from hermes_cli.skill_security import scan_skill
    r = scan_skill(tmp_path)
    rule_ids = {f.rule_id for f in r.findings}
    assert "YR3" in rule_ids or any("miner" in f.pattern.lower() for f in r.findings)


def test_scan_skill_completes_when_osv_offline(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# safe\n")
    (tmp_path / "requirements.txt").write_text("requests==2.31.0\nflask>=2.0\n")
    from hermes_cli.skill_security import scan_skill
    r = scan_skill(tmp_path)
    assert r.verdict.value in ("allow", "warn", "block")
