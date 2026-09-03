"""
YARA-based malware/hacktool scan pass.

Loads .yar rules from hermes_cli/yara_rules/ and scans each file
in a skill tree. YARA matches become Finding objects tagged with
rule_id YR1..YR4 and severity derived from rule meta.

If yara-python is not installed, scan_yara() returns an empty list
and logs a warning. The rest of the security gate still works.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .skill_security import Finding, Location, Severity

if TYPE_CHECKING:
    import yara as _yara_typing  # noqa: F401
else:
    _yara_typing = None


class _YaraMatch(Protocol):
    rule: str
    meta: dict[str, Any]
    strings: list[Any]
    tags: list[str]


class _YaraRules(Protocol):
    def match(self, filepath: str) -> list[_YaraMatch]: ...


logger = logging.getLogger(__name__)

_RULES_DIR = Path(__file__).parent / "yara_rules"

_CATEGORY_DEFAULTS: dict[str, tuple[Severity, str]] = {
    "cryptominer": (Severity.HIGH, "YR3"),
    "webshell": (Severity.HIGH, "YR2"),
    "malware": (Severity.CRITICAL, "YR1"),
    "hacktool": (Severity.HIGH, "YR4"),
}


@lru_cache(maxsize=1)
def _load_rules() -> _YaraRules | None:
    try:
        import yara  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("yara-python not installed; skipping YARA pass. pip install yara-python")
        return None
    if not _RULES_DIR.is_dir():
        return None
    sources: dict[str, str] = {}
    for p in _RULES_DIR.glob("*.yar"):
        sources[p.stem] = str(p)
    if not sources:
        return None
    try:
        return yara.compile(filepaths=sources)  # type: ignore[return-value]
    except Exception as e:
        logger.error("YARA compile failed: %s", e)
        return None


def _severity_for_rule(rule: _YaraMatch) -> Severity:
    sev = (rule.meta.get("severity", "") or "").upper()
    if sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return Severity(sev)
    cat = rule.meta.get("category", "")
    if cat in _CATEGORY_DEFAULTS:
        return _CATEGORY_DEFAULTS[cat][0]
    return Severity.HIGH


def _rule_id_for_rule(rule: _YaraMatch) -> str:
    rid = rule.meta.get("rule_id", "") or ""
    if rid:
        return rid
    cat = rule.meta.get("category", "")
    if cat in _CATEGORY_DEFAULTS:
        return _CATEGORY_DEFAULTS[cat][1]
    return "YR1"


def scan_yara(skill_path: Path) -> list[Finding]:
    rules = _load_rules()
    if rules is None:
        return []
    findings: list[Finding] = []
    files: list[Path] = []
    if skill_path.is_file():
        files = [skill_path]
    else:
        for p in skill_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".py", ".sh", ".js", ".ts", ".md", ".txt",
                                        ".php", ".jsp", ".aspx", ".rb", ".go", ".rs"}:
                continue
            try:
                if p.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            files.append(p)
    for fp in files:
        try:
            matches = rules.match(str(fp))
        except Exception as e:
            logger.debug("YARA match failed on %s: %s", fp, e)
            continue
        for m in matches:
            sev = _severity_for_rule(m)
            rid = _rule_id_for_rule(m)
            findings.append(Finding(
                rule_id=rid,
                category="YARA Match",
                pattern=m.rule,
                message=f"YARA rule {m.rule} matched {fp.name}",
                severity=sev,
                confidence=0.9,
                location=Location(file=str(fp), start_line=1),
                explanation=f"YARA signature {m.rule} matched in file",
                remediation="Inspect and remove the matching content; this is a known-malicious signature",
                code_snippet="",
                matched_text=", ".join(str(s) for s in m.strings[:3])[:200],
            ))
    return findings