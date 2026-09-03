"""
SARIF 2.1.0 output for skill scan results.

SARIF is a JSON schema used by GitHub Code Scanning, IDEs, and CI tools.
Each Finding becomes a result; the run is the scan that produced them.

Lifted from NVIDIA/SkillSpector (Apache 2.0) — same format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .skill_security import ScanReport


SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
SARIF_VERSION = "2.1.0"


_SEVERITY_TO_SARIF_LEVEL = {
    "CRITICAL": "error",
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "note",
}


def report_to_sarif(report: "ScanReport", tool_name: str = "hermes-skill-scan") -> dict:
    """Convert a ScanReport to a SARIF 2.1.0 dict."""
    results: list[dict] = []
    rules_index: dict[str, int] = {}
    rules_list: list[dict] = []
    for f in report.findings:
        if f.rule_id not in rules_index:
            rules_index[f.rule_id] = len(rules_list)
            rules_list.append({
                "id": f.rule_id,
                "name": f.pattern,
                "shortDescription": {"text": f.pattern},
                "fullDescription": {"text": f.explanation or f.message},
                "helpUri": "https://github.com/NVIDIA/SkillSpector",
                "defaultConfiguration": {
                    "level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity.value, "warning"),
                },
            })
        rid_idx = rules_index[f.rule_id]
        results.append({
            "ruleId": f.rule_id,
            "ruleIndex": rid_idx,
            "level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity.value, "warning"),
            "message": {"text": f.message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": f.location.file},
                    "region": {
                        "startLine": f.location.start_line,
                        "endLine": f.location.end_line or f.location.start_line,
                    },
                },
            }],
            "properties": {
                "confidence": f.confidence,
                "category": f.category,
                "remediation": f.remediation,
                "explanation": f.explanation,
                "matchedText": f.matched_text,
            },
        })
    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [{
            "tool": {
                "driver": {
                    "name": tool_name,
                    "informationUri": "https://github.com/NVIDIA/SkillSpector",
                    "rules": rules_list,
                }
            },
            "results": results,
            "properties": {
                "skillPath": report.skill_path,
                "riskScore": report.risk_score,
                "verdict": report.verdict.value,
                "severityCounts": report.by_severity(),
            },
        }],
    }


def write_sarif(report: "ScanReport", output_path: str | Path,
                tool_name: str = "hermes-skill-scan") -> Path:
    """Write SARIF output to file. Returns the written path."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_to_sarif(report, tool_name), indent=2))
    return out


def merge_to_sarif(reports: list, output_path: str | Path,
                   tool_name: str = "hermes-skill-scan") -> Path:
    """Merge multiple ScanReports into a single SARIF run."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged_rules: list[dict] = []
    rules_index: dict[str, int] = {}
    merged_results: list[dict] = []
    max_score = 0
    severities: dict[str, int] = {s: 0 for s in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
    has_block = False
    for r in reports:
        max_score = max(max_score, r.risk_score)
        for k, v in r.by_severity().items():
            severities[k] = severities.get(k, 0) + v
        if r.verdict.value == "block":
            has_block = True
        for f in r.findings:
            if f.rule_id not in rules_index:
                rules_index[f.rule_id] = len(merged_rules)
                merged_rules.append({
                    "id": f.rule_id,
                    "name": f.pattern,
                    "shortDescription": {"text": f.pattern},
                    "fullDescription": {"text": f.explanation or f.message},
                    "defaultConfiguration": {
                        "level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity.value, "warning"),
                    },
                })
            merged_results.append({
                "ruleId": f.rule_id,
                "ruleIndex": rules_index[f.rule_id],
                "level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity.value, "warning"),
                "message": {"text": f"[{r.skill_path}] {f.message}"},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.location.file},
                        "region": {
                            "startLine": f.location.start_line,
                            "endLine": f.location.end_line or f.location.start_line,
                        },
                    },
                }],
                "properties": {
                    "skillPath": r.skill_path,
                    "confidence": f.confidence,
                    "category": f.category,
                    "remediation": f.remediation,
                    "explanation": f.explanation,
                    "matchedText": f.matched_text,
                },
            })
    sarif = {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [{
            "tool": {
                "driver": {
                    "name": tool_name,
                    "informationUri": "https://github.com/NVIDIA/SkillSpector",
                    "rules": merged_rules,
                }
            },
            "results": merged_results,
            "properties": {
                "skillCount": len(reports),
                "maxRiskScore": max_score,
                "anyBlock": has_block,
                "severityCounts": severities,
            },
        }],
    }
    out.write_text(json.dumps(sarif, indent=2))
    return out
