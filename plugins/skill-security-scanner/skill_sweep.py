#!/usr/bin/env python3
"""
Skill security sweep — run by cron.

Walks every skill directory under ~/.hermes/skills/, runs the full
security scan, writes SARIF output, and sends a Telegram digest
when CRITICAL or HIGH findings are detected.

Exit codes:
    0 — all skills clean (allow) or warn-only
    1 — at least one skill produced a BLOCK verdict
    2 — script error (import failure, etc.)

Usage:
    python -m hermes_cli.skill_sweep [--skills-dir DIR] [--output PATH]
                                     [--no-alert] [--quiet]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path.home() / ".hermes" / "skills"
DEFAULT_OUTPUT_DIR = Path.home() / ".hermes" / "logs"


def _iter_skill_dirs(skills_dir: Path) -> list[Path]:
    """Top-level entries in skills_dir — each is treated as a skill."""
    if not skills_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(skills_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith(".") and p.name not in {".curator_backups"}:
            continue
        # Heuristic: must contain SKILL.md (anywhere) or top-level README.md
        if (p / "SKILL.md").exists() or (p / "README.md").exists() or any(p.rglob("SKILL.md")):
            out.append(p)
    return out


def _telegram_send(message: str) -> bool:
    """Best-effort Telegram delivery via send_message tool.

    Returns True on success. Never raises.
    """
    try:
        from hermes_tools import send_message  # type: ignore[import-not-found]
        send_message(action="send", target="telegram", message=message)
        return True
    except Exception as e:
        logger.warning("Telegram alert failed: %s", e)
        return False


def _format_digest(reports: list, total_skills: int) -> str:
    """Format a digest of CRITICAL/HIGH findings for Telegram."""
    blocked = [r for r in reports if r.verdict.value == "block"]
    if not blocked:
        return ""
    lines = [
        f"🛑 Skill security sweep: {len(blocked)}/{total_skills} skills BLOCKED",
        "",
    ]
    for r in blocked[:10]:
        sev_counts = r.by_severity()
        top = sorted(r.findings, key=lambda f: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[f.severity.value], -f.confidence))[:3]
        lines.append(
            f"📁 {Path(r.skill_path).name} (risk={r.risk_score}/100, "
            f"CRIT={sev_counts['CRITICAL']} HIGH={sev_counts['HIGH']})"
        )
        for f in top:
            lines.append(f"   [{f.severity.value}] {f.rule_id} {f.pattern}")
        lines.append("")
    if len(blocked) > 10:
        lines.append(f"   ... and {len(blocked) - 10} more")
    return "\n".join(lines)


def run_sweep(skills_dir: Path = DEFAULT_SKILLS_DIR,
              output_dir: Path = DEFAULT_OUTPUT_DIR,
              *,
              alert: bool = True,
              quiet: bool = False) -> int:
    """Run the full sweep. Returns exit code (0=ok, 1=block, 2=error)."""
    try:
        from .sarif_output import merge_to_sarif
        from .skill_security import scan_skill
    except ImportError as e:
        logger.error("Import failed: %s", e)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sarif_path = output_dir / f"skill-scan-{timestamp}.sarif"
    latest_sarif = output_dir / "skill-scan-latest.sarif"

    skills = _iter_skill_dirs(skills_dir)
    if not quiet:
        logger.info("Scanning %d skill(s) under %s", len(skills), skills_dir)

    reports: list = []
    errors: list[tuple[Path, str]] = []
    for skill in skills:
        try:
            r = scan_skill(skill)
            reports.append(r)
        except Exception as e:
            logger.warning("Scan failed for %s: %s", skill, e)
            errors.append((skill, str(e)))

    if not reports and not errors:
        if not quiet:
            logger.info("No skills found under %s", skills_dir)
        return 0

    try:
        merge_to_sarif(reports, sarif_path)
        latest_sarif.write_text(sarif_path.read_text())
    except Exception as e:
        logger.error("SARIF write failed: %s", e)
        return 2

    if errors and not quiet:
        for s, e in errors:
            logger.warning("  ERR %s: %s", s.name, e)

    blocked = [r for r in reports if r.verdict.value == "block"]
    if alert and blocked:
        digest = _format_digest(reports, total_skills=len(skills))
        if digest:
            _telegram_send(digest)

    if not quiet:
        sev_total: dict[str, int] = {s: 0 for s in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
        for r in reports:
            for k, v in r.by_severity().items():
                sev_total[k] = sev_total.get(k, 0) + v
        logger.info(
            "Sweep complete: %d skills scanned, %d blocked, "
            "findings CRIT=%d HIGH=%d MED=%d LOW=%d, SARIF=%s",
            len(skills), len(blocked),
            sev_total["CRITICAL"], sev_total["HIGH"],
            sev_total["MEDIUM"], sev_total["LOW"], sarif_path,
        )
    return 1 if blocked else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes skill security sweep")
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR,
                        help="Root directory of installed skills")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for SARIF reports")
    parser.add_argument("--no-alert", action="store_true",
                        help="Skip Telegram alerting even on block")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress info logging")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_sweep(
        skills_dir=args.skills_dir,
        output_dir=args.output,
        alert=not args.no_alert,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
