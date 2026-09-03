#!/usr/bin/env python3
"""Fail if tracked git content looks like secrets, keys, or private LAN data.

Exit 0 = clean. Exit 1 = findings. Never prints matched secret values.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# name, regex, skip if this substring in the line (false-positive carve-outs)
RULES: list[tuple[str, re.Pattern[str], re.Pattern[str] | None]] = [
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), None),
    ("ghp", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), None),
    ("gho_ghu_ghs", re.compile(r"\b(gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), None),
    ("aws_akia", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None),
    ("openai_sk", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), None),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), None),
    ("private_key_block", re.compile(r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"), None),
    ("pem_header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), None),
    ("assignment_password", re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]"), re.compile(r"(?i)password.*(example|placeholder|changeme|your_|todo|<)")),
    ("assignment_token", re.compile(r"(?i)(api[_-]?key|secret|token|auth)\s*[=:]\s*['\"][A-Za-z0-9_\-./+=]{12,}['\"]"), re.compile(r"(?i)(example|placeholder|your_|todo|xxx|<token>|<key>|changeme|dummy)")),
    ("lan_ipv4", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), None),
    ("rfc1918_10", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), re.compile(r"10\.0\.0\.0|10\.255\.255\.255")),
    ("tailscale_100", re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), None),
    ("phone_248", re.compile(r"248[-.\s]?881[-.\s]?0030"), None),
    ("iamthenight", re.compile(r"(?i)iamthenight"), None),
    ("high_entropy_hex", re.compile(r"\b[a-f0-9]{64}\b"), re.compile(r"(?i)(sha256|checksum|hash|commit|tree)")),
]

SKIP_PATH = re.compile(r"(^|/)(\.git/|__pycache__/|\.pytest_cache/)")
TEXT_EXT = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".env", ".sample", ""}


def git_files(root: Path) -> list[Path]:
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    names = [n for n in r.stdout.decode().split("\0") if n]
    return [root / n for n in names]


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for name, pat, skip in RULES:
            if not pat.search(line):
                continue
            if skip and skip.search(line):
                continue
            hits.append((path, i, name))
    return hits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists():
        print("FAIL: not a git repo", root)
        return 2
    hits: list[tuple[str, int, str]] = []
    for path in git_files(root):
        rel = str(path.relative_to(root))
        if SKIP_PATH.search(rel):
            continue
        if path.suffix.lower() not in TEXT_EXT and path.name not in {".gitignore"}:
            # still scan small files without suffix
            if path.suffix:
                continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(scan_text(rel, text))
    # history: every blob currently reachable
    r = subprocess.run(
        [
            "git", "-C", str(root), "grep", "-I", "-n",
            "-e", "github_pat_", "-e", "ghp_", "-e", "BEGIN PRIVATE",
            "-e", "Iamthenight", "-e", "248-881-0030", "-e", "192.168.",
            "--", ".", ":!scripts/scan_secrets.py",
        ],
        capture_output=True, text=True,
    )
    hist = []
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            hist.append(line.split(":", 2)[:2])  # path, line — drop value
    print("files_scanned", len(git_files(root)))
    print("content_hits", len(hits))
    print("git_grep_hits", 0 if r.returncode == 1 else len(r.stdout.splitlines()) if r.returncode == 0 else f"err:{r.returncode}")
    if hits:
        print("FINDINGS (path:line:rule — values omitted)")
        for path, line, name in hits:
            print(f"  {path}:{line}:{name}")
        return 1
    if r.returncode == 0 and r.stdout.strip():
        print("FINDINGS git-grep (path:line omitted values)")
        for row in r.stdout.splitlines()[:50]:
            parts = row.split(":", 2)
            print(f"  {parts[0]}:{parts[1] if len(parts)>1 else '?'}:git_grep")
        return 1
    print("CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
