"""
Multi-format input resolver for skill scanning.

Accepts:
    - Local directory (default)
    - Local file
    - Local zip path (.zip)
    - Git URL (https://github.com/...) — shallow clone
    - http(s) URL pointing to a zip or git repo

Returns a resolved Path to a directory ready for scan_skill().
Cached in /tmp/hermes-skill-scan/<hash>/ for repeat scans.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(tempfile.gettempdir()) / "hermes-skill-scan"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _hash_target(target: str) -> str:
    return hashlib.sha256(target.encode()).hexdigest()[:16]


def _is_git_url(s: str) -> bool:
    return s.startswith(("git@", "git://", "https://", "http://")) and (
        s.endswith(".git") or "github.com" in s or "gitlab.com" in s or "bitbucket" in s
    )


def _is_zip_url(s: str) -> bool:
    return s.lower().split("?")[0].split("#")[0].endswith(".zip")


def _is_http_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _cache_dir(target: str) -> Path:
    d = _CACHE_ROOT / _hash_target(target)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clone_git(url: str, dest: Path, timeout: int = 60) -> Path:
    if any(dest.iterdir()):
        try:
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--depth=1", "--ff-only"],
                check=True, timeout=timeout, capture_output=True,
            )
        except subprocess.CalledProcessError:
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth=1", url, str(dest)],
                check=True, timeout=timeout, capture_output=True,
            )
        return dest
    subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        check=True, timeout=timeout, capture_output=True,
    )
    return dest


def _download_zip(url: str, dest: Path, timeout: int = 60) -> Path:
    zip_path = dest / "source.zip"
    if not zip_path.exists():
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            zip_path.write_bytes(resp.read())
    extract_dir = dest / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = (extract_dir / member).resolve()
            if not str(member_path).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Zip-slip attempt blocked: {member}")
        zf.extractall(extract_dir)
    children = [c for c in extract_dir.iterdir() if c.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def _resolve_local_zip(path: Path) -> Path:
    extract_dir = path.parent / f".{path.stem}-extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            member_path = (extract_dir / member).resolve()
            if not str(member_path).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Zip-slip attempt blocked: {member}")
        zf.extractall(extract_dir)
    children = [c for c in extract_dir.iterdir() if c.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def resolve_input(target: str | Path, *, force_refresh: bool = False) -> Path:
    """Resolve a skill source to a local directory Path.

    Args:
        target: local dir/file/zip path, git URL, or http(s) URL.
        force_refresh: ignore cache; re-clone / re-download.

    Returns:
        Path to a directory containing the skill files.

    Raises:
        FileNotFoundError: local path missing.
        ValueError: unsupported URL scheme.
        RuntimeError: clone / download / extract failure.
    """
    s = str(target)
    p = Path(s).expanduser()

    if p.exists():
        if p.is_file() and p.suffix.lower() == ".zip":
            return _resolve_local_zip(p)
        if p.is_dir():
            return p.resolve()
        if p.is_file():
            return p.parent.resolve()
        raise FileNotFoundError(f"Path does not exist: {p}")

    if _is_http_url(s) or s.startswith("git@") or s.startswith("git://"):
        cd = _cache_dir(s)
        if force_refresh:
            shutil.rmtree(cd, ignore_errors=True)
            cd.mkdir(parents=True, exist_ok=True)
        if _is_git_url(s) and not _is_zip_url(s):
            try:
                return _clone_git(s, cd)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"git clone failed for {s}: {e}") from e
        if _is_zip_url(s) or _is_http_url(s):
            try:
                return _download_zip(s, cd)
            except Exception as e:
                raise RuntimeError(f"download/extract failed for {s}: {e}") from e

    parsed = urlparse(s)
    if parsed.scheme and parsed.scheme not in ("file", ""):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme} ({s})")
    raise FileNotFoundError(f"Skill target not found: {s}")


def clear_cache() -> int:
    """Wipe the entire scan cache. Returns number of dirs removed."""
    if not _CACHE_ROOT.exists():
        return 0
    n = 0
    for child in _CACHE_ROOT.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            n += 1
    return n
