"""
OSV.dev vulnerability lookup client.

Lifted from NVIDIA/SkillSpector (Apache 2.0) — same protocol:
batch endpoint, 1h in-memory cache, GHSA-severity priority,
CVSS-vector fallback estimator. No external CVSS library.

Public API:
    parse_requirements_txt(content) -> list[tuple[name, version]]
    query_packages(pkgs, ecosystem="PyPI") -> list[VulnResult]
    is_available() -> bool
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

OSV_BATCH_URL: Final = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL: Final = "https://api.osv.dev/v1/vulns"
REQUEST_TIMEOUT: Final = 10.0
CACHE_TTL_SECS: Final = 3600.0

ECOSYSTEM_PYPI = "PyPI"
ECOSYSTEM_NPM = "npm"

_REQ_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*([<>=!~]=?\s*[^\s,#;]+)?")
_cache: dict[tuple[str, str | None, str], tuple[float, list["VulnResult"]]] = {}


@dataclass(frozen=True)
class VulnResult:
    vuln_id: str
    summary: str
    severity: str
    aliases: tuple[str, ...]


def parse_requirements_txt(content: str) -> list[tuple[str, str | None]]:
    """Parse requirements.txt-style content. Returns (name, version_or_None)."""
    out: list[tuple[str, str | None]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _REQ_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        spec = m.group(2)
        version: str | None = None
        if spec:
            version = re.sub(r"^[<>=!~]+\s*", "", spec).strip() or None
        out.append((name, version))
    return out


def clear_cache() -> None:
    _cache.clear()


def _cache_key(name: str, version: str | None, ecosystem: str) -> tuple[str, str | None, str]:
    return (name.lower().replace("_", "-"), version, ecosystem)


def _get_cached(key: tuple[str, str | None, str]) -> list[VulnResult] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if (time.monotonic() - ts) > CACHE_TTL_SECS:
        del _cache[key]
        return None
    return results


def _put_cache(key: tuple[str, str | None, str], results: list[VulnResult]) -> None:
    _cache[key] = (time.monotonic(), results)


_CVSS_VECTOR_RE = re.compile(r"CVSS:[34][.\d]*/(.+)")
_CVSS_HIGH_METRICS = {
    "AV:N", "AC:L", "PR:N", "UI:N", "S:C", "C:H", "I:H", "A:H",
    "AT:N", "VC:H", "VI:H", "VA:H", "SC:H", "SI:H", "SA:H",
}


def _estimate_cvss_severity(vector: str) -> str | None:
    m = _CVSS_VECTOR_RE.match(vector)
    if not m:
        return None
    metrics = m.group(1).split("/")
    high_count = sum(1 for metric in metrics if metric in _CVSS_HIGH_METRICS)
    total = len(metrics)
    if total == 0:
        return None
    ratio = high_count / total
    if ratio >= 0.75:
        return "CRITICAL"
    if ratio >= 0.5:
        return "HIGH"
    if ratio >= 0.25:
        return "MEDIUM"
    return "LOW"


def _severity_from_vuln(vuln: dict) -> str:
    db_specific = vuln.get("database_specific") or {}
    ghsa_severity = (db_specific.get("severity") or "").upper()
    if ghsa_severity:
        return ghsa_severity
    for affected in vuln.get("affected") or []:
        eco = affected.get("ecosystem_specific") or {}
        sev = (eco.get("severity") or "").upper()
        if sev:
            return sev
    for sev_entry in vuln.get("severity") or []:
        score = sev_entry.get("score") or ""
        if score:
            estimated = _estimate_cvss_severity(score)
            if estimated:
                return estimated
    return "HIGH"


def _parse_vuln(vuln: dict) -> VulnResult:
    return VulnResult(
        vuln_id=vuln.get("id", "UNKNOWN"),
        summary=(vuln.get("summary") or (vuln.get("details") or "")[:200]),
        severity=_severity_from_vuln(vuln),
        aliases=tuple(vuln.get("aliases") or ()),
    )


@contextmanager
def _http_client():
    import httpx
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        yield c


def _fetch_vuln_details(vuln_ids: list[str]) -> list[VulnResult]:
    results: list[VulnResult] = []
    with _http_client() as client:
        for vid in vuln_ids[:10]:
            try:
                resp = client.get(f"{OSV_VULN_URL}/{vid}")
                resp.raise_for_status()
                results.append(_parse_vuln(resp.json()))
            except Exception:
                results.append(VulnResult(vuln_id=vid, summary="", severity="HIGH", aliases=()))
    return results


def query_packages(packages: list[tuple[str, str | None]],
                   ecosystem: str = ECOSYSTEM_PYPI) -> list[list[VulnResult]]:
    """Query OSV.dev for vulnerabilities across a batch of packages.

    Returns a list parallel to *packages*; each element is the (possibly empty)
    list of vulnerabilities for that package.

    On network/API failure returns [[] for _ in packages] (caller should fall
    back to static data).
    """
    if not packages:
        return []
    all_results: list[list[VulnResult]] = [[] for _ in packages]
    uncached_indices: list[int] = []
    uncached_queries: list[dict] = []
    for i, (name, version) in enumerate(packages):
        key = _cache_key(name, version, ecosystem)
        cached = _get_cached(key)
        if cached is not None:
            all_results[i] = cached
        else:
            uncached_indices.append(i)
            q: dict = {"package": {"name": name, "ecosystem": ecosystem}}
            if version:
                q["version"] = version
            uncached_queries.append(q)
    if not uncached_queries:
        return all_results
    try:
        with _http_client() as client:
            resp = client.post(OSV_BATCH_URL, json={"queries": uncached_queries})
            resp.raise_for_status()
            batch_results = resp.json().get("results", [])
        for batch_idx, idx in enumerate(uncached_indices):
            if batch_idx >= len(batch_results):
                break
            vulns_raw = batch_results[batch_idx].get("vulns") or []
            if not vulns_raw:
                _put_cache(_cache_key(packages[idx][0], packages[idx][1], ecosystem), [])
                continue
            vuln_ids = [v["id"] for v in vulns_raw if "id" in v]
            vuln_details = _fetch_vuln_details(vuln_ids)
            all_results[idx] = vuln_details
            _put_cache(_cache_key(packages[idx][0], packages[idx][1], ecosystem), vuln_details)
    except Exception as e:
        logger.warning("OSV.dev query failed, returning empty results: %s", e)
        return [[] for _ in packages]
    return all_results


def is_available() -> bool:
    """Quick connectivity check against the OSV.dev API."""
    try:
        with _http_client() as c:
            resp = c.post(
                OSV_BATCH_URL,
                json={"queries": [{"package": {"name": "pip", "ecosystem": ECOSYSTEM_PYPI}}]},
            )
            return resp.status_code == 200
    except Exception:
        return False
