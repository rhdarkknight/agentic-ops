#!/usr/bin/env python3
"""Fetch GitHub profile + repos for resume enrichment. No LLM."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USERNAME_RE = [
    re.compile(r"https?://github\.com/([^/\s?#]+)", re.I),
    re.compile(r"github\.com/([^/\s?#]+)", re.I),
    re.compile(r"^@([A-Za-z0-9-]+)$"),
    re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)$"),
]
RESERVED = {"orgs", "settings", "topics", "notifications", "login", "signup", "features"}


def load_hermes_env() -> None:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def github_token() -> str | None:
    load_hermes_env()
    for key in ("SNS_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return None


def extract_github_username(value: str) -> str | None:
    if not value:
        return None
    text = value.strip().replace(" ", "")
    for pattern in USERNAME_RE:
        match = pattern.search(text)
        if not match:
            continue
        username = match.group(1)
        username = username.split("?", 1)[0].split("#", 1)[0].strip("/")
        if username.lower() in RESERVED:
            return None
        return username
    return None


def extract_github_from_text(text: str) -> str | None:
    match = re.search(r"https?://github\.com/([A-Za-z0-9-]+)", text or "", re.I)
    if match and match.group(1).lower() not in RESERVED:
        return match.group(1)
    match = re.search(r"github\.com/([A-Za-z0-9-]+)", text or "", re.I)
    if match and match.group(1).lower() not in RESERVED:
        return match.group(1)
    return None


def _get(url: str, params: dict[str, str] | None = None) -> tuple[int, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-resume-scoring",
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"message": str(exc)}
        return exc.code, body
    except URLError as exc:
        return 0, {"message": str(exc.reason)}


def fetch_contributors(owner: str, repo: str) -> list[dict[str, Any]]:
    status, data = _get(f"https://api.github.com/repos/{owner}/{repo}/contributors")
    if status == 200 and isinstance(data, list):
        return data
    return []


def enrich(username_or_url: str, *, deep: bool = False, max_repos: int = 100) -> dict[str, Any]:
    username = extract_github_username(username_or_url)
    if not username:
        raise ValueError(f"could not parse GitHub username from {username_or_url!r}")
    status, profile = _get(f"https://api.github.com/users/{username}")
    if status == 404:
        raise ValueError(f"GitHub user not found: {username}")
    if status != 200 or not isinstance(profile, dict):
        raise RuntimeError(f"GitHub profile error {status}: {profile}")
    status, repos = _get(
        f"https://api.github.com/users/{username}/repos",
        {"sort": "updated", "per_page": str(min(max_repos, 100)), "type": "all"},
    )
    if status != 200 or not isinstance(repos, list):
        raise RuntimeError(f"GitHub repos error {status}: {repos}")

    projects: list[dict[str, Any]] = []
    for repo in repos:
        if repo.get("fork") and int(repo.get("forks_count") or 0) < 5:
            continue
        name = repo.get("name") or ""
        contributor_count = None
        author_commits = None
        project_type = "self_project"
        if deep:
            contributors = fetch_contributors(username, name)
            contributor_count = len(contributors)
            author_commits = 0
            for person in contributors:
                if (person.get("login") or "").lower() == username.lower():
                    author_commits = int(person.get("contributions") or 0)
            if contributor_count > 1:
                project_type = "open_source"
        elif repo.get("fork"):
            project_type = "open_source"
        projects.append(
            {
                "name": name,
                "description": repo.get("description"),
                "github_url": repo.get("html_url"),
                "live_url": repo.get("homepage") or None,
                "language": repo.get("language"),
                "stars": int(repo.get("stargazers_count") or 0),
                "forks": int(repo.get("forks_count") or 0),
                "fork": bool(repo.get("fork")),
                "archived": bool(repo.get("archived")),
                "project_type": project_type,
                "contributor_count": contributor_count,
                "author_commit_count": author_commits,
                "topics": repo.get("topics") or [],
                "updated_at": repo.get("updated_at"),
            }
        )
    projects.sort(key=lambda p: (p["stars"], p["forks"]), reverse=True)
    return {
        "profile": {
            "username": username,
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "public_repos": profile.get("public_repos"),
            "followers": profile.get("followers"),
            "hireable": profile.get("hireable"),
            "blog": profile.get("blog"),
            "created_at": profile.get("created_at"),
        },
        "projects": projects,
        "total_projects": len(projects),
        "deep": deep,
        "note": (
            "project_type is heuristic without --deep (fork => open_source). "
            "Personal non-fork repos are self_project, not OSS."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username_or_url")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--max-repos", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        payload = enrich(args.username_or_url, deep=args.deep, max_repos=args.max_repos)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
