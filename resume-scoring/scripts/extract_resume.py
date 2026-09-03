#!/usr/bin/env python3
"""Extract visible resume text from PDF, Markdown, or plain text.

PDF: PyMuPDF visible-only extract when installed; else pdftotext (no hidden-text strip).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


HIDDEN_SIZE_LT = 1.0
NEAR_WHITE = 0xF0F0F0


def _pdf_to_text(path: Path) -> str:
    binary = shutil.which("pdftotext")
    if not binary:
        raise RuntimeError("pdftotext not found (poppler-utils)")
    proc = subprocess.run(
        [binary, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"pdftotext failed: {err or proc.returncode}")
    return proc.stdout.decode("utf-8", errors="replace")


def _span_hidden_reasons(span: dict[str, Any], page_rect: tuple[float, float, float, float]) -> list[str]:
    text = (span.get("text") or "").strip()
    if not text:
        return []
    reasons: list[str] = []
    size = float(span.get("size") or 0)
    if size < HIDDEN_SIZE_LT:
        reasons.append("tiny_font")
    bbox = span.get("bbox") or (0, 0, 0, 0)
    x0, y0, x1, y1 = page_rect
    if bbox[2] < x0 or bbox[0] > x1 or bbox[3] < y0 or bbox[1] > y1:
        reasons.append("off_page")
    color = span.get("color", 0)
    if isinstance(color, int) and color >= NEAR_WHITE:
        reasons.append("near_white")
    flags = int(span.get("flags") or 0)
    if flags & 16:
        reasons.append("invisible_flag")
    return reasons


def _visible_pdf(path: Path) -> tuple[str, list[dict[str, Any]], int] | None:
    try:
        import fitz  # type: ignore
    except ImportError:
        return None
    hidden: list[dict[str, Any]] = []
    pages: list[str] = []
    with fitz.open(path) as doc:
        page_count = doc.page_count
        for i, page in enumerate(doc):
            rect = page.rect
            page_rect = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            lines_out: list[str] = []
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    visible = []
                    for span in line.get("spans", []):
                        reasons = _span_hidden_reasons(span, page_rect)
                        text = span.get("text") or ""
                        if reasons:
                            if text.strip():
                                hidden.append(
                                    {
                                        "page": i + 1,
                                        "text": text.strip()[:80],
                                        "reasons": reasons,
                                        "size": span.get("size"),
                                    }
                                )
                            continue
                        visible.append(text)
                    if visible:
                        lines_out.append("".join(visible))
            pages.append("\n".join(lines_out))
    return "\n\n".join(pages), hidden, page_count


def json_resume_to_text(data: dict[str, Any]) -> str:
    lines: list[str] = []
    if isinstance(data.get("basics"), dict):
        basics = data["basics"]
        if basics.get("name"):
            lines.append(str(basics["name"]))
        if basics.get("summary"):
            lines.append(str(basics["summary"]))
        for profile in basics.get("profiles") or []:
            if isinstance(profile, dict) and profile.get("url"):
                lines.append(f"{profile.get('network', 'profile')}: {profile['url']}")
        for key in ("work", "education", "skills", "projects"):
            block = data.get(key)
            if block:
                lines.append(f"## {key}")
                lines.append(json.dumps(block, indent=2, ensure_ascii=False))
        return "\n".join(lines).strip() + "\n"

    if data.get("title"):
        lines.append(str(data["title"]))
    if data.get("summary"):
        lines.append(str(data["summary"]))
    if data.get("linkedin"):
        lines.append(f"LinkedIn: {data['linkedin']}")
    if data.get("github"):
        lines.append(f"GitHub: {data['github']}")
    if data.get("portfolio"):
        lines.append(f"Portfolio: {data['portfolio']}")
    if data.get("core_competencies"):
        lines.append("## Competencies")
        lines.extend(f"- {c}" for c in data["core_competencies"])
    if data.get("technical_skills"):
        lines.append("## Skills")
        lines.extend(f"- {s}" for s in data["technical_skills"])
    if data.get("experience"):
        lines.append("## Experience")
        for job in data["experience"]:
            header = " ".join(
                str(job[k]) for k in ("title", "company", "dates") if job.get(k)
            )
            lines.append(f"### {header}")
            for h in job.get("highlights") or []:
                lines.append(f"- {h}")
    if data.get("education"):
        lines.append("## Education")
        for edu in data["education"]:
            lines.append(
                " - ".join(
                    str(edu[k]) for k in ("degree", "institution", "focus") if edu.get(k)
                )
            )
    if data.get("certifications"):
        lines.append("## Certifications")
        lines.extend(f"- {c}" for c in data["certifications"])
    if data.get("projects"):
        lines.append("## Projects")
        for proj in data["projects"]:
            if isinstance(proj, dict):
                name = proj.get("name") or "project"
                desc = proj.get("description") or ""
                url = proj.get("url") or ""
                lines.append(f"### {name}")
                if desc:
                    lines.append(desc)
                if url:
                    lines.append(url)
    return "\n".join(lines).strip() + "\n"


def extract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    hidden: list[dict[str, Any]] = []
    page_count = 1
    scan = "n/a"
    if suffix == ".pdf":
        parsed = _visible_pdf(path)
        if parsed is not None:
            text, hidden, page_count = parsed
            scan = "ok"
            if not text.strip() and not hidden:
                text = _pdf_to_text(path)
        else:
            text = _pdf_to_text(path)
            hidden = []
            page_count = max(1, text.count("\f") + 1)
            scan = "unavailable"
            print(
                json.dumps({
                    "warning": "pymupdf missing; hidden PDF text not stripped. Use: uv run --with pymupdf python3 extract_resume.py"
                }),
                file=sys.stderr,
            )
    elif suffix in {".md", ".txt", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("resume json must be an object")
        text = json_resume_to_text(payload)
    else:
        raise ValueError(f"unsupported type: {suffix or 'none'} (pdf/md/txt/json)")
    return {
        "source": str(path.resolve()),
        "text": text.strip("\n") + "\n",
        "hidden_text": hidden,
        "hidden_text_scan": scan,
        "page_count": page_count,
        "char_count": len(text),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = extract(args.path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
