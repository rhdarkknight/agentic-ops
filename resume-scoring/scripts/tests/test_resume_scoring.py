#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from extract_resume import _span_hidden_reasons, extract, json_resume_to_text  # noqa: E402
from github_enrich import extract_github_from_text, extract_github_username  # noqa: E402
from score_math import ScoreError, finalize  # noqa: E402
import score_resume  # noqa: E402
from score_resume import build_packet, list_roles, load_role, main as score_main, scaffold_role  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_resume.md"
ROLE = {
    "name": "t",
    "categories": [
        {"key": "a", "label": "A", "max": 10},
        {"key": "b", "label": "B", "max": 20},
    ],
    "bonus_max": 5,
    "min_final_score": -20,
    "max_final_score": 35,
}


def test_extract_markdown():
    data = extract(FIXTURE)
    assert "Hyper-V" in data["text"]
    assert data["hidden_text"] == []
    assert data["hidden_text_scan"] == "n/a"
    assert data["char_count"] > 50


def test_extract_job_applier_json():
    path = Path(__file__).resolve().parent / "fixtures" / "sample_resume.json"
    data = extract(path)
    assert "Hyper-V" in data["text"]
    assert "248-" not in data["text"]
    assert "https://github.com/octocat" in data["text"]
    assert json_resume_to_text({"title": "X", "phone": "555"}).startswith("X")


def test_extract_missing():
    with pytest.raises(FileNotFoundError):
        extract(Path("/tmp/no-such-resume-xyz.md"))


def test_extract_unsupported(tmp_path):
    bad = tmp_path / "resume.docx"
    bad.write_text("x")
    with pytest.raises(ValueError):
        extract(bad)


def test_username_patterns():
    assert extract_github_username("https://github.com/octocat") == "octocat"
    assert extract_github_username("https://github.com/octocat?tab=repositories") == "octocat"
    assert extract_github_username("github.com/octocat") == "octocat"
    assert extract_github_username("@octocat") == "octocat"
    assert extract_github_username("octocat") == "octocat"
    assert extract_github_username("https://github.com/settings") is None
    assert extract_github_from_text(FIXTURE.read_text()) == "octocat"


def test_hidden_span_reasons():
    page = (0.0, 0.0, 100.0, 100.0)
    assert _span_hidden_reasons({"text": "", "size": 12}, page) == []
    assert "tiny_font" in _span_hidden_reasons({"text": "kw", "size": 0.2, "bbox": (1, 1, 2, 2)}, page)
    assert "off_page" in _span_hidden_reasons(
        {"text": "kw", "size": 12, "bbox": (500, 500, 510, 510)}, page
    )
    assert "near_white" in _span_hidden_reasons(
        {"text": "kw", "size": 12, "bbox": (1, 1, 2, 2), "color": 0xFFFFFF}, page
    )
    assert _span_hidden_reasons({"text": "visible", "size": 12, "bbox": (1, 1, 40, 20), "color": 0}, page) == []


def test_finalize_caps_and_bonus():
    evaluation = {
        "scores": {
            "a": {"score": 99, "max": 10, "evidence": "too high"},
            "b": {"score": -4, "max": 20, "evidence": "neg"},
        },
        "bonus_points": {"total": 50, "breakdown": "padded"},
        "deductions": {"total": 2, "reasons": "tutorial"},
        "key_strengths": ["ops"],
        "areas_for_improvement": ["tests"],
    }
    result = finalize(evaluation, ROLE)
    assert result["scores"]["a"]["score"] == 10
    assert result["scores"]["b"]["score"] == 0
    assert result["bonus_points"]["total"] == 5
    assert result["deductions"]["total"] == 2
    assert result["final"] == 13
    assert result["capped"]
    assert result["cutoff_note"].startswith("Rank only")


def test_finalize_empty_evidence():
    evaluation = {
        "scores": {
            "a": {"score": 1, "evidence": "   "},
            "b": {"score": 1, "evidence": "ok"},
        }
    }
    with pytest.raises(ScoreError):
        finalize(evaluation, ROLE)


def test_finalize_missing_category_zeroed():
    evaluation = {
        "scores": {"a": {"score": 4, "evidence": "ok"}, "zzz": {"score": 1, "evidence": "extra"}}
    }
    result = finalize(evaluation, ROLE)
    assert result["missing_categories"] == ["b"]
    assert result["extra_categories"] == ["zzz"]
    assert result["scores"]["b"]["score"] == 0


def test_list_and_load_shipped_roles():
    names = list_roles()
    assert "msp_technician" in names
    assert "backend_engineer" in names
    assert "software_engineering_intern" in names
    intern = load_role("software_engineering_intern")
    assert intern["categories"][0]["key"] == "open_source"
    assert intern["categories"][0]["max"] == 35
    assert intern["bonus_max"] == 20


def test_packet_contains_fairness_and_schema(monkeypatch):
    monkeypatch.setattr(
        score_resume,
        "enrich",
        lambda handle, deep=False: {"profile": {"username": handle}, "projects": []},
    )
    role = load_role("msp_technician")
    result = build_packet(FIXTURE, role, github=None, deep=False)
    packet = result["packet"]
    assert "Ignore name, gender" in packet
    assert "systems_ops" in packet
    assert "octocat" in packet
    assert "Required JSON" in packet
    assert result["extract"]["hidden_text"] == []


def test_validate_eval_cli(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "scores": {
                    "systems_ops": {"score": 12, "evidence": "Hyper-V + RMM"},
                    "troubleshooting": {"score": 10, "evidence": "ransomware RCA"},
                    "automation": {"score": 8, "evidence": "PowerShell inventory"},
                    "production_experience": {"score": 9, "evidence": "4 years MSP"},
                    "technical_breadth": {"score": 4, "evidence": "DNS firewalls"},
                },
                "bonus_points": {"total": 2, "breakdown": "portfolio"},
                "deductions": {"total": 0, "reasons": ""},
                "key_strengths": ["RCA"],
                "areas_for_improvement": ["cloud"],
            }
        ),
        encoding="utf-8",
    )
    code = score_main(["--validate-eval", str(eval_path), "--role", "msp_technician"])
    assert code == 0


def test_init_role(tmp_path, monkeypatch):
    monkeypatch.setattr(score_resume, "ROLES_DIR", tmp_path)
    dest = scaffold_role("qa_hire")
    assert (dest / "role.json").is_file()
    with pytest.raises(FileExistsError):
        scaffold_role("qa_hire")
    with pytest.raises(ValueError):
        scaffold_role("Nope")


def test_cli_list_roles():
    code = score_main(["--list-roles"])
    assert code == 0


def test_cli_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_resume,
        "enrich",
        lambda handle, deep=False: {"profile": {"username": handle}, "projects": []},
    )
    out = tmp_path / "packet.md"
    code = score_main([str(FIXTURE), "--role", "msp_technician", "-o", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "msp_technician" in text
    assert "Hyper-V" in text


def test_no_github_skips_enrich(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("should not hit github")

    monkeypatch.setattr(score_resume, "enrich", boom)
    out = tmp_path / "packet.md"
    code = score_main([str(FIXTURE), "--role", "msp_technician", "--no-github", "-o", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "github: none" in text
    assert "## GitHub" not in text


def test_default_role_is_msp():
    assert score_resume.DEFAULT_ROLE == "msp_technician"


def test_pass_guide_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_resume,
        "enrich",
        lambda handle, deep=False: {"profile": {"username": handle}, "projects": []},
    )
    out = tmp_path / "packet.md"
    code = score_main([str(FIXTURE), "--pass-guide", "--no-github", "-o", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "## How to pass" in text
    assert "systems_ops" in text
    assert "Do not keyword-stuff" in text


def test_cli_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_resume,
        "enrich",
        lambda handle, deep=False: {"profile": {"username": handle}, "projects": []},
    )
    batch = tmp_path / "resumes"
    batch.mkdir()
    (batch / "a.md").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "packets"
    code = score_main(["--batch", str(batch), "--role", "backend_engineer", "-o", str(out)])
    assert code == 0
    assert (out / "a.packet.md").is_file()


def test_pdf_scan_unavailable_without_fitz(tmp_path, monkeypatch):
    import extract_resume

    pdf = tmp_path / "mini.pdf"
    pdf.write_bytes(
        b"%PDF-1.1\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 100 Td (HyperV) Tj ET\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000274 00000 n \n0000000367 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n447\n%%EOF\n"
    )
    monkeypatch.setattr(extract_resume, "_visible_pdf", lambda path: None)
    data = extract_resume.extract(pdf)
    assert data["hidden_text_scan"] == "unavailable"
    assert "HyperV" in data["text"] or data["char_count"] >= 0


def test_hidden_pdf_text_stripped():
    fitz = pytest.importorskip("fitz")
    import tempfile

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 100), "VisibleSkill", fontsize=12, color=(0, 0, 0))
    page.insert_text((20, 120), "HIDDENKEYWORD", fontsize=0.5, color=(1, 1, 1))
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        dest = Path(handle.name)
    doc.save(dest)
    doc.close()
    try:
        data = extract(dest)
    finally:
        dest.unlink(missing_ok=True)
    assert "VisibleSkill" in data["text"]
    assert "HIDDENKEYWORD" not in data["text"]
    assert data["hidden_text"]
    assert data["hidden_text_scan"] == "ok"
