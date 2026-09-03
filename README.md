# Agentic ops

Daily-driver tooling for running agents in production, not demos.

Built and used by [Ryan Smith](https://github.com/rhdarkknight) as VP of operations at an MSP: hiring, upgrades, memory, and verification loops.

## resume-scoring

Resume-to-score pipeline. Not a second agent.

- Extract PDF / Markdown / JSON (strips hidden PDF text when PyMuPDF is present)
- Optional GitHub enrich (`Authorization: token`, never Bearer)
- Role rubrics (`msp_technician` default, plus `backend_engineer` and the public HackerRank intern rubric)
- Deterministic score caps
- `--pass-guide` for reverse-ATS

```bash
python3 resume-scoring/scripts/score_resume.py resume.json --pass-guide --no-github
uv run --with pytest python -m pytest resume-scoring/scripts/tests/test_resume_scoring.py -q
```

## What else runs here (not in this tree yet)

These are in daily use. They stay private until they are sanitized and split out:

- **Hindsight governance** — forget + steward approval so deprecated memory cannot shadow replacements
- **Compounding-loop review** — adversarial review, two consecutive clean passes before ship
- **ACP / Hermes frontends** — same agent core from CLI, editor, and messaging

No credentials, client data, or internal IPs in this repo.

## License

MIT
