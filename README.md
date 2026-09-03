# Agentic ops

Original plugins and tooling built for daily production agent work.

Not a fork dump. Not a resume-scoring demo.

## Flagship (source in this repo)

| Plugin | What it does |
|---|---|
| `compounding-loops` | Plan → build → adversarial review until two consecutive clean passes |
| `hindsight-governance` | Forget + steward approval so deprecated memory cannot shadow replacements |
| `self-router` | Session-model cascade; `/model` swap applies to subsystems |
| `silent_build_enforcer` | Kill mid-build narration; send input gates + closeout only |
| `caveman_enforcer` | Compress style, not meaning |
| `health_monitor` | Agent/host health checks |
| `harness-conductor` | Harness loop orchestration |

## Catalog

[CATALOG.md](./CATALOG.md) — original plugins not in upstream Hermes.

## Secret scan (mandatory before push)

```bash
python3 scripts/scan_secrets.py --root .
git config core.hooksPath .githooks   # once per clone
```

Exit 0 required. Pre-push hook refuses otherwise.

## Also here

`resume-scoring/` is a **port** of interviewstreet/hiring-agent capabilities (extract, enrich, rubric). Kept because we use it. It is not the original work.

## License

MIT
