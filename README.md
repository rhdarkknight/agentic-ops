# Agentic ops

Original Hermes plugins built for daily production agent work.

Not a fork dump. Not a resume-scoring demo.

## Flagship

| Plugin | What it does |
|---|---|
| [`compounding-loops`](plugins/compounding-loops/) | Plan → build → adversarial review until two consecutive clean passes |
| [`hindsight-governance`](plugins/hindsight-governance/) | Forget + steward approval so deprecated memory cannot shadow replacements |
| [`self-router`](plugins/self-router/) | Session-model cascade; `/model` swap applies to subsystems |
| [`silent_build_enforcer`](plugins/silent_build_enforcer/) | Kill mid-build narration; input gates + closeout only |
| [`caveman_enforcer`](plugins/caveman_enforcer/) | Compress style, not meaning |
| [`health_monitor`](plugins/health_monitor/) | Agent/host health checks |
| [`harness-conductor`](plugins/harness-conductor/) | Harness loop orchestration |
| [`self-review-mode`](plugins/self-review-mode/) | In-loop self-review before closeout |
| [`atomic_pipeline`](plugins/atomic_pipeline/) | Atomic multi-step tool pipelines |
| [`compositional_skill_router`](plugins/compositional_skill_router/) | Route work to skills by composition |
| [`context_compressor`](plugins/context_compressor/) | Context compression without cache-break surprises |
| [`rate_limit_guard`](plugins/rate_limit_guard/) | Provider rate-limit backoff |
| [`skill-security-scanner`](plugins/skill-security-scanner/) | Scan skills/plugins for unsafe patterns |
| [`coding_agent_discipline`](plugins/coding_agent_discipline/) | Cross-prompt coding-agent rules |

Categorized index: [CATALOG.md](./CATALOG.md)

## Secret scan (mandatory before push)

```bash
python3 scripts/scan_secrets.py --root .
git config core.hooksPath .githooks
```

Exit 0 required. Pre-push hook refuses otherwise.

## Also here

`resume-scoring/` is a **port** of interviewstreet/hiring-agent capabilities. Used inbound. Not original architecture.

## License

MIT
