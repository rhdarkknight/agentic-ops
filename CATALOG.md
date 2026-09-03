# High-value plugin catalog

Only plugins that are original (not upstream Hermes), broadly useful to other Hermes operators, and leak-scanned.

Source for these lives under `plugins/`. The rest of the local tree stays private.

## Memory and governance

- [`hindsight-governance`](plugins/hindsight-governance/) — Hindsight Governance

## Agent loop and verification

- [`compounding-loops`](plugins/compounding-loops/) — compounding-loops
- [`harness-conductor`](plugins/harness-conductor/) — version: 0.1.3
- [`self-review-mode`](plugins/self-review-mode/) — self-review-mode
- [`atomic_pipeline`](plugins/atomic_pipeline/) — version: 1.0.2

## Routing and model-agnostic

- [`self-router`](plugins/self-router/) — version: 0.1.0
- [`compositional_skill_router`](plugins/compositional_skill_router/) — version: 0.1.0

## Context and cost

- [`context_compressor`](plugins/context_compressor/) — version: 2.1.0
- [`rate_limit_guard`](plugins/rate_limit_guard/) — Rate Limit Guard Plugin

## Reliability and ops

- [`silent_build_enforcer`](plugins/silent_build_enforcer/) — version: 0.3.0
- [`caveman_enforcer`](plugins/caveman_enforcer/) — version: 1.0.1
- [`health_monitor`](plugins/health_monitor/) — Health Monitor Plugin

## Security

- [`skill-security-scanner`](plugins/skill-security-scanner/) — Hermes plugin.
- [`coding_agent_discipline`](plugins/coding_agent_discipline/) — version: "1.0.0"

## Not published

SNS/PSA/RMM integrations, credential pools, one-off debug plugins, and anything that failed the secret scan stay private.
