# self-review-mode

Pre-emptive adversarial **self-speculation** — the Hermes inference-time
realization of arXiv:2607.25816 ("Speculate While You Reason").

## Why

The existing adversarial-work-review loop is **post-hoc**: work is built, a
hostile reviewer (often a fresh subagent) finds issues, the fixer repairs, and
the loop re-reviews until clean. The paper's distinguishing insight is
**self-speculation**: the *deployed agent itself* is the best predictor of what
a hostile reviewer will find, if prompted from its own partial trajectory before
it ships. External draft reviewers carry a *speculator–agent gap* — they guess a
different call than the agent will actually make (and cost extra weights / KV
cache / a subagent round-trip). Self-speculation closes that gap.

This plugin is **not** a replacement for compounding-loops or the
adversarial-work-review skill. It does not block exits, own convergence, or parse
message-history verdicts. It adds the *pre-emptive* self-review mode and the
reward-shaping mechanics the paper contributes, so the loop can:
- predict its own likely blockers **before** shipping (cheap, on-policy),
- score how well a self-review anticipated the real findings (hard-gate
  severity alignment + token-F1 + miss penalty),
- detect **thrashing** (repeated review passes that surface no new findings) and
  stop early — the paper's "penalty for meaningless repetition."

## Tools

| Tool | Purpose |
|------|---------|
| `self_review` | Emit the SPECULATOR-mode suffix (prompt to predict findings a hostile reviewer would raise) + a mode-isolation card, and set the on-disk mode marker. |
| `self_review_score` | Score predicted vs. actual findings: `alignment_score` in `[-1,1]`, `miss_breakdown` (incl. blocker/major misses), and a `thrash_detected` flag. |

## Scoring model (paper transfer)

1. **Hard gate (tool-name gate analog).** Predicted severity must *exactly*
   match the real severity. A blocker reported as a minor earns zero — it would
   not have been raised correctly and is not "reusable." This is anti-Goodhart:
   over-lenient self-review (downgrade everything, or predict nothing) scores
   negative.
2. **Token-F1 partial credit (argument-F1 analog).** Once severity matches, score
   how close the evidence came, via token-F1.
3. **Miss penalty.** Gold findings no prediction matched are penalized in
   proportion to severity weight (blocker miss = 8x a nit miss).
4. **Thrash detector.** N consecutive passes each finding `< min_new` new
   findings → flag for early termination.

## Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `SELF_REVIEW_ENABLED` | `1` | Master switch. `0` disables registering tools/hooks. |
| `SELF_REVIEW_SUFFIX` | `0` | When `1`, the `pre_llm_call` hook injects the speculator suffix as context on build-bound user turns (returns `{"context": ...}`). Default off (prefer the explicit `self_review` tool). |
| `SELF_REVIEW_MODE_FILE` | `~/.hermes/loop-state/self-review-mode.json` | Mode-isolation marker path. |

## Mode isolation (the "reset optimizer state at mode switch" analog)

Roles are: `BUILDER`, `SPECULATOR`, `REVIEWER`. Switch roles only via an explicit
mode card; after a SPECULATOR/REVIEWER block, emit `[MODE RESET]` before writing
more code so builder mode starts clean (no reviewer framing bleeding into new
code). The plugin persists the current role to the mode file for cross-call
continuity.

## Tests

```
pytest plugins/self-review-mode/tests/ -q
```

## License / provenance

Implements ideas from arXiv:2607.25816 (CC BY 4.0), synthesized for Hermes.
