# compounding-loops

Make the M3 **plan → build → test/audit → fix → re-audit → until clean** loop the
default for any non-trivial build, instead of one-shot reviews.

## What it does

After a non-trivial build (≥ 2 `write_file`/`patch`/`execute_code`/`terminal`
tool calls, **or** a user message with build intent + ≥ 1 such call), the
plugin refuses to let the agent exit until it has produced evidence of an
adversarial review pass and the loop has converged:

| Loop state | Gate response |
|---|---|
| Build done, no review yet | Reject — force a review pass with blockers/majors explicitly reported |
| Review found blockers | Reject — fix each, re-review, report new pass with `0 blockers` |
| Review found majors | Reject — address and re-review |
| 1 clean pass | Reject — need one more consecutive clean pass (no regressions) |
| 2 consecutive clean passes | Approve — loop converged |
| `pass ≥ MAX_PASSES` | Approve — hard cap, ship whatever you have |
| User message has bypass keyword (`quick`, `trivial`, `one-liner`) | Approve immediately |

The gate is **stateless** — every decision is derived from the message
history. There is no cross-session state to leak between users.

## Why

Ryan's preferred model (`minimax-m3`) shows its best work when it loops:
plan, build, audit, fix, repeat until nothing is left. Other models
(`glm-5.x` etc.) are too expensive to keep using as primary. By gating
exit on review evidence, this plugin makes the M3 loop discipline the
**default** rather than something Ryan has to remember to ask for.

The existing `harness-conductor` plugin already catches response-shape
problems (empty, truncated, hedge). `compounding-loops` adds the
**process** discipline that M3 performs naturally: the agent cannot
declare done without showing its work.

## Configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `HERMES_LOOPS_ENABLED` | `1` | `0` disables the plugin entirely (responses always approved) |
| `HERMES_LOOPS_MAX_PASSES` | `3` | Hard cap on review passes per session; once hit, ship whatever you have (the approval carries a `reason` so a downstream logger can flag a ship-with-findings outcome) |
| `HERMES_LOOPS_MIN_BUILD_TOOLS` | `2` | Minimum build tool calls to count as "a build happened" |
| `HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN` | `1` | `0` allows exit after a single clean pass |
| `HERMES_LOOPS_BYPASS_KEYWORDS` | `quick,trivial,one-liner` | Comma-separated; any match bypasses the gate for that turn |
| `HERMES_LOOPS_REVIEW_TOOLS` | *(empty)* | Comma-separated tool names that constitute a real review (e.g. `adversarial_review,review,audit`). When set, the gate additionally requires at least one such tool call in the message history before accepting any review evidence — converts the gate from narrative-police to process-police. Empty preserves the original text-evidence behaviour. |
| `HERMES_LOOPS_HARD_FAIL` | `0` | When set, `register` raises if the plugin manager provides no `register_hook`; otherwise it logs a warning and silently no-ops. |
| `HERMES_LOOPS_MAX_TURNS` | `0` (off) | Total mutating/exec tool-call cap per session. When ≥ this, the gate ships with `reason: step cap reached`. Defense against runaway sessions. |
| `HERMES_LOOPS_CIRCUIT_BREAKER` | `0` (off) | When ≥ this, the same tool+args repeating N times anywhere in recent history trips the gate. Hash-based signature (full arg hash, truncated to 16 hex) prevents collisions. Defense against the "same loop again" symptom. |
| `HERMES_LOOPS_PROOF_OF_FIX` | `0` | When `1`, every blocker fix must emit a `Proof-of-fix` block with `revert-verified: yes` before the loop can converge. Debug-grade insistence that fixes were tested, not just claimed. |


## Review evidence format

The plugin parses review evidence from the response text. Use any of:

```
Review pass 1: 0 blockers, 0 majors, 1 minor. Review clean.
Audit pass #2: 3 blockers, 2 majors.
Pass 3: no blockers, 1 major. Fixing.
```

Recognized patterns:
- `review pass N:`, `audit pass N:`, `adversarial pass N:`, `Pass N:` (a colon after the number is required — this avoids false positives like "we did 3 passes over the data" or "pass 5 of the compiler")
- `0 blockers` / `N blockers` / `no blockers` (the *latest* mention in a pass window wins, so `0 blockers, then 3 blockers` resolves to 3)
- `0 majors` / `N majors` / `no majors` (same latest-mention rule)
- `review clean`, `no findings`, `findings resolved`, `review passed`, `pass complete` (optional — a pass with `0 blockers, 0 majors` alone already counts as clean)

## Hooks consumed

- `pre_exit_verify` — fires before the agent returns a final response
- `post_tool_batch_reflect` — fires after each tool batch

## Bypass patterns

Three ways to skip the loop:

1. **Per-turn:** include `quick`, `trivial`, or `one-liner` in the user
   message.
2. **Disable per-task:** set `HERMES_LOOPS_ENABLED=0` for a single run.
3. **Loosen:** set `HERMES_LOOPS_REQUIRE_DOUBLE_CLEAN=0` to allow exit
   after a single clean pass (still requires at least one review).

## Files

- `plugin.yaml` — plugin manifest
- `__init__.py` — hook implementations (`_pre_exit_verify`,
  `_post_tool_batch_reflect`) plus pure helpers (`_extract_latest_review_from_text`,
  `_count_consecutive_clean_passes`, `_highest_pass_seen`)
- `loop_state.py` — persistence adapter for `~/.hermes/loop-state/STATUS.json`
  (cross-session advisory memory; not used for approval decisions)
- `tests/test_compounding_loops.py` — 70 unit tests covering all
  decision branches
- `tests/test_brakes_state.py` — circuit-breaker / step-cap / stuck-cap
  state-file tests
- `tests/test_adv_review_round1.py` — first adversarial-review regressions
- `tests/test_adversarial_probes.py` — second adversarial-review probes
- `tests/test_caveman_enforcer.py` — output-format gate
- `tests/test_final_probes.py` — additional adversarial probes
- `tests/test_loop_hooks.py` — hook registration tests
- `tests/test_probe_same_msg_fix.py` — same-message fix/probe tests
- `tests/test_proof_of_fix.py` — proof-of-fix-block parser tests
- `tests/test_runaway_loop_brakes.py` — 10 regression tests targeting
  the runaway-loop failure modes reported in 2026-06-29 (history-not-tail
  circuit-breaker trap, truncation-collision signatures, synthetic-tool-call
  contamination, bypass-after-cap escape, force_build pollution).

## Related

- `software-development/adversarial-work-review` — the *skill* the loop
  drives. The plugin enforces the discipline; the skill provides the
  procedure.
- `software-development/harness-conductor` — companion plugin that
  enforces response-shape safety (empty, truncated, hedge). The two
  plugins do not conflict; `compounding-loops` defers to it on
  empty-response rejections.
- `software-development/harness-conductor` skill — composition rules
  and verification loop policy.
- Forward Future Loop Library
  (https://signals.forwardfuture.ai/loop-library/) — inspiration; entry
  #002 ("architecture satisfaction loop") matches M3's natural
  behaviour.
