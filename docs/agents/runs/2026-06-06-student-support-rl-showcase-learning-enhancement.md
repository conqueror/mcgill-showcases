# Run Ledger — Student Support RL Showcase: Learning-Resource Enhancement

> **Date:** 2026-06-06 · **Status:** complete, all gates green · **Mode:** claude-harness + Workflow fan-outs
> **Spec/contract:** `docs/superpowers/specs/2026-06-06-student-support-rl-showcase-learning-enhancement.md`

## Goal

Turn the correct-but-thinly-documented `projects/student-support-rl-showcase` into a great
graduate-level RL/DRL learning resource: docstrings on every file/class/function, full-equation
concept guides with diagrams, and a few small from-scratch teaching modules that fill real concept
gaps — without changing behavior, breaking determinism, or breaking the artifact contract.

## Approach (multi-agent)

`/claude-harness` as the spine; three sequential `Workflow` fan-outs, each a generator → separate
adversarial-reviewer pipeline (the harness's mandatory no-self-eval gate), with the orchestrator
doing the spec, shared-file integration, notation-anchor docs, and finishing/verification:

1. **Build teaching modules** (3 generators → 3 skeptics): `dynamic_programming.py` (exact `Q*` by
   backward induction), `sarsa.py` (on-policy TD), `policy_gradient.py` (tabular REINFORCE). TDD.
2. **Document every pre-existing file** (~11 targets → reviewers): docstrings + algorithm-line
   comments, strictly non-behavioral, with file-specific honesty notes.
3. **Author concept guides** (8 guides → reviewers): full equations, mermaid diagrams, cross-links.

Orchestrator-authored: the spec, package/Makefile/manifest/script wiring, `math-notes.md`,
`glossary.md`, `00-start-here.md`, README + root-surface updates, ladder diagram.

## What shipped

- 3 new TDD'd modules (+ 25 new tests) wired into `__init__`, scripts (`run_dynamic_programming`,
  `run_sarsa`, `run_policy_gradient`), `run_showcase.py`, Makefile (`run-dp/run-sarsa/run-reinforce`,
  `run-all`), the manifest, the verifier, and the contract test. 5 new required artifacts.
- Docstrings on every module/class/public function across `src/`, `scripts/`, `tests/` (only nested
  test-local closures left undocumented, by design).
- 10 new docs (start-here, glossary, math-notes, 7 concept guides, exercises) + enriched
  algorithm-ladder/learning-guide/method-notes; 13 mermaid diagrams; 0 broken intra-repo links.
- Review-finding fixes: `run_showcase.py` now wired (`make run-all`) and consistent with
  `write_business_memo.py`; `mdp_spec` lists all policies; magic action-count literal removed.
- Repo-root discovery surfaces updated for DP/SARSA/REINFORCE (root README, getting-started,
  aspect-coverage-matrix).

## Gate matrix (final)

| Gate | Result |
|---|---|
| `make check` (ruff + mypy --strict + pytest) | PASS — 36 files, 42 tests |
| `make verify` (artifact contract) | PASS |
| `make smoke` | PASS, ~3s (DRL bridge takes documented fallback without extras) |
| Docstring coverage (AST) | PASS — only 5 nested test closures undocumented |
| Intra-repo markdown links | PASS — 253 checked, 0 broken |

## Independent reviews

- **Completeness critic (subagent):** PASS_WITH_CONCERNS → all resolved. It re-verified equation
  consistency and reproduced every worked numeric example against artifacts. Concerns were a
  6-rung-vs-9-rung drift in three legacy surfaces (fixed) and brittle hardcoded line numbers in a
  guide (de-brittled to grep-able expressions).
- **Codex (external):** `partial` → all 6 findings fixed. Caught truth-drift the per-file reviewers
  missed (prose vs. generated-artifact values / runtime): `deep-rl.md` implied tabular beats DQN
  while the checked-in CSV shows the opposite (made artifact-honest + non-committal); REINFORCE
  baseline/advantage explanation corrected to match the code (per-step `G_t − b`, episode-mean, not a
  running average); hardcoded "400 episodes" → "400 quick / 2000 full"; `00-start-here` "every
  artifact" → "every required artifact" + optional-DRL note; consistent optional-DRL framing across
  README/start-here; repo-root surfaces updated; added a beginner "shortest path" on-ramp.

## Residual risks / follow-ups (pre-existing, out of scope)

- `policies.greedy_action` documents `Raises RuntimeError` on empty input, but `max([])` raises
  `ValueError` first (unreachable guard). Never triggered in practice (action lists are length 4).
- `reporting.recommendation_from_summary` indexes `avg_unsafe_or_questionable_decisions`, which is
  not in the contract's *required* columns for `policy_comparison.csv` (it is emitted by
  `evaluate_policies`, so the real pipeline is fine; a hand-written minimal CSV could `KeyError`).
  Flagged as a background follow-up task.
- Several behavioral test thresholds are intentionally tied to this fixed-seed environment.

## Resume notes

Everything is on disk and green. Optional next steps suggested by reviewers: a tiny from-scratch
actor-critic (actor-critic is currently only instantiated via PPO); a Monte-Carlo-vs-TD note; a
doc-only UCB/Thompson contrast for the bandit. Re-run `make check && make verify` after any change.
