# Autoresearch Showcase

Learn how a fixed-budget autonomous research loop works for tiny language-model pretraining.

This project translates the ideas behind `autoresearch-macos` and `autoresearch` into a guided, readable showcase. It does two things at once:

- teaches the real workflow and decision logic behind the upstream repos,
- gives you concrete Codex and Claude Code launch briefs for macOS and Unix.

The local run in this repo is deterministic and CPU-friendly. The real overnight experiment loop still happens in the upstream repos.

## Learning outcomes

By the end of this project, you should be able to:

- explain why `prepare.py`, `train.py`, and `program.md` have different roles,
- explain why a fixed 5-minute budget makes experiments comparable,
- interpret `val_bpb` as the keep-or-discard metric,
- compare the macOS and Unix variants of autoresearch,
- decide when a change is worth keeping based on quality, memory, and complexity,
- launch a real Codex or Claude Code run against the upstream repo for your platform.

## Quickstart

```bash
cd projects/autoresearch
make sync
make run
make verify
```

Generate a single launch brief on demand:

```bash
make render-macos AGENT=codex
make render-unix AGENT=claude
```

## Key outputs

After `make run`, inspect:

- `artifacts/overview/platform_comparison.csv`
- `artifacts/overview/upstream_snapshot.json`
- `artifacts/overview/research_loop_summary.md`
- `artifacts/analysis/decision_scenarios.csv`
- `artifacts/analysis/simulated_results.tsv`
- `artifacts/agent/codex_macos.md`
- `artifacts/agent/codex_unix.md`
- `artifacts/agent/claude_macos.md`
- `artifacts/agent/claude_unix.md`
- `artifacts/summary.md`

## Why this showcase is useful

- It separates the fixed evaluation harness from the editable research surface.
- It turns the upstream loop into explicit concepts, not just a cool demo.
- It gives you platform-specific launch instructions instead of vague advice.
- It makes the keep/discard policy concrete through simulated results and decision scenarios.

## Suggested study path

1. Run `make run`.
2. Read `docs/learning-guide.md`.
3. Open `artifacts/overview/platform_comparison.csv`.
4. Read `docs/how-the-loop-works.md`.
5. Use `docs/platform-notes.md` to choose macOS or Unix.
6. Open the matching generated brief under `artifacts/agent/`.
7. Launch Codex or Claude Code in the real upstream repo.

## Real agent workflow

This project treats real agent usage as a first-class outcome.

Use:

- `docs/agent-workflow.md` for the full workflow,
- `artifacts/agent/codex_macos.md` or `artifacts/agent/codex_unix.md` for Codex,
- `artifacts/agent/claude_macos.md` or `artifacts/agent/claude_unix.md` for Claude Code.

## Common failure modes

- Treating `train.py` as the only thing that matters and ignoring the role of `program.md`.
- Comparing results across different hardware as if they were directly equivalent.
- Keeping tiny improvements that add too much complexity.
- Forgetting that `results.tsv` should stay local and untracked in the upstream workflow.
- Assuming this repo itself is the best place to run the real overnight experiments.

## Suggested next projects

- `../automl-hpo-showcase/README.md`
- `../model-release-rollout-showcase/README.md`
- `../mlops-drift-production-showcase/README.md`

## Project structure

```text
autoresearch/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/autoresearch_showcase/
├── tests/
├── artifacts/
└── data/
```
