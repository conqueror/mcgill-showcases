# Learning Guide

## Goal

Understand how autoresearch turns a coding agent into a fixed-budget experiment operator, then launch the real workflow on the platform you actually have.

## Step 1: Generate the local learning artifacts

Run:

```bash
make run
```

This creates the platform comparison, simulated result trace, decision scenarios, and Codex/Claude briefs.

## Step 2: Compare platforms before you choose one

Open:

- `artifacts/overview/platform_comparison.csv`
- `docs/platform-notes.md`

Answer:

1. Which platform do you have?
2. Which repo matches it?
3. Which hardware-specific constraints matter before the first run?

## Step 3: Learn the fixed research loop

Read:

- `docs/how-the-loop-works.md`
- `artifacts/overview/research_loop_summary.md`

Focus on:

- what is fixed,
- what the agent edits,
- why `val_bpb` is the decision metric,
- why the 5-minute cap matters.

## Step 4: Practice keep-or-discard decisions

Inspect:

- `artifacts/analysis/decision_scenarios.csv`
- `artifacts/analysis/simulated_results.tsv`

For each scenario, explain:

1. whether the model improved,
2. whether the change added or removed complexity,
3. whether you would keep, discard, or abandon it.

## Step 5: Read the launch brief for your setup

Choose one:

- `artifacts/agent/codex_macos.md`
- `artifacts/agent/codex_unix.md`
- `artifacts/agent/claude_macos.md`
- `artifacts/agent/claude_unix.md`

The brief gives you the exact repo, preflight commands, and kickoff prompt.

## Step 6: Run the real workflow

Clone the real upstream repo for your platform, then launch Codex or Claude Code there.

Do not confuse the showcase repo with the actual experiment repo:

- this repo teaches the workflow,
- the upstream repo is where the overnight loop runs.

## Step 7: Check your understanding

Use `docs/checkpoint-answer-key.md` after you answer the self-check questions on your own.
