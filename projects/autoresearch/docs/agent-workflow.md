# Agent Workflow

This project expects real agent usage to happen in the upstream repo, not here.

## Step 1: Pick the right upstream repo

- macOS: `https://github.com/miolini/autoresearch-macos`
- Unix/NVIDIA: `https://github.com/karpathy/autoresearch`

## Step 2: Run the local showcase first

Generate the teaching artifacts and launch briefs:

```bash
cd projects/autoresearch
make run
```

## Step 3: Read the matching brief

Choose one of:

- `artifacts/agent/codex_macos.md`
- `artifacts/agent/codex_unix.md`
- `artifacts/agent/claude_macos.md`
- `artifacts/agent/claude_unix.md`

## Step 4: Clone the upstream repo

Example:

```bash
git clone https://github.com/miolini/autoresearch-macos.git
cd autoresearch-macos
uv sync
uv run prepare.py
```

Swap the repository URL if you are on the Unix track.

## Step 5: Launch Codex or Claude Code in that repo

Recommended kickoff prompt:

```text
Read README.md, prepare.py, train.py, and program.md.
Do not modify prepare.py or install new dependencies.
Create results.tsv if it is missing.
Run one baseline experiment, log the result, and then continue the keep/discard loop using val_bpb and code simplicity.
```

## Step 6: Preserve the workflow rules

- Keep `results.tsv` local and untracked.
- Treat `prepare.py` as fixed.
- Only advance the branch when the result is actually better or the change is meaningfully simpler.
- Kill experiments that run far beyond the expected budget.

## Step 7: Iterate on `program.md` intentionally

The human is not out of the loop. The main human-owned lever is the instruction surface in `program.md`.

A useful pattern is:

1. let the agent run,
2. inspect the result trace,
3. rewrite `program.md` to tighten heuristics or priorities,
4. run again.
