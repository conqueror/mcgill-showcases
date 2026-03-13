# Concept Learning Map

Use this map to connect each concept to the exact code, doc, or artifact that teaches it.

| Concept | Where to read or run | What to inspect |
|---|---|---|
| Fixed-budget experimentation | `docs/how-the-loop-works.md`, `artifacts/overview/research_loop_summary.md` | Why every run gets the same 5-minute wall-clock budget |
| Platform differences | `src/autoresearch_showcase/platforms.py`, `artifacts/overview/platform_comparison.csv` | What changes between macOS and Unix, and what stays invariant |
| Mutable vs fixed research surface | `docs/how-the-loop-works.md`, `artifacts/agent/*.md` | Why `train.py` is editable but `prepare.py` is not |
| Keep/discard logic | `src/autoresearch_showcase/decision_policy.py`, `artifacts/analysis/decision_scenarios.csv` | How improvement, crash risk, and complexity trade off |
| Results ledger shape | `artifacts/analysis/simulated_results.tsv` | How `results.tsv` captures commit, score, memory, and status |
| Agent launch design | `src/autoresearch_showcase/agent_brief.py`, `docs/agent-workflow.md` | How Codex or Claude Code is pointed at `program.md` |
| Upstream source grounding | `artifacts/overview/upstream_snapshot.json`, `docs/platform-notes.md` | Which upstream repo and commit each platform guide references |
| Showcase artifact generation | `src/autoresearch_showcase/reporting.py`, `scripts/run_showcase.py` | How the project turns source facts into educational outputs |
