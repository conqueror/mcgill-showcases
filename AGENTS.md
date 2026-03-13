# Repo Instructions

This repository is public and open source.

## Core Rules

- Do not copy private policy, wording, or tooling assumptions from other repositories.
- Prefer small, reproducible, student-friendly project structures.
- Follow `docs/new-showcase-playbook.md` when adding or changing showcase projects.
- Keep commands runnable on a normal laptop with local tools only.
- Treat `.codex/` and `docs/agents/` as the source of truth for the public harness-lite flow in this repo.

## Harness Lite

- This repo uses a public-safe harness bootstrap, not a private enterprise control plane.
- Supported harness behavior is defined by:
  - `.codex/config.toml`
  - `.codex/harness/role-skill-matrix.toml`
  - `docs/agents/oodaris-harness-v2-operating-pack.md`
- If a task requires unsupported harness features, stop and design the extension before implementing it.

## Showcase Quality

- New showcases should include `README.md`, `Makefile`, `pyproject.toml`, `src/`, `scripts/`, `tests/`, `docs/`, `artifacts/`, and `artifacts/manifest.json`.
- Prefer script-first execution paths over notebook-only workflows.
- Keep artifact names stable and document how to interpret them.
- Add root-level integrations for CI, docs, and issue templates when a new showcase is introduced.
