# New Showcase Playbook

This playbook defines the required standards for adding a new showcase project to this repository.

Use this as a strict checklist. A new showcase is not complete until every required item is satisfied.

## 1) Scope And Learning Contract

- Define one primary learning outcome.
- Define the target audience level (beginner, intermediate, advanced).
- Define expected runtime on a student laptop.
- State prerequisites clearly in the project `README.md`.
- Keep scope tight: one coherent workflow per showcase.

## 2) Required Project Structure

Each project should include:

- `README.md` with quickstart and artifact interpretation guide.
- `Makefile` with required quality and runtime targets.
- `src/<package_name>/` for implementation code.
- `scripts/` for runnable pipeline and verification entrypoints.
- `tests/` for unit/integration checks.
- `artifacts/` (git-ignored where needed) and `artifacts/manifest.json`.

## 3) Required Make Targets

Every showcase must expose the following targets:

- `sync`: install dependencies.
- `ruff`: lint checks.
- `ty` (or equivalent type-check target): static typing checks.
- `test`: unit/integration tests.
- `check`: consolidated quality gate.
- `smoke`: lightweight runnable path for demos.
- `verify`: artifact contract or manifest verification.

## 4) Required Documentation Sections In Project README

Every project `README.md` must include:

- Project purpose and learning outcomes.
- Prerequisites.
- Quickstart commands.
- Key artifacts and what each artifact means.
- Common failure modes and how to recover.
- Suggested next project(s) in the track.

## 5) Artifact Contract Requirements

Every project must publish and maintain:

- `artifacts/manifest.json` with `required_files`.
- Deterministic artifact paths used by README and tests.
- A script-based verifier (`scripts/verify_artifacts.py` or equivalent).

For supervised projects, required contract files include:

- `artifacts/splits/split_manifest.json`
- `artifacts/eda/univariate_summary.csv`
- `artifacts/eda/bivariate_vs_target.csv`
- `artifacts/eda/missingness_summary.csv`
- `artifacts/eda/correlation_matrix.csv`
- `artifacts/leakage/leakage_report.csv`
- `artifacts/eval/metrics_summary.csv`
- `artifacts/experiments/experiment_log.csv`

Supervised split manifests must include:

- `train_rows`, `val_rows`, `test_rows`
- `strategy`
- `task_type`
- `random_state`
- `no_overlap_checks_passed`

If a project is added to supervised contract validation, update:

- `shared/config/supervised_projects.json`
- `shared/scripts/verify_supervised_contract.py` compatibility expectations (if needed)

## 6) Test Requirements

Minimum expectations:

- Unit tests for key data/model utility functions.
- Smoke test for end-to-end script path.
- Artifact verification test or command path.
- API projects: contract/openapi consistency checks.

Quality gate requirements:

- New/changed code passes lint and type checks.
- `make check` passes in the project directory.
- Root-level integration commands still pass.

## 7) CI And Root Integration Requirements

For every new project:

- Add project to root `Makefile` orchestration (`sync`, `lint`, `ty`, `test`, `smoke`, `verify` where relevant).
- Add project to `.github/workflows/ci.yml` matrix.
- Add docs references:
  - root `README.md` project catalog,
  - `docs/getting-started.md`,
  - `docs/learning-path.md` when track-relevant,
  - `docs/aspect-coverage-matrix.md` when introducing new methods.
- Add issue template project option updates (bug/learning templates).

## 8) Student Experience Checklist

- Commands are copy/paste runnable.
- Default path is laptop-friendly.
- Quick mode exists where full runs are heavy.
- Artifact names are stable and explicitly referenced.
- At least one interpretation prompt is present in docs.

## 9) Definition Of Done For New Showcase PRs

All items below must be true:

- Scope is clear and focused on one learning objective.
- Local project quality gate (`make check`) passes.
- Root quality gates pass without regressions.
- Artifact contract is verified and documented.
- Student-facing docs and track mapping are updated.
- CI includes the new project and is green.
