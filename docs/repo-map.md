# Repository Map

- `projects/`: showcase projects.
  - Includes foundations tracks (supervised/unsupervised/forecasting), API productization tracks, and production lifecycle tracks.
- `docs/`: cross-project student guides and support docs.
- `docs/index.md`: landing page for the MkDocs site.
- `docs/tracks/`: track-first learning guides (foundations, production, ranking, forecasting, responsible AI, optimization).
- `docs/api/`: API reference pages for showcase FastAPI services.
- `docs/api/assets/openapi/`: static OpenAPI JSON assets used by docs pages.
- `docs/aspect-coverage-matrix.md`: canonical mapping from requested ML aspects to showcase evidence artifacts.
- `docs/showcase-architecture.md`: track-level map of in-repo showcase architecture.
- `docs/new-showcase-playbook.md`: strict contributor checklist for adding new projects.
- `docs/requirements-mkdocs.txt`: pinned docs site dependencies.
- `.github/`: issue templates, PR template, CI workflows.
- `shared/contracts/`: JSON schemas for cross-project artifact contracts.
- `shared/config/`: project-level contract registries (including supervised artifact bootstrap commands).
- `shared/scripts/`: shared verification tooling used by root checks.
- `shared/python/ml_core/`: reusable split, EDA, leakage, and experiment helpers.
- `shared/assets/`: shared assets for future expansion.
- `shared/templates/`: reusable project templates for README, learning guides, artifacts, and Makefile patterns.
- `mkdocs.yml`: MkDocs Material site configuration and navigation.

Project details live in each project's own `README.md` and `docs/` folder.
