# McGill ML Showcases

Public, student-friendly machine learning showcase projects for learning by doing.

This repository contains multiple tutorial-style projects with reproducible tooling (`uv` + `make`), clear learning flows, and practical artifacts.

[![CI](https://github.com/conqueror/mcgill-showcases/actions/workflows/ci.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/ci.yml)
[![Markdown Links](https://github.com/conqueror/mcgill-showcases/actions/workflows/markdown-links.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/markdown-links.yml)
[![Notebook Smoke](https://github.com/conqueror/mcgill-showcases/actions/workflows/notebooks-smoke.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/notebooks-smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents
- [Start Here](#start-here)
- [Project Catalog](#project-catalog)
- [Repository Commands](#repository-commands)
- [Learning Path](#learning-path)
- [How to Get Help](#how-to-get-help)
- [Contributing](#contributing)
- [License](#license)

## Start Here
1. Install Python 3.11+ and `uv`.
2. Run:
```bash
make sync
```
3. Pick one project from the catalog below.
4. Enter that project and follow its `README.md`.

If this is your first time, start with `sota-supervised-learning-showcase` because it uses built-in datasets and no API keys.

## Project Catalog

| Project | Topic | Difficulty | Estimated Time | Prerequisites | Start Link |
|---|---|---|---|---|---|
| `causalml-kaggle-showcase` | Causal inference, uplift modeling, policy simulation | Intermediate | 2-3 hours | Python, basic ML, Kaggle token | [`projects/causalml-kaggle-showcase/README.md`](projects/causalml-kaggle-showcase/README.md) |
| `sota-supervised-learning-showcase` | Supervised learning foundations + SOTA-style evaluation | Beginner-Intermediate | 1.5-2.5 hours | Python, basic classification/regression | [`projects/sota-supervised-learning-showcase/README.md`](projects/sota-supervised-learning-showcase/README.md) |
| `sota-unsupervised-semisup-showcase` | Unsupervised, semi-supervised, self-supervised, active learning | Intermediate | 2-3 hours | Python, basic ML intuition | [`projects/sota-unsupervised-semisup-showcase/README.md`](projects/sota-unsupervised-semisup-showcase/README.md) |

## Repository Commands

Use root commands to run quality gates across all projects:

```bash
make help
make sync
make lint
make ty
make test
make check
make verify
```

Project-specific runs (pipelines, notebooks, domain scripts) should be started from each project folder.

## Learning Path
- Beginner path: supervised showcase -> unsupervised/semisup showcase -> causal showcase.
- Applied/business path: causal showcase -> supervised showcase -> unsupervised/semisup showcase.
- See detailed guidance in `docs/learning-path.md`.

## How to Get Help
- Read `docs/faq.md` and `docs/troubleshooting.md` first.
- Ask learning questions using GitHub Issues template: "Learning Question".
- Open bug reports with reproducible steps and command output.

## Contributing
See `CONTRIBUTING.md` for setup, standards, and pull request workflow.

## License
MIT License. See `LICENSE`.
