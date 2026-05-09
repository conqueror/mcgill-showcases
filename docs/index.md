# McGill ML Showcases Documentation

Welcome to the documentation site for `mcgill-showcases`.

This repo is organized as learning-by-doing showcase projects with reproducible scripts, clear artifacts, and track-based progression paths.

## Flagship Agentic Systems Showcase

!!! tip "Agentic Course Assistant"
    Start here when you want to learn agent frameworks without making your first run depend on hosted credentials. The [Agentic Course Assistant deep dive](deep-dives/agentic-course-assistant.md) teaches routing, tools, guardrails, trace evidence, eval rubrics, and framework comparison through a deterministic offline harness first, then shows how the same design maps to optional OpenAI Agents SDK and Google ADK usage.

    - Track: [Agent Frameworks](tracks/agent-frameworks.md)
    - Project: `projects/agentic-course-assistant-showcase`
    - Default path: local, deterministic, and CI-safe with no API keys
    - Optional extension: live OpenAI or Google ADK runs after students install SDK extras and configure credentials

## Start Here

1. Run local setup from [README.md](https://github.com/conqueror/mcgill-showcases/blob/main/README.md) or [Getting Started](getting-started.md).
2. Pick a track based on your goal.
3. Run one showcase end-to-end.
4. Interpret generated artifacts with [Coverage Matrix](aspect-coverage-matrix.md).

## Learning Tracks

- Foundations: math foundations, neural network mechanics, PyTorch training loops, core supervised workflows, unsupervised workflows, EDA, and feature engineering.
- Production: serving, drift monitoring, rollout decisions, and system patterns.
- Ranking: grouped ranking modeling and API productization.
- Forecasting: time-aware demand modeling and observability-ready APIs.
- Responsible AI: fairness, explainability, and causal decision support.
- Optimization: HPO, agentic workflow, and policy optimization workflows.
- Agent Frameworks: deterministic agent workflows, course-assistant tools, guardrails, traces, eval rubrics, OpenAI Agents SDK concepts, and Google ADK concepts.

## Project Deep Dives

- [Deep dive overview](deep-dives/index.md)
- [Supervised learning deep dive](deep-dives/sota-supervised.md)
- [Causal inference deep dive](deep-dives/causal-inference.md)
- [MLOps drift deep dive](deep-dives/mlops-drift.md)
- [Ranking track deep dive](deep-dives/ranking-track.md)
- [Forecasting track deep dive](deep-dives/forecasting-track.md)
- [Agentic course assistant deep dive](deep-dives/agentic-course-assistant.md)

## Contributor Entry

- [Showcase architecture map](showcase-architecture.md)
- [New project checklist](new-showcase-playbook.md)
- [Troubleshooting](troubleshooting.md) and [FAQ](faq.md)

## API Reference

- [API overview](api/index.md)
- [Ranking API docs](api/ranking-api.md)
- [Demand API docs](api/demand-api.md)
