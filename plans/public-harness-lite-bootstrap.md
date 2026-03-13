# Public Harness Lite Bootstrap Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap the minimum public-safe harness surface required to run the deep learning math foundations showcase work under `core-harness-flow`.

**Architecture:** Add a repo-local `.codex/` configuration, a small role routing manifest, lightweight harness docs, and local readiness/lint scripts. Keep the bootstrap compatible with canonical harness flow without importing private operational systems or private policy.

**Reasoning:** The harness artifacts are missing in this public repo, so the orchestration skill cannot admit or route the showcase implementation task. A minimal bootstrap unblocks the requested workflow while keeping the repo independent from private OODARIS infrastructure and language.

**Tech Stack:** TOML, Markdown, Bash, Python 3.11 standard library

---

### Task 1: Add repo-local harness policy and config

**Files:**
- Create: `AGENTS.md`
- Create: `.codex/config.toml`
- Create: `.codex/harness/role-skill-matrix.toml`

**Step 1: Create the policy and routing files**

**Step 2: Verify the files are present**
- Command: `test -f AGENTS.md && test -f .codex/config.toml && test -f .codex/harness/role-skill-matrix.toml`

### Task 2: Add the supported role files

**Files:**
- Create: `.codex/agents/workflow_orchestrator.toml`
- Create: `.codex/agents/requirements_clarifier.toml`
- Create: `.codex/agents/design_strategist.toml`
- Create: `.codex/agents/tracking_operator.toml`
- Create: `.codex/agents/backend_executor.toml`
- Create: `.codex/agents/quality_gate_runner.toml`
- Create: `.codex/agents/independent_critic.toml`
- Create: `.codex/agents/commit_curator.toml`

**Step 1: Create the supported role files**

**Step 2: Verify routing targets resolve**
- Command: `python3 scripts/harness_config_lint.py`

### Task 3: Add public-safe harness docs

**Files:**
- Create: `docs/agents/oodaris-harness-v2-operating-pack.md`
- Create: `docs/agents/harness-evals/README.md`
- Create: `docs/agents/runs/.gitkeep`

**Step 1: Write the public-safe operating docs**

**Step 2: Verify the docs exist**
- Command: `test -f docs/agents/oodaris-harness-v2-operating-pack.md && test -f docs/agents/harness-evals/README.md`

### Task 4: Add readiness and lint scripts

**Files:**
- Create: `scripts/dev/harness-cli-preflight.sh`
- Create: `scripts/harness_config_lint.py`

**Step 1: Write the scripts**

**Step 2: Verify the scripts run**
- Commands:
  - `bash scripts/dev/harness-cli-preflight.sh`
  - `python3 scripts/harness_config_lint.py`

### Task 5: Integrate harness-lite into root repo ergonomics

**Files:**
- Modify: `README.md`
- Modify: `Makefile`

**Step 1: Add root command documentation and helper targets**

**Step 2: Verify the helper targets work**
- Commands:
  - `make harness-preflight`
  - `make harness-lint`

### Task 6: Resume the showcase implementation under harness-lite

**Files:**
- Use: `plans/deep-learning-math-foundations-showcase.md`

**Step 1: Run harness-lite readiness checks**

**Step 2: Proceed with the showcase implementation plan**
