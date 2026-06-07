# Learning Agents Showcase — Implementation Run Ledger

Run owner: Claude Code (`/loop` dynamic mode + `claude-harness`/Workflow orchestration).
Build window: 2026-06-06 → 2026-06-07.
Project: `projects/learning-agents-showcase/` (self-contained capstone; no cross-showcase imports).

## 1) Intent

Build a self-contained teaching showcase answering **"where does learning live in an agentic
system?"** It demonstrates the *locus of learning* taxonomy across four lanes and makes an agent
*learn its orchestration policy* (route / retrieve / clarify / escalate / stop) with RL — without
retraining the LLM. Reward is driven by a judge rubric.

## 2) Locked decisions

- Scope = all four lanes (not just the core ladder).
- Totally independent showcase: vendors its own RL algorithm library and judge-rubric reward; no
  imports from sibling showcases.
- Name = `learning-agents-showcase`.
- Reward from a judge rubric (`judge_reward`), with a deliberately `hackable_reward` foil.

## 3) Process constraints honored

- Did NOT commit planning files (`docs/superpowers/specs/`, `plans/`, `docs/agents/runs/`).
- No `Co-Authored-By` trailer added to anything.
- Stayed entirely out of Codex's projects (`adaptive-course-assistant-rl-showcase`,
  `student-support-rl-showcase`, `agentic-course-assistant-showcase`).
- All edits to shared root files (`.github/workflows/ci.yml`, root `README.md`,
  `docs/aspect-coverage-matrix.md`, `.gitignore`, issue templates) were strictly ADDITIVE —
  Codex was modifying these in parallel.

## 4) Phases (each verified before the next)

- Phase 0 — Scaffold: project skeleton, `pyproject.toml`, Makefile, artifact contract plumbing.
- Phase A — Foundation: `environment.py` (assistant control loop as an MDP), `reward.py`
  (judge rubric + hackable foil), vendored RL library skeleton.
- Phase B — Lane 1 learning + governance: bandit, Q-learning, SARSA, dynamic programming (exact
  Q*), REINFORCE; `evaluation.py` reconciled to the artifact contract; reward-hacking study;
  deploy/shadow/reject governance memo.
- Phase C — Offline RL + cost cascade: `offline_rl.py` (Fitted-Q Iteration from a logged
  dataset), `ope.py` (IS / WIS / direct method / doubly robust), `cost_cascade.py`
  (effort-budget policy + cost/quality Pareto frontier).
- Phase D — Lane A: `sdk_bridge.py` — OpenAI Agents SDK as environment/executor/logger (NOT
  trainer); live SDK gated behind `make sync-sdk`; offline demo by default.
- Phase E — Lane B: `preference_optimization.py` — toy RLHF, DPO, GRPO, RLVR on a 4×5 quality
  matrix with a KL leash to a reference policy.
- Phase F — Lane C: `marl.py` — cooperative Climbing game; independent Q-learning (IQL) vs
  joint-action learning (JAL); CTDE framing.
- Phase G — Docs: 14 new guides authored in parallel and adversarially verified
  (author → skeptic → revise, 29 agents) for number-accuracy and lychee link-safety. Centerpiece
  `locus-of-learning.md` plus per-lane/method guides, glossary, math-notes, exercises, and 5
  mermaid diagrams. `docs/00-start-here.md` and root `README.md` updated so all lanes read as
  shipped (not "future work").
- Phase H — Integration + verification + ledger: confirmed CI matrix entry + clean-checkout
  `make smoke && make verify`; `.gitignore` whitelist for `manifest.json`; **fixed a latent CI
  failure** (`run_marl.py` was missing from the `smoke` target, so `make verify` would have failed
  on a clean checkout); ran a completeness-critic pass; this ledger.
- Phase I — Optional deep-RL lane: vendored NumPy DQN (neural fitted-Q with replay + target net)
  and PPO (clipped actor-critic), a `make run-drl` runner emitting the five `drl_optional`
  artifacts, 9 tests, a `docs/deep-rl.md` guide, and an additive CI step. NumPy-only (no torch);
  fulfills the previously dead `OPTIONAL_DRL_ARTIFACTS` surface and the pyproject "RL/DRL" claim.

## 5) Architecture

`environment.py` (MDP) → `reward.py` (judge rubric) → vendored RL library (one module per
method, none importing siblings) → four locus lanes → `scripts/` runners → deterministic
`artifacts/`. The artifact contract ties `reporting.py` REQUIRED_ARTIFACTS ⇄
`artifacts/manifest.json` ⇄ `scripts/verify_artifacts.py` ⇄ `tests/test_reporting.py`.
Everything is deterministic by seed.

## 6) Headline empirical results (full run)

- Policy comparison (`artifacts/eval/policy_comparison.csv`), avg reward / escalation rate:
  dp_optimal 1.2142 / 0.2833 (planning ceiling) > offline_fqi 1.2067 / 0.30 (offline FQI nearly
  matches the ceiling) > heuristic_router 1.16 / 0.0 (baseline) > q_learning 0.8525 / 0.65
  (online, under-trained, OVER-ESCALATES → governance-REJECTED) > random −1.1817 / 1.0 (floor).
- Offline RL: 1418-transition log, 196/371 states covered (0.5283), FQI Bellman residual → 0 in
  6 sweeps.
- OPE: in-support targets accurate (abs err < 0.05); off-support random target shows IS variance
  blow-up (err 0.56) cut by WIS (0.17).
- Cost cascade: non-monotonic cost — budget 4 has the best reward (1.16) AND lower cost than
  budget 3, because escalation falls to 0; Pareto-non-dominated budgets are 0, 2, 4.
- Preferences: all four methods lift quality 0.49 → ~0.999 with controlled KL (~1.6).
- MARL: IQL miscoordinates (success 0.0, team reward 5) vs JAL reaches the optimum (1.0, 11).
- Deep RL (optional, `make run-drl`): DQN recovers the DP ceiling (avg reward 1.1783, esc 0.33,
  solved 1.0); PPO settles into a safe over-escalating local optimum (0.6933) — a value-based vs
  policy-gradient contrast on a small discrete MDP (no torch; vendored NumPy networks).

## 7) Verification evidence

- `make check`: ruff clean, mypy clean (59 source files), pytest 180 passed.
- `make verify`: all required artifacts present (33-file core set; 38 with the optional DRL group).
- Clean-checkout simulation: from a bare `artifacts/` (only `manifest.json`), `make smoke` →
  exit 0, `make verify` → pass; then the optional `run_deep_rl --quick` + `verify` path passes.
- Docs: independent link check across README + 16 docs — no artifact links, no external URLs, no
  dangling cross-links; artifact references are code spans (lychee contract).
- Completeness-critic subagent (independent, read-only; executed the MARL code): found one
  CRITICAL doc-vs-artifact contradiction (Climbing-game payoff matrix mislabeled with invented
  `none`/`refuse` actions in 3 docs — traced to a wrong line in the Phase-G workflow facts block),
  two IMPORTANT, two NICE-TO-HAVE. C1 + I1 + N2 fixed this iteration; I2 confirmed by-design
  (frozen, drift-guarded, committable manifest); N1 resolved in Phase I (the deep-RL lane built).
- Deep-RL lane (Phase I): 9 unit tests; the optional `drl_optional` group validates (all-or-nothing,
  q_learning+dqn+ppo families, DQN/PPO named, policy-gradient/actor-critic/ppo/dqn term coverage).

## 8) Status

Phases 0 → I complete and verified. The showcase is runnable from a clean checkout, CI-exercised,
fully documented (17 guides, including a visual ASCII results dashboard), self-contained (no torch;
NumPy-only deep RL), and covers all four
loci of learning plus the optional deep-RL lane. The previously dead `OPTIONAL_DRL_ARTIFACTS`
surface and the pyproject "RL/DRL" claim are now fulfilled. Gate: ruff + mypy (59 files) + 180
tests; both the core and the optional clean-checkout CI paths pass.
