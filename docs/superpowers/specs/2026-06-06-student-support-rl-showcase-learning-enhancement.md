# Student Support RL Showcase — Learning-Resource Enhancement Spec & Contract

> **Status:** active · **Date:** 2026-06-06 · **Mode:** claude-harness (consolidated)
> **Predecessor:** `docs/superpowers/specs/2026-06-06-student-support-rl-showcase-design.md`
> **Audience decision (locked):** rigorous graduate DRL — full equations inline.
> **Scope decision (locked):** docs layer **+** small TDD'd teaching modules (no notebook).

This file is the **single source of truth** for the enhancement. Every subagent (generator
or reviewer) is given the relevant section verbatim so that ~20 parallel agents produce a
*consistent* result: same notation, same docstring shape, same acceptance bar.

---

## 1. Goal

Turn the **correct** `projects/student-support-rl-showcase` codebase into a **great graduate
learning resource** for RL/DRL. The code already works; we are adding the teaching layer
(docstrings, math, diagrams, guides, exercises) and a few small, inspectable teaching modules
that fill real concept gaps — without breaking behaviour, determinism, or the artifact contract.

## 2. Non-goals / hard constraints

- **No behaviour change** to existing modules except: (a) adding docstrings/comments,
  (b) the explicitly listed review-finding fixes, (c) additive artifact-contract growth.
- `make check` (ruff + mypy --strict + pytest), `make test`, and `make verify` **must stay green**.
- **Deterministic-by-seed** everywhere; no network on the core path; laptop-friendly runtimes
  (`smoke` < 60s, `run` < 10min).
- Keep the environment **tiny and inspectable**. Do **not** add MARL, RLHF, large benchmarks,
  continuous control, or GPU deps.
- New artifacts **extend** the manifest/verifier consistently (with tests); they do not remove
  or rename existing artifacts.

## 3. Style: rigorous, layered, honest

- **Lead with intuition, then give the equation.** Every core mechanism gets its defining
  equation inline (in docstrings and guides), plus a one-sentence plain-language gloss.
- **Be honest about boundaries.** Required honesty callouts (verbatim intent):
  - The contextual bandit is an **ε-greedy linear (ridge-regression) bandit**, *not* LinUCB —
    there is **no UCB optimism bonus**; exploration is ε-greedy. Regret is measured against the
    *known* optimal action because the reward model is synthetic.
  - The evaluation is **simulator-based offline evaluation** (fixed scenarios in the known
    environment), *not* true off-policy evaluation (OPE) from logged real data.
  - The DRL bridge (DQN/PPO via Stable-Baselines3) is **optional** and intentionally a
    black box; the from-scratch tabular modules are where mechanisms are taught.
- **No AI-slop tone.** Concrete, specific, no filler. (Generators should write like the
  existing docs read.)

## 4. RL notation conventions (use VERBATIM across all docstrings and guides)

Sutton & Barto convention. Reward received after taking $A_t$ in $S_t$ is $R_{t+1}$.

- State $S_t = s$; action $A_t = a$; reward $R_{t+1}$; horizon $H$ (here $H=6$).
- Discount $\gamma \in [0,1]$ (code: $\gamma=0.9$ tabular, $0.95$ DRL bridge).
- Return $G_t = \sum_{k=0}^{H-t-1} \gamma^{k} R_{t+k+1}$.
- Policy $\pi(a\mid s)$ (stochastic) or $\pi(s)$ (deterministic/greedy).
- State value $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]$.
- Action value $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$.
- Bellman optimality: $Q^*(s,a) = \mathbb{E}\!\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a') \mid S_t=s,A_t=a\right]$.
- TD error (Q-learning, off-policy): $\delta_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)$.
- TD error (SARSA, on-policy): $\delta_t = R_{t+1} + \gamma\, Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)$.
- TD update: $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\delta_t$.
- Contextual-bandit regret: $\text{Regret}_T = \sum_{t=1}^{T}\big[\mu^*(x_t) - \mu_{a_t}(x_t)\big]$,
  where $\mu_a(x)$ is the expected reward of action $a$ in context $x$ and $\mu^*$ the best.
- Ridge estimate (bandit, per action): $\hat\theta_a = A_a^{-1} b_a$, with
  $A_a = \lambda I + \sum x x^\top$, $b_a = \sum r\,x$; prediction $\hat\mu_a(x)=\hat\theta_a^\top x$.
- Softmax policy: $\pi_\theta(a\mid s) = \dfrac{e^{\theta_{s,a}}}{\sum_{a'} e^{\theta_{s,a'}}}$.
- Policy-gradient (REINFORCE): $\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\big[\sum_t \nabla_\theta \log \pi_\theta(A_t\mid S_t)\,(G_t - b(S_t))\big]$.
- Advantage: $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$.
- PPO clipped surrogate: $L(\theta) = \mathbb{E}\big[\min(\rho_t \hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\big]$, $\rho_t = \pi_\theta/\pi_{\theta_\text{old}}$.

## 5. Docstring convention (Google-style; apply to every module/class/public function)

```python
"""<imperative one-line summary>.

<2–4 sentences: what it does and WHY; the RL concept it implements and where it
sits on the ladder (bandit → MDP → Q-learning → DQN → policy gradient → actor-critic → PPO).>

Args:
    name: meaning (units/range where it matters).
Returns:
    meaning and shape/type.
Raises:
    ErrorType: when.

RL concept:
    <name the concept + cross-link, e.g. "Off-policy TD control; see
    docs/value-based-learning.md">.

Math:
    <only where a defining equation clarifies, using the §4 notation>
"""
```

- **Module docstrings**: what concept this file teaches, its ladder position, which
  artifact(s) and guide(s) it maps to.
- **Class docstrings**: the abstraction + invariants (e.g. "frozen/immutable state").
- **Private helpers** (`_name`): one line; add a `Math:`/concept note only if the helper
  *is* an algorithm step (e.g. `_epsilon_greedy_action`, `_estimated_reward`).
- **Inline comments**: only on algorithm-defining lines, naming the concept, e.g.
  `# TD target: R + γ·max_a' Q(s',a'); bootstrap term is 0 at the terminal step`.
- Keep line length ≤ 100 (ruff). Do not introduce new ruff/mypy errors.

## 6. New teaching modules (TDD; tiny, tabular, deterministic, no torch)

Each new module: new `src/student_support_rl/<name>.py` + new `tests/test_<name>.py` +
(where it fits) a runner script + artifact(s) added to the manifest/verifier.

### 6.1 `dynamic_programming.py` — exact ground-truth $Q^*$
- The transition `_transition` is **deterministic**; over $H=6$ the reachable state set is
  small. Compute exact $V^*,Q^*$ by **backward induction** (finite-horizon DP); optionally also
  expose value iteration to convergence as the infinite-horizon analogue.
- API: `optimal_action_values(...) -> dict[StateKey, list[float]]`, plus a helper that reports
  the **gap** $\max_{s\in\text{visited}}|Q_\text{learned}(s,a)-Q^*(s,a)|$ vs the trained Q-table.
- Artifact: `artifacts/dp/optimal_action_values.csv`, `artifacts/dp/q_learning_gap.csv`.
- Tests: (a) $V^*$/$Q^*$ satisfy the one-step Bellman optimality equation; (b) tabular
  Q-learning with enough episodes shrinks the gap to $Q^*$ on reachable states.
- Teaching point: the *model-based* optimum that Q-learning approximates *without a model*.

### 6.2 `sarsa.py` — on-policy TD control
- `train_sarsa(...) -> SarsaResult` mirroring `QLearningResult`, but the update uses the
  **actually-chosen** next action $A_{t+1}$ (on-policy) instead of $\max_{a'}$.
- Artifact: `artifacts/sarsa/training_curve.csv`; a short comparison note vs Q-learning.
- Tests: returns curve + q_table; demonstrate the on-policy vs off-policy difference
  (SARSA's value of the ε-greedy behaviour policy ≠ Q-learning's value of the greedy policy).

### 6.3 `policy_gradient.py` — tabular REINFORCE (softmax)
- Per-state-action logits $\theta_{s,a}$; sample episodes under $\pi_\theta$; update
  $\theta \mathrel{+}= \alpha \sum_t \nabla_\theta \log\pi_\theta(A_t|S_t)\,(G_t-b)$ with an
  optional mean-return baseline $b$.
- Artifact: `artifacts/policy_gradient/training_curve.csv` and a learned-policy snapshot.
- Tests: average return improves over training (seeded); $\sum_a \pi_\theta(a|s)=1$.
- Teaching point: optimise the **policy directly** — the idea behind PPO, made inspectable.

## 7. Guide set (new `docs/*.md`; full equations; cross-link code + artifacts)

`00-start-here.md` (reading order + repo map) · `glossary.md` · `mdp-and-environment.md` ·
`exploration-and-bandits.md` · `value-based-learning.md` (DP/$Q^*$, TD error, Q-learning, SARSA) ·
`deep-rl.md` (function approximation, **experience replay**, **target networks**, why DRL is
harder) · `policy-gradient-and-actor-critic.md` (REINFORCE → advantage → A2C → PPO clip) ·
`reward-design-and-hacking.md` (+ potential-based **shaping** vs hacking) ·
`evaluation-and-governance.md` (metrics beyond reward, OPE honesty, deploy/shadow/reject) ·
`math-notes.md` (equation appendix) · `exercises.md` (with a separate solutions section).

**Diagrams (mermaid):** agent–environment loop; MDP transition structure; the algorithm ladder.
Enrich existing `algorithm-ladder.md`, `learning-guide.md`, `method-notes.md` with cross-links;
do not duplicate content — link to the canonical guide.

## 8. Acceptance criteria (must-pass)

1. Every file in `src/`, `scripts/`, `tests/` has a **module docstring**; every class and
   public function has a docstring per §5.
2. The algorithm-defining lines carry concept-naming inline comments.
3. New modules (§6) exist, are TDD'd, and their artifacts are in the manifest + verified.
4. New guides (§7) exist, use §4 notation, and cross-link the code/artifacts they describe.
5. `make check`, `make test`, `make verify` are **green**; `make smoke` still < 60s.
6. Determinism preserved; no new network dependency on the core path.
7. Honesty callouts from §3 appear in the relevant docstrings/guides.
8. README updated with the new modules, guides, and reading order.

## 9. Review rubric (every reviewer agent scores 1–5; pass = all ≥3, avg ≥4)

- **Accuracy** — does the docstring/guide match what the code *actually does*? (off-policy vs
  on-policy, ridge vs UCB, deterministic vs stochastic, etc.)
- **Math correctness** — equations correct and in §4 notation.
- **Teaching value** — would a grad student learn the concept from this? Intuition + equation +
  pointer to the artifact to inspect.
- **Honesty** — boundaries stated where relevant (§3).
- **Consistency** — docstring shape (§5), cross-link scheme, no AI-slop.
Reviewer returns a verdict + specific required edits; a separate agent (never the original
generator) applies fixes. Orchestrator runs `make check` after each fan-out to catch breakage.

## 10. Execution order (phases)

1. **(this file)** spec/contract.
2. Document existing `src/` (parallel generate → review), then `scripts/`+`tests/`.
3. Build new modules §6 (TDD) — new files in parallel; shared-file wiring done by orchestrator.
4. Author guides §7 (parallel generate → review).
5. Integrate (scripts/Makefile/manifest/README/concept-artifacts) + fix review findings.
6. Verify (`make check/test/verify`) + `/code-review` + completeness critic + run-ledger update.
