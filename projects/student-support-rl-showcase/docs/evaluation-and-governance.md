# Offline Evaluation and Governance

## 1. Intuition

Every other guide in this showcase produces a *policy* — a contextual-bandit rule, a tabular
Q-table, a SARSA table, a REINFORCE softmax, or an optional deep DQN/PPO network. This guide is
about the discipline that comes *after* any rung of the ladder (contextual bandit → MDP →
Q-learning → DQN → policy gradient → actor-critic → PPO): before you trust a learned controller
to touch a real student, you have to *judge* it. The naive instinct is to rank policies by their
average return and ship the winner. That is exactly the trap. A reward signal is a *proxy* for
what we actually want; a policy can rack up high `total_reward` while behaving badly — escalating
trivial cases to a human advisor, spamming interventions at an already-supported student, or
quietly coasting a high-risk student to the end of term. Offline evaluation here means
re-simulating each candidate policy across a *fixed* bank of scenarios and reporting a **vector**
of behavioural metrics, not one scalar — so reward hacking and unsafe shortcuts are visible
side-by-side with the objective. Governance is the layer on top: a written, machine-checkable
rule that turns that metric vector into a **deploy / shadow / reject** decision, plus the safety
controls (synthetic data, human-reviewed escalation, mandatory offline eval) that wrap the whole
loop.

> **Honesty, stated up front (read this before the numbers).** This is *simulator-based offline
> evaluation*: we have white-box access to `StudentSupportEnvironment` and we simply **re-run**
> each policy inside that known model. It is **not** off-policy evaluation (OPE). True OPE
> estimates a target policy's value from a *fixed log of trajectories collected by some other
> behaviour policy* — via importance sampling, doubly-robust estimators, or fitted-Q evaluation —
> *without ever running the target*. We do the opposite: we execute the target in the model. That
> is honest, reproducible, and cheap, but it inherits every bias of the simulator. A policy that
> looks good here is only guaranteed good *in this model*, never on real students. Treat these
> numbers as a **comparison harness, not a deployment certificate.** (Same caveat as
> [math-notes.md](math-notes.md) §11.)

## 2. The core mechanism

### 2.1 What a rollout estimates

For each `(policy, scenario)` pair the harness rolls out one episode in a fresh environment and
accumulates the finite-horizon return. In the notation of [math-notes.md](math-notes.md) (reward
after acting is `R_{t+1}`; horizon `H = 6`):

```
G_t = Σ_{k=0}^{H-t-1} γ^k · R_{t+k+1}
```

The evaluation harness sums the **undiscounted** finite-horizon reward (`γ = 1` in the rollout
loop — discounting belongs to the *learner*, not the scorer). Averaging over scenarios and seeds
turns a handful of noisy rollouts into a Monte-Carlo estimate of the policy's value and typical
behaviour under the scenario distribution:

```
v̂(π) = (1/N) Σ_{i=1}^{N} G^{(i)}        (empirical mean return over N episodes)
```

Seeds are derived deterministically as `base_seed + episode_index`, so the whole comparison is
reproducible. The same averaging is applied to every side-metric below — see
[math-notes.md §1](math-notes.md) for `V`/`Q` and the return, and
[mdp-and-environment.md](mdp-and-environment.md) for the MDP the rollout walks.

### 2.2 Why reward alone is insufficient: the metric vector

For every rollout the harness records — alongside `total_reward` — a set of behavioural probes,
each one designed to expose a specific failure mode that a scalar return would hide. From
`evaluation.py` (`evaluate_policies`):

| Metric | What it measures | What a bad value reveals |
|---|---|---|
| `total_reward` / `avg_reward` | the optimization objective `G_t` | nothing on its own — the proxy |
| `final_risk` / `avg_final_risk` | the student's risk level at the terminal week | a high-risk student left un-de-risked |
| `intervention_cost` / `avg_intervention_cost` | Σ of `ACTION_COSTS[a]` over the episode | a policy buying reward with expensive help |
| `over_intervention_count` | steps where `a ≠ 0` **and** `prior_interventions ≥ 2` | spamming help at an already-supported student |
| `escalation_count` | steps where `a == 3` (advisor meeting) | over-using the scarce human resource |
| `unsafe_or_questionable_decisions` | steps that did nothing at max risk **or** escalated at minimal risk | clearly wrong calls in either direction |
| `solved` / `solved_rate` | fraction of episodes ending at `risk ≤ 1` | the actual outcome we care about |

The two safety/hacking probes are worth stating precisely, because the governance rule keys on
them. From the rollout loop:

```
over_intervention_count          += 1[ a ≠ 0  and  s.prior_interventions ≥ 2 ]
unsafe_or_questionable_decisions += 1[ (a == 0 and s.risk ≥ 3)  or  (a == 3 and s.risk ≤ 1) ]
solved                            = 1[ final_state.risk ≤ 1 ]
```

The `unsafe_or_questionable` probe is two-sided: doing *nothing* while the student is at maximal
risk (`risk == 3`) is dangerous; *escalating to a human advisor* when the student is at minimal
risk (`risk ≤ 1`) wastes a costly, human-in-the-loop resource. Both are decisions a reviewer
would question, so both are counted. The over-intervention probe is the direct numerical analogue
of the reward-hacking story in [reward-design-and-hacking.md](reward-design-and-hacking.md): a
policy that wins reward by piling on interventions shows up here as a large
`over_intervention_count`, even when `avg_reward` looks fine.

### 2.3 The deploy / shadow / reject rule

Governance turns the metric vector into one decision. `reporting.recommendation_from_summary`
compares the **`q_learning`** row against the **`heuristic`** row (the incumbent baseline) and
applies a two-gate rule. With `unsafe_rate = avg_unsafe_or_questionable_decisions`:

```
# Gate 1 — safety first, before any reward comparison:
if  unsafe_rate > 0.5  OR  avg_final_risk > 1.6:
        decision = reject        # too much unsafe / residual-risk exposure

# Gate 2 — beat the incumbent within the safety bound:
elif  avg_reward > baseline_reward  AND  unsafe_rate <= 0.5:
        decision = shadow        # better than the heuristic, but guarded rollout only

# Otherwise — safe but not clearly better:
else:
        decision = reject        # insufficient margin over the heuristic
```

The structure is deliberate. **Safety is checked first**, so a policy can never buy its way past
the safety gate with a high return — this is precisely what stops the harness from accepting a
reward-hacking policy that inflated `avg_reward` by over-intervening. Only a policy that is *both*
safe enough *and* strictly better than the incumbent heuristic earns a `shadow` (guarded,
non-actioning) rollout; everything else is rejected. `deploy` is intentionally **not** an
automatic output of this rule — in a teaching repo with simulator-only evidence, the most a
policy earns is `shadow`. If either the `q_learning` or `heuristic` row is missing, the function
fails closed to `reject` ("missing comparable evidence"). See
[reward-design-and-hacking.md](reward-design-and-hacking.md) for why the unsafe-rate gate is the
load-bearing guard.

### 2.4 Where governance sits in the pipeline

```mermaid
flowchart LR
    train["Train a policy (any ladder rung)"] --> eval["evaluate_policies: re-simulate on fixed scenarios"]
    eval --> vec["Metric vector: reward, final_risk, cost, over_intervention, escalation, unsafe, solved_rate"]
    vec --> gate["recommendation_from_summary: two-gate rule"]
    gate -->|"unsafe_rate > 0.5 or final_risk > 1.6"| reject["reject"]
    gate -->|"beats heuristic and safe"| shadow["shadow (guarded rollout)"]
    gate -->|"safe but no margin"| reject
    shadow --> controls["Safety controls: synthetic data, human-reviewed escalation"]
```

## 3. In this showcase

Open these in order; each line says what to look for.

- **`src/student_support_rl/evaluation.py`** — the harness itself.
  - `evaluate_policies(...)` runs the triple loop (policy × scenario × seeded episode), builds the
    per-episode `scenario_rows`, and aggregates them into per-policy `summary_rows`. Read the
    inner loop (the `over_intervention_count` / `unsafe_or_questionable_decisions` accumulators) to
    see each probe computed from `(state, action)` before the transition.
  - `simulate_episode(...)` traces a single rollout step-by-step `(s, a, R_{t+1}, s')` — use it to
    answer *why* a policy scored the way it did, not just *what* it scored.
  - `_summarize_scenarios(...)` is the Monte-Carlo averaging step; note it ranks by `avg_reward`
    descending, which is the *leaderboard* order, **not** the governance order (the gate can reject
    the top-of-leaderboard policy).
- **`src/student_support_rl/reporting.py`** — the governance/evidence layer.
  - `recommendation_from_summary(...)` is the deploy/shadow/reject rule in §2.3.
  - `governance_artifacts()` renders the three narrative memos (safety controls, offline-eval plan,
    business memo). **Honest note:** the `business_memo` *string returned by this function* is a
    static placeholder that says "shadow first"; the runner (`scripts/run_showcase.py` /
    `scripts/write_business_memo.py`) **overwrites** it with the *live* verdict from
    `recommendation_from_summary`, so trust the file on disk, not the stub.
  - `REQUIRED_ARTIFACTS`, `missing_required_artifacts`, `artifact_validation_errors`, and the
    per-path `required_columns` table are the *artifact contract* — the machine-checkable promise
    that every run emits well-formed evidence. This is governance applied to the *evidence itself*,
    enforced by `scripts/verify_artifacts.py` and `tests/test_artifact_contract.py`.
- **`artifacts/eval/policy_comparison.csv`** — the per-policy summary leaderboard. Look at the
  `q_learning` row's `avg_unsafe_or_questionable_decisions` and `avg_final_risk`: those two numbers
  are exactly what the governance gate reads.
- **`artifacts/eval/scenario_results.csv`** — one row per `(policy, scenario, episode)` with the
  full `actions` trace. This is where you *audit* a decision sequence — e.g. spot the `random`
  policy escalating to `advisor_meeting` and then idling.
- **`artifacts/governance/safety_controls.md`** — the three controls: synthetic students only;
  advisor escalation stays human-reviewed in any real deployment; offline eval runs before any
  shadow/live rollout.
- **`artifacts/governance/offline_eval_plan.md`** — the procedure: hold scenarios fixed across
  policies; compare reward *and* final risk *and* intervention volume; reject any policy that
  improves reward only by over-intervening.
- **`artifacts/business/deploy_shadow_reject_memo.md`** — the live verdict for the checked-in run.
  In the current artifacts it reads **`reject`**, because `q_learning`'s
  `avg_unsafe_or_questionable_decisions ≈ 0.67 > 0.5` trips Gate 1 ("too much safety or
  residual-risk exposure"). That a model topping the reward leaderboard is still *rejected* by the
  safety gate is the whole point of this guide.

### Worked example (current checked-in numbers)

From `artifacts/eval/policy_comparison.csv`:

| policy | avg_reward | avg_final_risk | avg_unsafe_or_questionable | solved_rate |
|---|---|---|---|---|
| `q_learning` | 0.22 | 0.60 | 0.6667 | 0.8667 |
| `heuristic` | −2.1067 | 0.7333 | 0.0 | 0.80 |
| `random` | −3.2333 | 0.6667 | 1.2667 | 0.80 |

`q_learning` wins on `avg_reward` (0.22 vs −2.11) and on `solved_rate` (0.87 vs 0.80) — it would
top a reward-only leaderboard. But Gate 1 fires: `0.6667 > 0.5`, so the verdict is **reject**, not
shadow. (Note also `heuristic`'s `avg_unsafe_or_questionable = 0.0` — the hand-written baseline is
*safer*, just lower-reward. That is the kind of trade-off a scalar return would have buried.)

## 4. Honest caveats

- **This is simulator-based offline evaluation, not OPE.** Restating §1 because it is the single
  most important caveat: every metric comes from *re-running* the policy in the known model. There
  is no logged real-world data, no importance weighting, no distribution-shift correction. Good
  numbers here transfer to reality only as far as the simulator is faithful — and the simulator's
  transition is a hand-crafted deterministic teaching model (see
  [mdp-and-environment.md](mdp-and-environment.md)).
- **The thresholds (`0.5`, `1.6`) are pedagogical, not calibrated.** They are round numbers chosen
  to make the gate *demonstrably bite* on this environment, not values derived from a clinical
  risk budget. In a real deployment they would be set with domain experts against real harm costs.
- **`risk` is a heuristic, not a learned risk model.** Every metric built on `risk` (`final_risk`,
  `solved`, the `unsafe` probe) inherits that. `risk_from_metrics` is a hand-tuned scoring rule in
  `environment.py`, not a calibrated predictor.
- **The probes are hand-defined.** `unsafe_or_questionable_decisions` encodes *our* notion of a
  questionable call (idle-at-max-risk, escalate-at-min-risk). A different stakeholder might draw
  the line elsewhere; the harness is honest about *which* line it draws, but the line is a choice.
- **Determinism is by seed, and the transition is deterministic.** The only stochasticity is the
  start-state jitter from `reset(seed=...)`; the dynamics never sample. So `episodes_per_scenario`
  averages over *start-state* noise only, not transition noise — there is none.
- **`deploy` never appears automatically.** The rule emits only `shadow` or `reject`. This is a
  deliberate safety posture for a simulator-evidenced teaching repo, not a limitation to "fix."

## 5. See also

- [reward-design-and-hacking.md](reward-design-and-hacking.md) — *why* a single reward number is
  insufficient, and the over-intervention loophole the unsafe-rate gate is built to catch.
- [mdp-and-environment.md](mdp-and-environment.md) — the MDP, the deterministic transition, and the
  `risk` heuristic every metric depends on.
- [value-based-learning.md](value-based-learning.md) — the `q_learning` policy this guide
  evaluates and gates.
- [deep-rl.md](deep-rl.md) and
  [policy-gradient-and-actor-critic.md](policy-gradient-and-actor-critic.md) — the optional DQN/PPO
  policies that flow through the same harness (`artifacts/drl_optional/`).
- [exercises.md](exercises.md) — practice problems, including re-deriving the gate's verdict.
- [glossary.md](glossary.md) — definitions (off-policy, regret, reward hacking, return).
- [math-notes.md](math-notes.md) — full notation and the return/value derivations (§1, §11).
