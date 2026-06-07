# Offline RL and Off-Policy Evaluation

This is arguably the most practically important lane in the showcase. Every other
learner here is *online*: it talks to the live simulator and picks which transitions
to collect. Real agents rarely get that luxury. You are handed a frozen log of
decisions some *behavior* policy already made in production, and you must (1) learn a
better policy from that log alone and (2) estimate how good a *candidate* policy
would be **before** you dare deploy it. Part 1 is offline RL (learning from the log);
Part 2 is off-policy evaluation (scoring a policy from the log). Both live in
`src/learning_agents/offline_rl.py` and `src/learning_agents/ope.py`.

The thread tying them together is one word: **coverage**. A logged dataset is only
trustworthy where it has data; a policy that wanders into states or actions the log
never visited has *no evidence* behind it. That single constraint explains every
number below.

## Notation

- `s`, `s-prime` — a state and the next state; `a` an action, `|A|` the action-space
  size; `r` (or `R`) the reward after acting; `D` the fixed logged dataset of
  `(s, a, r, s-prime, done)` transitions; `epsilon` the exploration rate.
- `gamma` — discount factor (FQI uses `0.9`; OPE uses `1.0`, the undiscounted
  finite-horizon return, so its estimates line up with the simulator ground truth).
- `Q(s,a)` — action value (expected return from `a` in `s`); `V(s)` — state value.
- `pi(a|s)` — a policy's action probability; `pi_beta(a|s)` the **behavior** policy
  that produced the log; `pi_e(a|s)` the **target** policy we want to score.

## Part 1 — Offline RL: learning from a fixed log

Online control (Q-learning, SARSA on [the RL ladder](rl-ladder.md)) decides what to
try next. Offline RL removes that control: the dataset `D` is frozen.
`collect_logged_dataset` simulates one production log by rolling out an
**epsilon-soft** behavior policy — a deterministic base router most of the time,
with uniform random exploration probability `epsilon`:

```text
pi_beta(a|s) = epsilon / |A|  +  (1 - epsilon) * 1[a = b(s)]
```

Here `b(s)` is the base router's greedy action and `1[.]` is the indicator (1 if
true, else 0). A strictly-positive `epsilon` keeps every action's probability above
zero, so the log has *full action support* — which later makes importance-sampling
evaluation well-defined (you never divide by a zero behavior probability). The hard
part is the **out-of-distribution (OOD) action problem**: a naive learner assigns a
high value to an action it has barely seen, because the bootstrap (`max` over
next-state values) can chase optimistic estimates into regions the data does not
support. Offline RL must respect what the log actually covers.

### The real dataset

The showcase's log, summarized in `artifacts/offline_rl/dataset_summary.csv`:

| Quantity | Value |
| --- | --- |
| Logged transitions | 1418 |
| Decision states covered | 196 |
| Reachable decision states | 371 |
| Coverage fraction | 0.5283 |
| Behavior policy | `heuristic_router`, epsilon-soft |
| Exploration rate `epsilon` | 0.6 |

The log touches only **196 of 371** reachable decision states — about **53%**. Nearly
half the state space appears *only as a successor* `s-prime` and *never as a decision
point* `s`: that is the coverage gap, and the learner must not pretend to know what to
do there.

### Fitted-Q Iteration (FQI)

FQI is the offline learner (`fitted_q_iteration`). Its backup is *identical* to
Q-learning's — what makes it offline is that the `(s, a)` it updates and the
successor `s-prime` it bootstraps from come only from the frozen log. It is a batch
fixed-point iteration: each sweep recomputes a Bellman optimality target for every
logged `(s, a)` from the **current** table, then applies all updates together:

```text
Q_{k+1}(s, a) = mean over D of [ R + (0 if done else gamma * max_{a'} Q_k(s', a')) ]
```

`mean over D` averages the target across every logged transition sharing that
`(s, a)` cell (so duplicate or stochastic rewards average out); the inner term is
exactly the Bellman optimality backup, with the future term zeroed on a terminal step
(`done`). The fixed point is the optimal `Q*` **restricted to the support of `D`** —
and because the agent-decision MDP is finite-horizon and deterministic, sweeping to
convergence recovers `Q*` exactly on the covered region.

The stopping rule watches the **Bellman residual**, the largest entry change this
sweep:

```text
residual_k = max over (s,a) of | Q_{k+1}(s, a) - Q_k(s, a) |
```

From `artifacts/offline_rl/training_curve.csv`:

| Sweep | Bellman residual | Cells updated |
| --- | --- | --- |
| 1 | 2.0 | 466 |
| 2 | 1.8 | 466 |
| 3 | 1.62 | 466 |
| 4 | 0.945 | 466 |
| 5 | 0.3402 | 466 |
| 6 | 0.0 | 466 |

The residual drops monotonically to **0.0 in 6 sweeps**, each sweep refreshing the
same **466** covered state-action cells; at zero residual the table is the fixed
point. (The crisp convergence reflects the deterministic finite-horizon model; with
stochasticity or function approximation the residual would only plateau.) States
outside those 466 cells keep their all-zeros initialization — a greedy policy falls
back to a safe default there, having no logged evidence to do anything smarter. That
honesty about the coverage gap is the point of doing this offline.

### Why offline FQI nearly matches the planning ceiling

The headline result, from `artifacts/eval/policy_comparison.csv`
(avg_reward / escalation_rate / avg_steps / solved_rate):

| Policy | avg_reward | escalation | avg_steps | solved | Note |
| --- | --- | --- | --- | --- | --- |
| `dp_optimal` | 1.2142 | 0.2833 | 2.05 | 1.0 | Planning ceiling: exact `Q*` by backward induction |
| `offline_fqi` | 1.2067 | 0.30 | 2.0 | 1.0 | Offline FQI from the heuristic log |
| `heuristic_router` | 1.16 | 0.0 | 3.0667 | 1.0 | The behavior policy's base router |
| `q_learning` | 0.8525 | 0.65 | 1.2167 | 1.0 | Online tabular, 400 episodes; over-escalates |
| `random` | -1.1817 | 1.0 | 3.0 | 0.5333 | Floor |

Offline FQI scores **1.2067**, a whisker under the dynamic-programming ceiling of
**1.2142** (exact backward induction in `src/learning_agents/dynamic_programming.py`)
and well above online `q_learning` at **0.8525**.

**Why does it nearly match the ceiling?** Because *the data is good*. The behavior
policy is the `heuristic_router` — a competent baseline (1.16 alone) — softened with
exploration, so the log concentrates its 1418 transitions on the sensible, high-value
states where an optimal policy actually spends its time. FQI computes the exact
Bellman fixed point on that well-covered region, recovering near-optimal behavior with
zero new interaction, and even shaves a step off the base router (2.0 vs 3.0667
average steps) by stopping earlier where the router over-deliberated.

**Why does it beat online `q_learning`?** The online learner was trained for only 400
episodes on purpose; tabular Q-learning needs roughly 5000 episodes to converge to
`Q*` here, so 400 leaves a real residual gap. Worse, `q_learning` **over-escalates**
(0.65 vs the optimum's 0.2833) and is **governance-REJECTED** in
[evaluation and governance](evaluation-and-governance.md). Offline FQI inherits the
discipline of a good behavior log *and* computes an exact fixed point on it. The
lesson is not "offline always wins" — it is that *the quality of your data source can
dominate the choice of algorithm*. The flip side: this near-optimality is
*conditional on coverage*. With 53% of states covered, the policy is trustworthy on
the covered region and defaults elsewhere; if production drifts into the uncovered
47%, the log no longer vouches for it — exactly the problem Part 2 helps you detect.

## Part 2 — Off-policy evaluation (OPE)

You have a candidate target `pi_e` and want its value *before* deploying it. The
simulator comparison in
[evaluation and governance](evaluation-and-governance.md) gets this by re-running each
policy in the environment — honest for a teaching simulator, but exactly what you
cannot do in production. OPE estimates `E[return | pi_e]` from the **behavior log
only**. `src/learning_agents/ope.py` implements the four canonical episodic
estimators, and because the showcase has a simulator it also computes each policy's
true value (`true_policy_value`) so you can grade accuracy and see *why* some
estimators blow up.

The engine of the importance-based estimators is the **per-step ratio** `rho_t`,
which is `0` whenever the behavior took an action the target would not have. The
**trajectory weight** `w` chains these, and `G` is the return:

```text
rho_t = 1[a_t = pi_e(s_t)] / pi_beta(a_t | s_t)
w = product over t of rho_t        (trajectory importance weight)
G = sum over t of gamma^t * r_t     (trajectory return)
```

Because `pi_e` is deterministic, `w` collapses to `0` as soon as the behavior log
diverges from the target *once* — only trajectories the behavior followed end-to-end
survive, which is the source of IS's variance.

### The four estimators

**Importance sampling (IS).** Reweight each trajectory's return and average:

```text
V_IS = mean over n of ( w_n * G_n )
```

IS is **unbiased** but, for a deterministic target, most trajectories contribute `0`
and a few survivors carry all the mass — so it has **high variance**.

**Weighted IS (WIS).** Self-normalize by the total weight instead of the count:

```text
V_WIS = ( sum over n of w_n * G_n ) / ( sum over n of w_n )
```

WIS is **biased but consistent**, with **far lower variance** — usually the better
IS-family choice.

**Direct method (DM) — Fitted-Q Evaluation.** Fit `Q^pi` for the *target* by
Fitted-Q Evaluation (like FQI, but the bootstrap uses the target's next action, not
the `max` — policy evaluation, not control), then read the value off the start states
(the empirical initial-state distribution):

```text
Q^pi_{k+1}(s, a) = mean over D of [ R + (0 if done else gamma * Q^pi_k(s', pi_e(s'))) ]
V_DM = mean over start states s0 of  Q^pi(s0, pi_e(s0))
```

DM has **low variance** but is **biased by model error and coverage**.

**Doubly robust (DR).** Anchor on the direct method's `V^pi(s0)` and add an
importance-weighted correction for the model's per-step Bellman error:

```text
V_DR = mean over n of [ V^pi(s0)
        + sum over t of gamma^t * (product_{i<=t} rho_i)
          * ( r_t + gamma * V^pi(s_{t+1}) - Q^pi(s_t, a_t) ) ]
```

with `V^pi(s) = Q^pi(s, pi_e(s))`. DR is **doubly robust**: unbiased if *either* the
weights or the fitted `Q^pi` is correct, with variance between IS and DM. When the
weight collapses to `0`, the correction vanishes and DR **falls back to the
direct-method value** — the property that keeps it sane on thin coverage.

### Results: estimators graded against ground truth

From `artifacts/ope/estimator_comparison.csv`. Each cell is the estimate, with the
absolute error against the true value in parentheses.

| Target (true value) | IS | WIS | DM | DR |
| --- | --- | --- | --- | --- |
| `heuristic_router` (1.179) | 1.1541 (0.0249) | 1.1361 (0.0429) | 1.1507 (0.0283) | 1.1507 (0.0283) |
| `dp_optimal` (1.219) | 1.1775 (0.0415) | 1.1858 (0.0332) | 1.192 (0.027) | 1.192 (0.027) |
| `random` (-1.074) | -0.5126 (0.5614) | -0.905 (0.169) | -0.5597 (0.5143) | -0.7201 (0.3539) |

### The key lesson: overlap is everything

**In-support targets are easy.** The `heuristic_router` *is* the behavior base, and
`dp_optimal` mostly agrees with it on the covered high-value states. The log overlaps
both well, so **every estimator is accurate** — every absolute error is below **0.05**
and the choice of estimator barely matters.

**The off-support target breaks naive estimators.** The `random` target has almost
no overlap with the heuristic log: the actions a uniform-random policy would take
are rarely the actions the competent behavior policy logged. So:

- **IS variance explodes** — error **0.5614**. With few matching trajectories, the
  surviving handful are unrepresentative and the unbiased-in-expectation estimate is
  wildly off on this finite sample: textbook IS failure under poor overlap.
- **WIS slashes it** — error **0.169**, under a third of IS's. Self-normalizing by
  the realized weight mass tames the variance, exactly when WIS earns its keep.
- **DM is also badly biased** (error 0.5143): the fitted `Q^pi` extrapolates past
  coverage into the random target's poorly-supported actions.
- **DR sits between** (error 0.3539): it beats raw IS and DM, but on this severe
  coverage failure neither component is reliable, so it cannot fully rescue an
  estimate the data does not support.

The governance takeaway is blunt: **OPE is trustworthy exactly to the extent the
behavior log overlaps the target.** Prefer WIS or DR over raw IS, and treat any
candidate whose actions stray far from the logged support as *unverifiable from this
log* rather than confidently scored. The `random` row demonstrates that boundary —
not buggy estimators, but the impossibility of conjuring evidence never logged.

### How this connects back

Note that the same `Q^pi` fitting powers both lanes — policy *evaluation* (bootstrap
from `pi_e`'s action) in OPE's direct method, policy *control* (bootstrap from the
`max`) in offline FQI — and the Part 1 coverage fraction (0.5283) is what decides
whether the Part 2 estimates can be trusted. Together they are the offline toolkit
for shipping agent policies responsibly: learn from the log, then prove the candidate
safe *from the log* before it touches production.

## See also

- [The RL ladder](rl-ladder.md) — online Q-learning and the planning-vs-learning
  gap.
- [Evaluation and governance](evaluation-and-governance.md) — the policy-comparison
  table, the simulator ground truth, and why `q_learning` is rejected.
- [Math notes](math-notes.md) — the Bellman backups, importance ratios, and DR
  derivation in full.
- [Glossary](glossary.md) — coverage, importance sampling, and the OPE estimator
  family in brief.
