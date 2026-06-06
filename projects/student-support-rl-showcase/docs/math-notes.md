# Math Notes — The Equations Behind the Showcase

A compact, rigorous appendix for the RL/DRL methods this project implements. Every section names
the concept, gives the defining equation, and points at the code that implements it and the
artifact you can inspect. Notation follows Sutton & Barto: the reward received **after** taking
action `A_t` in state `S_t` is `R_{t+1}`.

> Convention: display equations are in fenced blocks so they render the same everywhere. `γ` is
> the discount, `α` a step size, `π` a policy, `θ` parameters, `δ` a TD error, `∇` a gradient.

## Notation at a glance

| Symbol | Meaning | In this repo |
|---|---|---|
| `S_t, A_t, R_{t+1}` | state, action, reward after acting | `StudentState`, action `0..3`, `default_reward` |
| `H` | horizon (episode length) | `6` weekly decisions |
| `γ ∈ [0,1]` | discount factor | `0.9` tabular, `0.95` DRL bridge |
| `π(a\|s)` / `π(s)` | stochastic / deterministic policy | `policies.py` |
| `V^π(s)` | state value under `π` | — |
| `Q^π(s,a)` | action value under `π` | `q_table`, `artifacts/q_learning/q_table.csv` |
| `Q*(s,a)` | optimal action value | `dynamic_programming.optimal_action_values` |
| `δ_t` | temporal-difference error | `q_learning.py`, `sarsa.py` |

---

## 1. MDP, return, and value functions

A Markov Decision Process is the tuple `(S, A, P, R, γ, H)`. The agent–environment loop produces a
trajectory `S_1, A_1, R_2, S_2, A_2, R_3, …`. The **(discounted) return** from step `t` is

```
G_t = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + … = Σ_{k=0}^{H-t-1} γ^k · R_{t+k+1}
```

The agent maximizes expected return. Two value functions summarize "how good" things are:

```
V^π(s)   = E_π[ G_t | S_t = s ]            (state value)
Q^π(s,a) = E_π[ G_t | S_t = s, A_t = a ]   (action value)
```

`V` and `Q` are linked by `V^π(s) = Σ_a π(a|s)·Q^π(s,a)`, and for a deterministic greedy policy
`V^π(s) = max_a Q^π(s,a)`. *Implemented by:* `environment.py` (the MDP), `evaluation.py` (estimates
returns by simulation). See [mdp-and-environment.md](mdp-and-environment.md).

## 2. Bellman equations

Value functions obey recursive consistency conditions. **Expectation** (for a fixed `π`):

```
Q^π(s,a) = E[ R_{t+1} + γ·Q^π(S_{t+1}, A_{t+1}) | S_t=s, A_t=a ]
```

**Optimality** (the best achievable values): the optimal policy `π*` satisfies

```
Q*(s,a) = E[ R_{t+1} + γ·max_{a'} Q*(S_{t+1}, a') | S_t=s, A_t=a ]
V*(s)   = max_a Q*(s,a),     π*(s) = argmax_a Q*(s,a)
```

The `max` over next actions is the difference between "value of *this* policy" and "value of the
*best* policy" — and it is precisely the seam between SARSA (§6) and Q-learning (§5).
See [value-based-learning.md](value-based-learning.md).

## 3. Dynamic programming: backward induction for exact `Q*`

When the model `P, R` is **known** and the horizon is finite, you can solve the Bellman optimality
equation exactly — no sampling. This showcase's transition is **deterministic**, so for each
reachable acting state we sweep **backward** in time:

```
Q*(s,a) = R(s,a) + ( 0           if the step terminates
                     γ·max_{a'} Q*(s', a')   otherwise )
```

Because an acting state reached at decision step `t` always has `week == t`, processing states by
descending `week` guarantees every successor `Q*(s', ·)` is already known — one ordered sweep gives
the exact fixed point (this is value iteration specialized to a finite horizon).

*Implemented by:* `dynamic_programming.optimal_action_values`. *Inspect:*
`artifacts/dp/optimal_action_values.csv` and `artifacts/dp/q_learning_gap.csv` (how far the
*learned* table sits from this ground truth). This is the **planning** rung — the optimum that the
*model-free* methods below approximate **without** ever seeing `P` or `R`.

## 4. Temporal-difference learning and the TD error

Model-free methods learn from sampled transitions. After observing `(S_t, A_t, R_{t+1}, S_{t+1})`
they form a **target** and nudge the estimate toward it. The signed surprise is the **TD error**:

```
δ_t   = target − Q(S_t, A_t)
Q(S_t, A_t) ← Q(S_t, A_t) + α·δ_t
```

The two tabular methods below differ **only** in how they build `target`. The `(target − old)`
expression in the code *is* `δ_t`.

## 5. Q-learning — off-policy TD control

Q-learning bootstraps from the **greedy** next value, regardless of what the behaviour policy
actually does next. That `max` makes it **off-policy**: it learns `Q*` while behaving
ε-greedily.

```
target = R_{t+1} + γ·max_{a'} Q(S_{t+1}, a')     (0 at a terminal step)
δ_t    = target − Q(S_t, A_t)
```

*Implemented by:* `q_learning.train_q_learning` (ε-greedy with multiplicative decay).
*Inspect:* `artifacts/q_learning/training_curve.csv`, `artifacts/q_learning/q_table.csv`, and the
gap to `Q*` in `artifacts/dp/q_learning_gap.csv`.

## 6. SARSA — on-policy TD control

SARSA bootstraps from the action `A_{t+1}` it **actually takes next** under its ε-greedy policy.
It therefore learns the value of the policy it *follows* (exploration included), which can make it
more conservative near costly mistakes (the textbook Cliff-Walking result — though whether that
shows up here depends on the reward structure and seed).

```
target = R_{t+1} + γ·Q(S_{t+1}, A_{t+1})         (0 at a terminal step)
δ_t    = target − Q(S_t, A_t)
```

*Implemented by:* `sarsa.train_sarsa`. *Inspect:* `artifacts/sarsa/training_curve.csv`.

**Off-policy vs on-policy in one line:** Q-learning's target uses `max_{a'} Q(s',a')` (value of the
*greedy* policy); SARSA's uses `Q(s', A')` for the *sampled* `A'` (value of the *behaviour* policy).

## 7. Contextual bandits — linear payoff, ridge regression, regret

A contextual bandit is the **one-step** special case: a context `x` arrives, you pick an action,
you get a reward — there is no next state to plan for. The warm-up estimates each action's expected
reward as a **linear** function of the context, fit by **ridge regression** (regularized
least squares):

```
A_a = λ·I + Σ x·xᵀ ,    b_a = Σ r·x ,    θ_a = A_a⁻¹·b_a ,    μ̂_a(x) = θ_aᵀ·x
```

Exploration is **ε-greedy**. **Honesty note:** this is *not* LinUCB — there is **no** upper-
confidence (optimism) bonus added to `μ̂_a(x)`; exploration comes only from the ε-greedy coin flip.
Performance is measured by **cumulative regret** versus the *known* best action (knowable only
because the reward model is synthetic):

```
Regret_T = Σ_{t=1}^{T} [ μ*(x_t) − μ_{a_t}(x_t) ]      where  μ*(x) = max_a μ_a(x)
```

*Implemented by:* `bandit.run_bandit_experiment`. *Inspect:* `artifacts/bandit/reward_trace.csv`,
`artifacts/bandit/regret_trace.csv`. See [exploration-and-bandits.md](exploration-and-bandits.md).

## 8. Policy gradients — optimize the policy directly

Instead of learning values and acting greedily, policy-gradient methods parameterize the policy
`π_θ` and ascend the expected-return objective `J(θ) = E_{π_θ}[G_1]`. The **policy gradient
theorem** gives an estimator that needs no model:

```
∇_θ J(θ) = E_{π_θ}[ Σ_t ∇_θ log π_θ(A_t | S_t) · (G_t − b(S_t)) ]
```

`b(S_t)` is a **baseline** that reduces variance without adding bias (here, the episode-mean
return). With a tabular **softmax** policy

```
π_θ(a|s) = exp(θ_{s,a}) / Σ_{a'} exp(θ_{s,a'})
```

the score function has a clean closed form, giving the **REINFORCE** update applied to every
visited `(s_t, A_t)` and every action `a'`:

```
∂/∂θ_{s,a'} log π_θ(A_t|s) = 1[a' = A_t] − π_θ(a'|s)
θ_{s,a'} ← θ_{s,a'} + α·(G_t − b)·( 1[a'=A_t] − π_θ(a'|s) )
```

*Implemented by:* `policy_gradient.train_reinforce`. *Inspect:*
`artifacts/policy_gradient/training_curve.csv`. See
[policy-gradient-and-actor-critic.md](policy-gradient-and-actor-critic.md).

## 9. Actor-critic and PPO

REINFORCE's baseline is the seed of a **critic**. An **actor-critic** method learns a value
estimator `V_w` (critic) alongside the policy `π_θ` (actor) and replaces the Monte-Carlo return
with the **advantage**:

```
A^π(s,a) = Q^π(s,a) − V^π(s)        (how much better a is than the state's average)
```

**PPO** is the actor-critic used in the optional bridge. It maximizes a **clipped surrogate** that
discourages destructively large policy updates, with `ρ_t(θ) = π_θ(A_t|S_t) / π_{θ_old}(A_t|S_t)`:

```
L(θ) = E[ min( ρ_t·Â_t ,  clip(ρ_t, 1−ε, 1+ε)·Â_t ) ]
```

*Implemented by (black box):* `drl.py` via Stable-Baselines3 `PPO`. *Inspect:*
`artifacts/drl_optional/policy_gradient_notes.md`. See [deep-rl.md](deep-rl.md).

## 10. Deep value-based RL: DQN

DQN keeps the Q-learning **target** of §5 but replaces the table with a neural function
approximator `Q_φ`. Two tricks stabilize the otherwise-divergent combination of bootstrapping,
function approximation, and off-policy data ("the deadly triad"):

- **Experience replay** — store transitions in a buffer and train on random minibatches, breaking
  the temporal correlation of consecutive samples.
- **Target network** — bootstrap against a slowly-updated copy `Q_{φ⁻}` so the regression target
  does not chase the weights being trained.

```
L(φ) = E_{(s,a,r,s')~buffer} [ ( r + γ·max_{a'} Q_{φ⁻}(s',a')  −  Q_φ(s,a) )² ]
```

*Implemented by (black box):* `drl.py` via Stable-Baselines3 `DQN`. *Inspect:*
`artifacts/drl_optional/rl_family_comparison.csv`. See [deep-rl.md](deep-rl.md).

## 11. Convergence and honest caveats

- **Tabular Q-learning / SARSA** converge to `Q*` / `Q^π` under standard conditions (every
  state-action visited infinitely often, suitable step sizes). In a finite run they only converge
  on the states they actually *visit* — which is exactly why `artifacts/dp/q_learning_gap.csv`
  shows large residual error on rarely-reached tail states, while `Q*` (from DP) is defined
  everywhere.
- **The bandit** uses ε-greedy exploration, not an optimism bonus, so its regret is not the
  `O(√T)` of UCB-style algorithms; it is a deliberately simple teaching baseline.
- **The evaluation** in `evaluation.py` is *simulator-based offline evaluation* (re-running each
  policy in the known environment), **not** off-policy evaluation (OPE) from logged real data.
- **Determinism** is by seed; the environment transition is deterministic given `(s, a)` and the
  reset seed only jitters the start state.

---

**See also:** [glossary.md](glossary.md) for term definitions ·
[value-based-learning.md](value-based-learning.md) ·
[policy-gradient-and-actor-critic.md](policy-gradient-and-actor-critic.md) ·
[algorithm-ladder.md](algorithm-ladder.md) for the narrative arc.
