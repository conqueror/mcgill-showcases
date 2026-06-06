# Glossary

Plain-language definitions of the RL/DRL terms used in this showcase, each with a pointer to
where it lives in the code or artifacts. Equations are in [math-notes.md](math-notes.md); the
narrative arc is in [algorithm-ladder.md](algorithm-ladder.md).

### Action
A decision the agent can take. Here: `0` no intervention, `1` resource email, `2` TA session,
`3` advisor meeting (`ACTION_LABELS` in `environment.py`), with rising `ACTION_COSTS`.

### Action-value function `Q(s,a)`
Expected return from taking action `a` in state `s`, then following the policy. The thing tabular
Q-learning and SARSA estimate. *Artifact:* `artifacts/q_learning/q_table.csv`.

### Actor-critic
A method that learns a policy (**actor**) and a value estimator (**critic**) together, using the
critic's **advantage** to lower the variance of the policy-gradient update. PPO is the example here
(`drl.py`). See [policy-gradient-and-actor-critic.md](policy-gradient-and-actor-critic.md).

### Advantage `A(s,a)`
How much better an action is than the state's average: `A(s,a) = Q(s,a) − V(s)`. The signal an
actor-critic uses in place of the raw return.

### Agent
The learner/decision-maker. It observes a state, picks an action via its **policy**, and receives
a reward. Embodied by the `Policy` objects in `policies.py`.

### Backward induction
Solving a finite-horizon MDP exactly by computing optimal values from the last step backward.
*Code:* `dynamic_programming.optimal_action_values`. See §3 of [math-notes.md](math-notes.md).

### Baseline (policy gradient)
A state-dependent quantity subtracted from the return to reduce gradient variance without adding
bias. This repo uses the episode-mean return in `policy_gradient.train_reinforce`.

### Bellman equation
The recursive consistency condition every value function satisfies. The **optimality** form,
`Q*(s,a) = E[R_{t+1} + γ·max_a' Q*(s',a')]`, is what value-based control chases.

### Bootstrapping
Updating an estimate using other current estimates rather than a full return — the `γ·Q(s',·)`
term in TD methods. Contrast: Monte Carlo (no bootstrapping).

### Contextual bandit
A one-step decision problem: a context arrives, you act, you get a reward — no next state. The
showcase's warm-up (`bandit.py`). Teaches exploration/exploitation and **regret**.

### Discount factor `γ`
Weight on future rewards in the return (`0 ≤ γ ≤ 1`). `γ=0.9` tabular, `0.95` in the DRL bridge.

### DQN (Deep Q-Network)
Q-learning with a neural network approximating `Q`, stabilized by **experience replay** and a
**target network**. Optional bridge in `drl.py`. See [deep-rl.md](deep-rl.md).

### Environment
Everything outside the agent: it holds state, applies transitions, and emits rewards.
`StudentSupportEnvironment` in `environment.py`.

### Episode
One trajectory from reset to terminal. Here: `H=6` weekly decisions.

### Epsilon-greedy
Exploration rule: act greedily with probability `1−ε`, uniformly at random with probability `ε`.
Used by the bandit, Q-learning, and SARSA; ε **decays** over training in the TD methods.

### Experience replay
Storing transitions in a buffer and training on random minibatches, decorrelating consecutive
samples. One of DQN's two stabilizers.

### Exploration vs exploitation
The trade-off between trying actions to gather information (explore) and taking the
currently-best action (exploit). Made measurable by **regret** in the bandit.

### Function approximation
Representing `Q` or `π` with a parameterized function (e.g. a neural net) instead of a lookup
table, so the agent can generalize across states. The jump from `q_learning.py` to `drl.py`.

### Greedy policy
The deterministic policy that always takes the highest-value action, `π(s) = argmax_a Q(s,a)`.
`greedy_action` in `policies.py`; `QLearningPolicy` and `ReinforcePolicy` wrap it.

### Horizon `H`
The number of decision steps in an episode (`6` here). Finiteness is what makes exact backward
induction (§3 of [math-notes.md](math-notes.md)) cheap.

### MDP (Markov Decision Process)
The formal model `(S, A, P, R, γ, H)` of sequential decision-making under the Markov assumption
(the next state depends only on the current state and action). Defined in `environment.py`.
See [mdp-and-environment.md](mdp-and-environment.md).

### Model-based vs model-free
**Model-based** methods use a known/learned `P, R` to plan (here: dynamic programming).
**Model-free** methods learn purely from sampled experience (Q-learning, SARSA, REINFORCE).

### Monte Carlo
Estimating values from complete-episode returns rather than bootstrapping. REINFORCE uses
Monte-Carlo returns `G_t`.

### Off-policy
Learning the value of a *different* policy than the one generating data — Q-learning learns the
greedy policy's values while behaving ε-greedily (the `max` in its target).

### On-policy
Learning the value of the policy you actually follow — SARSA bootstraps from the next action it
truly takes. *Code:* `sarsa.py`. *Contrast:* see §6 of [math-notes.md](math-notes.md).

### Optimal policy `π*`
A policy that maximizes expected return from every state; `π*(s) = argmax_a Q*(s,a)`. Computed
exactly here by `dynamic_programming.py`.

### Policy `π`
The agent's decision rule: `π(a|s)` (stochastic) or `π(s)` (deterministic). The object every
method ultimately produces. `policies.py`.

### Policy gradient
Optimizing the policy parameters directly by ascending expected return, `∇_θ J = E[∇_θ log π(A|s)
·(G_t − b)]`. Implemented tabularly by `policy_gradient.train_reinforce`.

### PPO (Proximal Policy Optimization)
An actor-critic, policy-gradient method that uses a **clipped** objective to keep updates small
and stable. The showcase's actor-critic reference point (`drl.py`).

### Q-learning
Off-policy tabular TD control; learns `Q*` via the Bellman-optimality target.
`q_learning.train_q_learning`. See [value-based-learning.md](value-based-learning.md).

### Regret
Cumulative reward lost by not always playing the best action:
`Σ_t [μ*(x_t) − μ_{a_t}(x_t)]`. The bandit's score. *Artifact:* `artifacts/bandit/regret_trace.csv`.

### REINFORCE
The Monte-Carlo policy-gradient algorithm; updates `θ` using `(G_t − baseline)·∇log π`.
`policy_gradient.py`.

### Return `G_t`
The (discounted) sum of future rewards from step `t`. What the agent maximizes.

### Reward `R_{t+1}`
The scalar feedback after acting. **Not** the same as true success — see reward hacking.
`default_reward` in `environment.py`.

### Reward hacking
A policy scoring high reward via a flawed proxy rather than real success — here, over-intervening
to farm an intensity bonus under the **bad** reward. *Code:* `reward_design.py`. *Artifact:*
`artifacts/reward/reward_hacking_report.md`. See [reward-design-and-hacking.md](reward-design-and-hacking.md).

### Reward shaping
*Adding* a (ideally potential-based) term to the reward to guide learning without changing the
optimal policy — the constructive cousin of reward hacking.

### Ridge regression
Regularized least squares (`θ = (λI + XᵀX)⁻¹ Xᵀy`) used by the contextual bandit to estimate each
action's linear payoff from context. **Not** LinUCB — there is no optimism bonus.

### SARSA
On-policy tabular TD control; named for the `(S, A, R, S', A')` tuple its update uses.
`sarsa.train_sarsa`.

### State `s`
The information the agent acts on. Here a 6-tuple: week, engagement, completion, pressure, risk,
prior interventions (`StudentState` in `environment.py`).

### State-value function `V(s)`
Expected return from state `s` under the policy; `V(s) = max_a Q(s,a)` for a greedy policy.

### Target network
A slowly-updated copy of the Q-network used to compute DQN's bootstrap target, preventing the
target from chasing the weights being trained.

### TD error `δ`
The temporal-difference surprise, `δ = target − Q(s,a)`; the core update signal of Q-learning and
SARSA. The `(target − old)` term in the code.

### TD learning
Learning from incomplete episodes by bootstrapping — the family containing Q-learning and SARSA.

### Transition `P`
The environment's dynamics: how `(s, a)` produces `s'`. Deterministic here (`_transition` in
`environment.py`); the reset seed only jitters the start state.

### Value iteration
A dynamic-programming algorithm that repeatedly applies the Bellman-optimality update until values
converge; its finite-horizon form is the single backward sweep in `dynamic_programming.py`.
