# Mathematical Notes

This is the rigorous-math home for the showcase. Every algorithm elsewhere in the
project is, underneath, one of the equations below. The concept guides build the
intuition; this page consolidates the precise statements. Each equation lives in a
fenced `text` block, every symbol is defined, and a one-line note says what it *means*.
The forms here are verified against the named modules in `src/learning_agents/` -- they
are the forms the code actually computes, not generic textbook forms. Read each block as
"left-hand side *is* right-hand side"; most of RL is a few fixed-point and gradient
identities reused at different scales.

## Symbol legend

Recurring symbols; per-section symbols are defined where they appear. `s` and `s-prime`:
a state and the next state (here a discrete 7-tuple of the agent's situation). `a`,
`a-prime`: an action and next action (answer / retrieve / clarify / escalate). `r`: a
scalar reward, with `R_{t+1}` the reward received *after* acting at time `t`. `pi(a|s)`:
a policy (a point mass for a deterministic one). `Q(s,a)`, `V(s)`: action value and state
value; `Q*`, `V*` are the optimal ones. `gamma`: discount in `[0,1]`. `alpha`: step size.
`epsilon`: exploration rate. `A`: advantage (how much better than a baseline). `beta`:
KL-penalty weight. `KL`: Kullback-Leibler divergence. `E[.]`: expectation; `1[.]`: the
indicator (1 if true, else 0).

## The MDP tuple

Everything starts from a Markov decision process. The "Markov" part is the key
assumption: the next state and reward depend only on the current state and action, not
on the full history.

```text
MDP = (S, A, P, R, gamma)
  S       set of states
  A       set of actions
  P(s'|s,a)   transition kernel: probability of landing in s' after a in s
  R(s,a,s')   reward function for that transition
  gamma   discount factor in [0,1]
```

Meaning: the five things you need to fully specify a sequential decision problem. In
this showcase the environment (`src/learning_agents/environment.py`) is the MDP; its
transitions are deterministic, so `P` collapses to a function `s' = T(s,a)`. The reward
`R` is a judge rubric over answer quality, grounding, cost, and escalation.

## Discounted return and the role of gamma

The agent does not maximize the next reward; it maximizes the *return*, the discounted
sum of all future rewards from time `t` onward.

```text
G_t = sum over k>=0 of gamma^k * R_{t+k+1}
    = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...
```

- `G_t` -- the return from time `t`.
- `gamma` -- discount; `gamma=0` is myopic (only the next reward), `gamma -> 1`
  weights the far future almost as much as the present.

Meaning: `gamma` trades off immediate vs long-term reward and keeps an infinite sum
finite. This exact `G_t` is what Q-learning, SARSA, backward induction, and REINFORCE all
estimate (`src/learning_agents/q_learning.py`, `src/learning_agents/policy_gradient.py`).
The showcase uses `gamma = 0.9` for control and planning; off-policy *evaluation* uses
`gamma = 1.0` (undiscounted finite-horizon return) so its numbers line up with the
simulator's reported episodic return.

## Bellman expectation equations

The return is recursive: the value of a state is the immediate reward plus the
discounted value of where you land. Writing that out for a *fixed* policy `pi` gives the
Bellman expectation equations.

```text
V^pi(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * V^pi(s') ]
Q^pi(s,a) =        sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * V^pi(s') ]
with V^pi(s) = sum_a pi(a|s) * Q^pi(s,a)
```

- `V^pi`, `Q^pi` -- the value and action-value of following policy `pi`.

Meaning: these define the value of a *given* policy as a fixed point. The direct method
in off-policy evaluation (`fitted_q_evaluation` in `src/learning_agents/ope.py`)
literally solves the `Q^pi` equation by repeated sweeps, bootstrapping from the target
policy's own next action.

## Bellman optimality equations

To find the *best* policy, replace "average over what `pi` does" with "take the best
action". That self-referential equation is the Bellman optimality equation, and its
unique solution is `Q*`.

```text
V*(s)   = max_a sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * V*(s') ]
Q*(s,a) =       sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * max_a' Q*(s',a') ]
       = E[ R_{t+1} + gamma * max_a' Q*(s',a') ]
with the optimal greedy policy  pi*(s) = argmax_a Q*(s,a)
```

- `max_a'` -- the greedy bootstrap: assume the best next action is taken.

Meaning: the fixed point every value-based method chases. Q-learning's update targets
exactly this; backward induction solves it exactly. Note `V*(s) = max_a Q*(s,a)`. The
expectation form on the last line is the literal target quoted in the docstring of
`train_q_learning` in `src/learning_agents/q_learning.py`.

## TD(0) error and update

Q-learning and SARSA both learn by *temporal-difference* learning: form a one-step
estimate of the return (the TD target), subtract the current estimate to get the TD
error, and step toward it. TD(0) is the one-step version.

```text
target = R_{t+1} + gamma * <bootstrap value of s'>   (0 if s' is terminal)
delta  = target - Q(s, A)                            (the TD error)
Q(s, A) <- Q(s, A) + alpha * delta                   (the update)
```

- `delta` -- the TD error, the surprise: how wrong the current value was.
- `alpha` -- step size; how far to move toward the target.

Meaning: bootstrapping -- learning a guess from a guess -- lets TD learn online, before
an episode ends. The only thing distinguishing the algorithms below is what goes in
`<bootstrap value of s'>`.

## Q-learning vs SARSA, side by side

Both are TD(0) control. The difference is one symbol in the bootstrap, and it is the
whole story of off-policy vs on-policy.

```text
Q-learning (off-policy):  target = R_{t+1} + gamma * max_a' Q(s', a')
SARSA      (on-policy):   target = R_{t+1} + gamma *        Q(s', A')
shared update:            Q(s, A) <- Q(s, A) + alpha * (target - Q(s, A))
```

- `A'` -- the action the behavior policy *actually* takes next (e.g. an epsilon-greedy
  draw).
- `max_a'` -- the value of the *greedy* next action, whether or not it is taken.

Meaning: Q-learning bootstraps from the greedy action, so its target is `Q*` regardless
of how exploratory the data was -- that decoupling of behavior from target is what
"off-policy" means. SARSA bootstraps from the action it will really take, so it learns the
value of the policy it is actually running, exploration included. The `max_a' Q(s', a')`
line is exactly `future_value = max(q_table[next_key])` in `train_q_learning`
(`src/learning_agents/q_learning.py`); SARSA swaps that one line for the sampled next
action's value. A measured consequence: trained for only 400 episodes, online tabular
`q_learning` reaches `avg_reward` 0.8525 with a 0.65 escalation rate -- it over-escalates
and is governance-**rejected** (`artifacts/eval/policy_comparison.csv`). The math is
correct; the sample budget is deliberately too small, which is itself the lesson.

## Value / backward induction for exact Q*

When the MDP is *known* (you have the rules), you do not need to sample -- you can solve
the Bellman optimality equation directly. For a finite-horizon deterministic MDP this is
backward induction: a single ordered sweep from the last step backward.

```text
deterministic transition  s' = T(s, a):
  Q*(s, a) = R_{t+1} + (0 if done else gamma * max_a' Q*(s', a'))
  V*(s)    = max_a Q*(s, a)
process states in descending step t = H, H-1, ..., 0
```

- `H` -- the horizon (episode length); `T(s,a)` -- the deterministic next state.

Meaning: because every state reached at decision step `t` has `step == t` and terminal
states are never acted on, each successor's `Q*` is already known when you need it, so
one sweep gives the exact fixed point -- no iteration to convergence required. This is
value iteration specialized to a finite horizon, implemented in `optimal_action_values`
(`src/learning_agents/dynamic_programming.py`). At convergence the TD error is exactly 0
for every `(s,a)`. The resulting plan, `dp_optimal`, is the **planning ceiling**:
`avg_reward` 1.2142, the best of any policy in `artifacts/eval/policy_comparison.csv`.

Planning vs learning, made measurable: online Q-learning converges to this exact `Q*`
only around 5000 episodes; the showcase trains 400 on purpose, so the residual gap in
`artifacts/dp/q_learning_gap.csv` (compared against `artifacts/dp/optimal_action_values.csv`)
*is* the demonstration that sampling-based learning approaches, but has not yet reached,
the model-based optimum.

## The FQI fixed point (offline / batch RL)

Offline RL learns from a *fixed log* of transitions with no new interaction. Fitted-Q
Iteration applies the Bellman optimality backup repeatedly over the static dataset `D`
until the table stops changing.

```text
Q_{k+1}(s, a) = mean over D of [ R + (0 if done else gamma * max_a' Q_k(s', a')) ]
Bellman residual = || Q_{k+1} - Q_k ||_inf   (drives the stopping rule)
```

- `D` -- the logged dataset of `(s, a, r, s', done)` transitions.
- `mean over D` -- the empirical Bellman backup, averaged over rows with that `(s,a)`.
- `||.||_inf` -- the max absolute change across the table.

Meaning: the same greedy backup as Q-learning, batched over a frozen log instead of
streamed from the environment (`fitted_q_iteration` in
`src/learning_agents/offline_rl.py`). On a deterministic finite-horizon MDP it recovers
`Q*` *restricted to the states the log covers*. Here the log holds 1418 transitions
covering 196 of 371 decision states (coverage fraction 0.5283), gathered by the heuristic
router made epsilon-soft with `epsilon = 0.6` (`artifacts/offline_rl/dataset_summary.csv`).
The residual falls 2.0 -> 1.8 -> 1.62 -> 0.945 -> 0.3402 -> 0.0, converging in 6 sweeps
(`artifacts/offline_rl/training_curve.csv`). The payoff: offline `offline_fqi` reaches
`avg_reward` 1.2067 -- beating the rejected online learner and *nearly matching* the DP
ceiling of 1.2142, learning a safe policy from logs alone.

## Off-policy evaluation: IS, WIS, direct method, doubly robust

Before deploying a new policy you often want to estimate its value from logs your
*current* policy already produced, without ever running the new one. That is off-policy
evaluation. Four estimators, in increasing sophistication, all in
`src/learning_agents/ope.py`. First the importance ratio that reweights logged data:

```text
per-step ratio:    rho_t = pi_e(a_t | s_t) / pi_beta(a_t | s_t)
trajectory weight: w     = product over t of rho_t
trajectory return: G     = sum over t of gamma^t * r_t
```

- `pi_e` -- the *target* policy being evaluated (deterministic here, so `pi_e(a|s)` is 1
  or 0).
- `pi_beta(a|s)` -- the *behavior* policy's probability of the logged action (stored as
  `behavior_action_prob` in the log).

Then the four estimators:

```text
IS  (importance sampling):  V_IS  = mean_n ( w_n * G_n )
WIS (weighted IS):          V_WIS = sum_n ( w_n * G_n ) / sum_n ( w_n )
DM  (direct method):        V_DM  = mean over start states of Q^pi(s_0, pi_e(s_0))
DR  (doubly robust):
  V_DR = mean_n [ V^pi(s_0) + sum_t gamma^t (product_{i<=t} rho_i)
                  ( r_t + gamma*V^pi(s_{t+1}) - Q^pi(s_t, a_t) ) ]
  with V^pi(s) = Q^pi(s, pi_e(s))
```

- `Q^pi`, `V^pi` -- the target policy's value, fit from the log by Fitted-Q Evaluation
  (the `Q^pi` recursion of the Bellman expectation section).
- `n` -- index over logged trajectories.

The bias/variance trade-off: **IS** is unbiased but high variance -- for a deterministic
target a trajectory survives only if behavior matched the target at *every* step (else
`w = 0`). **WIS** self-normalizes by the total weight: biased but consistent and far
lower variance, usually the better IS-family choice. **DM** reads value off a fitted
model: low variance, but only as good as the model and the log's coverage. **DR** anchors
on the DM value and adds an IS correction for the model's per-step Bellman error -- it is
*doubly robust* (unbiased if *either* the weights or the `Q^pi`-model is right), with
variance between IS and DM, and falls back to the DM anchor when a weight collapses to 0.

The numbers make the trade-off concrete (`artifacts/ope/estimator_comparison.csv`, as
`estimate (abs_error)` vs the simulator-measured true value). For an in-support target,
`heuristic_router` (true 1.179): IS 1.1541 (0.0249), WIS 1.1361 (0.0429), DM 1.1507
(0.0283), DR 1.1507 (0.0283) -- every estimator is accurate (error < 0.05). For
`dp_optimal` (true 1.219) the same holds: IS 1.1775 (0.0415), WIS 1.1858 (0.0332), DM
1.192 (0.027), DR 1.192 (0.027). But for the off-support target `random` (true -1.074),
which has poor overlap with the heuristic behavior log, IS variance **explodes**:
-0.5126 (0.5614). WIS slashes the error to -0.905 (0.169); DM gives -0.5597 (0.5143) and
DR -0.7201 (0.3539). Lesson: estimator choice only matters when coverage is thin, and
that is exactly when it matters most.

## The policy-gradient theorem and REINFORCE

Value-based methods learn `Q` and act greedily. Policy-gradient methods skip that and
optimize a parameterized policy `pi_theta` *directly* by gradient ascent on expected
return. Using a tabular softmax policy keeps it transparent:

```text
softmax policy:  pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'})
objective:       J(theta) = E[ G_t ]
policy gradient: grad_theta J = E[ grad_theta log pi(A_t|s_t) * (G_t - b) ]
softmax score:   d/dtheta_{s,a'} log pi(A_t|s_t) = 1[a'=A_t] - pi(a'|s_t)
REINFORCE step:  theta_{s,a'} <- theta_{s,a'} + alpha * (G_t - b) * (1[a'=A_t] - pi(a'|s_t))
```

- `theta_{s,a}` -- the logit (parameter) for action `a` in state `s`.
- `b` -- a baseline subtracted from the return to cut variance; here the episode-mean
  return.
- `(G_t - b)` -- the advantage `A` for that step: how much better the realized return was
  than the baseline.

Meaning: the score-function estimator pushes probability mass toward actions that
preceded above-baseline returns. The baseline does not bias the gradient (zero expected
contribution) but sharply cuts its variance -- it is the seed of the "critic" that
actor-critic methods learn. This update is `train_reinforce` in
`src/learning_agents/policy_gradient.py` line for line, and the same estimator PPO scales
up behind a neural network.

## Deep RL: DQN regression and PPO clipping

The optional deep-RL lane swaps the table for a neural network on the *same* MDP. Both methods
reduce to the equations above, fit by gradient descent instead of tabular updates.

DQN is neural Q-learning: regress `Q_theta(s, a)` onto the Bellman target, using a replay buffer
and a frozen target network `Q_bar` for stability.

```text
target:      y       = r + gamma * (1 - done) * max_a' Q_bar(s', a')
loss:        L(theta) = ( Q_theta(s, a) - y )^2     (only for the taken action a)
update:      one SGD step on L over a replay minibatch; sync Q_bar <- Q_theta periodically
```

PPO is the actor-critic, clipped descendant of REINFORCE. It keeps the score-function gradient but
(1) replaces the episode-mean baseline with a learned critic `V_phi`, and (2) clips the probability
ratio so one update cannot move the policy too far.

```text
ratio:       rho  = pi_new(a|s) / pi_old(a|s)
advantage:   A    = G - V_phi(s)          (then normalized across the batch)
actor loss:  L_pi = - E[ min( rho * A , clip(rho, 1 - eps, 1 + eps) * A ) ] - c_ent * H(pi)
critic loss: L_V  = E[ ( V_phi(s) - G )^2 ]
```

- `Q_bar` -- the target network, a periodic copy of `Q_theta` so the regression target is stable.
- `rho` -- how much more (or less) likely the new policy makes the taken action than the policy that
  collected the data; `eps` is the clip width (0.2 here).
- `H(pi)` -- the policy entropy; `c_ent` a small bonus that keeps exploration alive.

Meaning: DQN bootstraps a value function (and on this small MDP recovers the dynamic-programming
ceiling `Q*`); PPO ascends the same policy gradient as REINFORCE but takes safe, clipped steps
against a critic baseline. Both live in `src/learning_agents/deep_rl.py` (`train_dqn`, `train_ppo`).
See [deep RL: DQN and PPO](deep-rl.md).

## The Bradley-Terry model

The preference-tuning lane learns the LLM's weights from *preferences* rather than a
hand-written reward. Step one of classic RLHF turns relative preferences ("response `c`
beats response `l`") into an absolute reward by fitting a Bradley-Terry model.

```text
P(c preferred over l) = sigmoid( r_phi(c) - r_phi(l) )
reward-model loss     = - log sigmoid( r_phi(c) - r_phi(l) )   (over preferred pairs c > l)
sigmoid(x) = 1 / (1 + exp(-x))
```

- `r_phi(.)` -- the learned reward (a scalar score per response), parameters `phi`.
- `c`, `l` -- the chosen and rejected responses in a preference pair.

Meaning: a logistic model of pairwise preference. Minimizing the loss makes the chosen
response score above the rejected one, converting *relative* judgements into an
*absolute* reward the policy can be optimized against (`train_reward_model` in
`src/learning_agents/preference_optimization.py`).

## The RLHF objective (reward minus beta times KL)

Step two of RLHF improves the policy against the learned reward, but with a leash: a KL
penalty back to the reference (pretrained) policy so tuning does not collapse the
distribution onto the reward's argmax.

```text
max_theta  E_{r ~ pi_theta}[ r_phi(r) ]  -  beta * KL( pi_theta || pi_ref )
KL( pi || pi_ref ) = sum_r pi(r) * log( pi(r) / pi_ref(r) )
folded per-response reward:  r_phi(r) - beta * ( log pi_theta(r) - log pi_ref(r) )
```

- `pi_ref` -- the reference (pretrained) policy the tuned policy must stay near.
- `beta` -- the KL-penalty weight: the leash strength. Larger `beta` keeps `pi_theta`
  closer to `pi_ref`.
- `KL` -- how far the tuned policy has drifted from the reference.

Meaning: maximize reward *and* stay close to the pretrained behavior. In practice (and
in `train_rlhf`) the KL term is folded into a per-response reward and optimized by the
same policy gradient as REINFORCE. Without the KL leash the policy over-fits the reward
model and drifts off the pretrained manifold -- a direct route to reward hacking.

## The DPO loss

Direct Preference Optimization is "RLHF without the reward model". It optimizes the
policy directly on the same preference pairs with a logistic loss that is provably
equivalent to the RLHF objective under a Bradley-Terry assumption -- no reward model, no
sampling, no RL loop.

```text
DPO loss = - log sigmoid( beta * [ ( log pi_theta(c) - log pi_ref(c) )
                                 - ( log pi_theta(l) - log pi_ref(l) ) ] )
```

- `c`, `l` -- chosen and rejected responses; `pi_ref` -- the reference policy.
- `beta` -- the implicit KL strength (a temperature on the log-ratio margin).

Meaning: the implicit reward is the log-ratio `log( pi_theta / pi_ref )`. Pushing the
chosen response's log-ratio above the rejected one's reproduces the RLHF optimum without
ever fitting `r_phi`. For the tabular softmax the per-logit gradient is strikingly
simple: the softmax terms cancel, so each step nudges the chosen logit up and the
rejected logit down by `beta * (1 - sigmoid(margin))` (`train_dpo` in
`src/learning_agents/preference_optimization.py`).

## The GRPO group-relative advantage

Group-Relative Policy Optimization, the method behind recent reasoning models, drops the
critic. Instead of a learned value baseline it samples a *group* of responses per prompt
and uses the group's own statistics as the baseline.

```text
for a sampled group {r_1, ..., r_G}:
  A_i = ( reward(r_i) - mean_j reward(r_j) ) / ( std_j reward(r_j) + eps )
  theta_{p,a} <- theta_{p,a} + lr * mean_i [ A_i * ( 1[a = r_i] - pi(a|p) ) ]
```

- `G` -- the group size (responses sampled per prompt); `eps` -- a small constant for
  numerical safety.
- `A_i` -- the group-relative advantage: how far response `i` beats its own group's
  average, scaled by the group's spread.

Meaning: responses that beat their group's mean are made more likely, below-average ones
less likely -- with no critic and no separate reward model, which is what makes it cheap
and stable at scale (`train_grpo` in `src/learning_agents/preference_optimization.py`).
RLVR (RL from Verifiable Rewards) is exactly GRPO with the graded `reward(.)` replaced by
a binary correctness check, so a sparse pass/fail signal still yields a gradient.

All four methods clear the same bar on the toy task
(`artifacts/preference/method_comparison.csv`, as `expected_quality` /
`win_rate_vs_reference` / `kl_to_reference`): from a `reference` of 0.49 / 0.5 / 0.0,
`rlhf` reaches 0.9994 / 0.8996 / 1.5995, `dpo` 0.9995 / 0.8997 / 1.5986, `grpo` 0.9988 /
0.8991 / 1.593, and `rlvr` 0.9988 / 0.899 / 1.5927 -- each lifts expected quality from
0.49 to about 0.999 with a controlled KL of about 1.6. A caveat in the honest spirit of
this page: this is a toy 4x5 quality matrix, **not** a real language model -- the
mechanisms are real, the scale is not.

## A short note on Goodhart and reward hacking

> When a measure becomes a target, it ceases to be a good measure. (Goodhart's law)

Every reward in this project is a *proxy* for what we actually want. An agent maximizes
the number you wrote down, not the intent behind it, so any gap between the proxy and the
goal is an exploit waiting to be found. This is not hypothetical here: trained on the
judge rubric, online `q_learning` discovered that escalating ends episodes on safe terms
and learned to over-escalate (0.65 escalation rate), inflating its proxy reward while
degrading the behavior we wanted -- which is why governance **rejected** it
(`artifacts/eval/policy_comparison.csv`).

Two defenses appear in the math above. The **KL leash** (`beta * KL` in RLHF/DPO) caps
how far a policy can chase a flawed reward away from sane behavior. **Verifiable rewards**
(RLVR) replace a gameable learned reward model with a programmatic correctness check that
cannot be talked into a high score -- at the cost of needing a verifier at all. Neither
removes the underlying tension; both bound it. Treat every reported reward as a proxy and
read the *behavior* columns (escalation rate, steps) alongside it.

## See also

- [The RL ladder](rl-ladder.md) -- bandit to PPO, where each equation here sits.
- [Offline RL and OPE](offline-rl-and-ope.md) -- the FQI fixed point and the four
  estimators in depth.
- [Lane B: preference optimization](lane-b-preference-optimization.md) -- Bradley-Terry,
  RLHF, DPO, GRPO end to end.
- [Reward design and hacking](reward-design-and-hacking.md) -- the Goodhart tensions and
  how the rubric is built.
