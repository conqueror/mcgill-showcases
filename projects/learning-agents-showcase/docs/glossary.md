# Glossary

An alphabetical reference for the vocabulary used across this showcase. Each entry gives a one- or
two-line working definition in the sense this project uses the term, plus a cross-link to the doc
that teaches it in depth. Symbols follow the project convention: `gamma` (discount factor),
`Q(s,a)` (action value), `V(s)` (state value), `pi(a|s)` (policy), `alpha` (step size), `epsilon`
(exploration rate), `r` (reward), `s` and `s-prime` (current and next state), `A` (advantage),
`beta` (KL weight), and `KL` (Kullback-Leibler divergence).

The recurring concrete object is the agent's **orchestration policy**: a small rule that chooses, at
each step, among `answer_direct`, `retrieve`, `clarify`, and `escalate`. We learn that policy, not
the language model's weights. See [start here](00-start-here.md) and
[locus of learning](locus-of-learning.md).

## A

**Action value (Q)** — `Q(s,a)` is the expected discounted return from taking action `a` in state
`s` and then following a fixed policy thereafter. The control methods on the ladder learn or compute
`Q`; the greedy policy picks `argmax_a Q(s,a)`. See [the RL ladder](rl-ladder.md).

**Advantage** — `A(s,a) = Q(s,a) - V(s)`: how much better action `a` is than the policy's average
behaviour in state `s`. Used as a lower-variance learning signal in policy-gradient and actor-critic
methods. See [the RL ladder](rl-ladder.md) and [math notes](math-notes.md).

## B

**Bellman equation** — the self-consistency identity for values. For action values the optimal form
is `Q*(s,a) = r + gamma * max_a' Q*(s-prime, a')`. Every value-based method here is a way of solving
or approximating this fixed point. See [math notes](math-notes.md).

**Bradley-Terry model** — a probabilistic model of pairwise preferences: the chance that item `i`
beats item `j` is `sigmoid(score_i - score_j)`. It is the statistical backbone of preference
learning from comparison data, including RLHF reward models and DPO. See
[preference optimization](lane-b-preference-optimization.md).

## C

**Contextual bandit** — a one-step decision problem: observe a context (the request features),
choose one action, receive a reward, with no state transition. The simplest rung on the ladder and
the warm-up for full RL. See [exploration and bandits](exploration-and-bandits.md).

**Coverage** — how much of the relevant state-action space a logged dataset actually visits. In
`artifacts/offline_rl/dataset_summary.csv` the behavior log covers 196 of 371 decision states
(`coverage_fraction` 0.5283). Coverage bounds what any offline method can reliably learn or
evaluate. See [offline RL and OPE](offline-rl-and-ope.md).

**CTDE (centralised training, decentralised execution)** — a multi-agent recipe: share information
during training so agents learn to coordinate, then act on local observations at run time. The joint
learner (JAL) here is the centralised-training extreme. See [multi-agent RL](lane-c-marl.md).

## D

**Direct method (DM)** — an off-policy evaluation estimator that fits a `Q` model for the target
policy from the log and reads off its predicted value; it is model-based and low-variance but
biased if the model is wrong. For in-support targets it is accurate (e.g. `dp_optimal` abs error
0.027 in `artifacts/ope/estimator_comparison.csv`). See [offline RL and OPE](offline-rl-and-ope.md).

**Discount factor (gamma)** — `gamma` in `[0,1)` weights future rewards: the return is
`G_t = sum_k gamma^k * r_{t+k+1}`. Smaller `gamma` makes the agent more myopic. See
[math notes](math-notes.md).

**DPO (Direct Preference Optimization)** — preference learning that skips the separate reward model
and optimizes the policy directly on preferred-vs-rejected pairs with an implicit KL anchor to a
reference. In the toy study it reaches expected quality 0.9995 at `kl_to_reference` 1.5986
(`artifacts/preference/method_comparison.csv`). See
[preference optimization](lane-b-preference-optimization.md).

**Doubly robust (DR)** — an OPE estimator that anchors on the direct method's model value and adds
an importance-sampling correction, so it is accurate if *either* the model or the behavior-policy
estimate is good. On the off-support `random` target it cuts IS error from 0.5614 to 0.3539
(`artifacts/ope/estimator_comparison.csv`). See [offline RL and OPE](offline-rl-and-ope.md).

**Dynamic programming (DP)** — solving a *known* MDP exactly by sweeping the Bellman equations.
Here `dynamic_programming.py` uses backward induction to get exact `Q*` for the finite-horizon
decision MDP; it is the planning ceiling (`dp_optimal`, avg reward 1.2142) the learners are measured
against. See [the RL ladder](rl-ladder.md).

## E

**Epsilon-greedy** — an exploration rule: with probability `epsilon` take a random action, else act
greedily. Used by the bandit and Q-learning; the behavior policy for the offline log is the
heuristic router made `epsilon`-soft with `epsilon=0.6`. See
[exploration and bandits](exploration-and-bandits.md).

**Escalation** — the `escalate` action (action 3), which hands the request to a human. It is the
safe fallback but the most expensive action (cost 1.5), so a good policy escalates only when needed.
Over-escalation is exactly why online `q_learning` (escalation rate 0.65) is governance-rejected.
See [evaluation and governance](evaluation-and-governance.md).

## F

**Fitted-Q Iteration (FQI)** — an offline, batch value-learning method: repeatedly regress `Q`
toward the Bellman target using only logged transitions. Here it converges in 6 sweeps (Bellman
residual 2.0 to 0.0 in `artifacts/offline_rl/training_curve.csv`) and the resulting `offline_fqi`
policy (avg reward 1.2067) nearly matches the DP ceiling. See
[offline RL and OPE](offline-rl-and-ope.md).

## G

**GRPO (Group Relative Policy Optimization)** — a policy-gradient preference method that normalizes
rewards within a group of sampled responses instead of using a learned value baseline. In the toy
study it reaches expected quality 0.9988 (`artifacts/preference/method_comparison.csv`). See
[preference optimization](lane-b-preference-optimization.md).

## I

**Importance sampling (IS)** — an OPE estimator that reweights each logged trajectory's return by
the ratio of target-to-behavior action probabilities. Unbiased but high-variance: on the off-support
`random` target its error explodes to 0.5614 (`artifacts/ope/estimator_comparison.csv`). See
[offline RL and OPE](offline-rl-and-ope.md).

**Independent Q-learning (IQL)** — a multi-agent baseline where each agent runs its own Q-learning
and treats the others as part of the environment. In the cooperative Climbing game it miscoordinates
to a safe, suboptimal joint action (`skim+brief`, team reward 5.0 vs optimal 11.0) with coordination
success 0.0 (`artifacts/marl/coordination_comparison.csv`). See [multi-agent RL](lane-c-marl.md).

## J

**Joint-action learning (JAL)** — a centralised multi-agent learner that learns values over the
*joint* action space, so it can reach a coordinated optimum. In the Climbing game it attains
`deep_research+detailed`, team reward 11.0, coordination success 1.0
(`artifacts/marl/coordination_comparison.csv`). See [multi-agent RL](lane-c-marl.md).

**Judge reward** — the reward source for the decision MDP: a rubric (`reward.py`) that scores the
committed action on answer quality, grounding, cost, and safety, standing in for an LLM-as-judge or
learned reward model. Contrasted with a deliberately hackable proxy. See
[reward design and hacking](reward-design-and-hacking.md).

## K

**KL divergence** — `KL(p || q)` measures how far distribution `p` is from `q`; it is asymmetric
and non-negative. Preference methods add `beta * KL(pi || pi_ref)` to keep the tuned policy near a
reference. All four toy methods land near `KL` 1.6 to the reference
(`artifacts/preference/method_comparison.csv`). See
[preference optimization](lane-b-preference-optimization.md).

## L

**Locus of learning** — the question of *what object is actually being trained*. Here it is the
orchestration policy around the agent (which tool, when to escalate, when to stop), not the language
model's weights. This is the showcase's central framing. See
[locus of learning](locus-of-learning.md).

## M

**MARL (multi-agent reinforcement learning)** — RL with more than one learning agent, where each
agent's environment is non-stationary because the others are also learning. See
[multi-agent RL](lane-c-marl.md).

**MDP (Markov Decision Process)** — the formal model `(S, A, P, r, gamma)`: states, actions,
transition dynamics `P`, reward `r`, and discount `gamma`, with the Markov property that the next
state depends only on the current state and action. The decision MDP here has a 5-step horizon and a
hard cost budget. See [showcase architecture](showcase-architecture.md) and
[math notes](math-notes.md).

## O

**Off-policy** — learning about one (target) policy from data generated by a different (behavior)
policy. Q-learning is off-policy because it bootstraps from `max_a' Q(s-prime, a')` regardless of
which action was actually taken. See [the RL ladder](rl-ladder.md).

**Off-policy evaluation (OPE)** — estimating a target policy's value from a fixed log *without*
deploying it. The four estimators compared here are IS, WIS, DM, and DR; for in-support targets all
are accurate (abs error < 0.05), and they diverge under poor overlap
(`artifacts/ope/estimator_comparison.csv`). See [offline RL and OPE](offline-rl-and-ope.md).

**Offline RL** — learning a policy purely from a fixed dataset of logged transitions, with no
further interaction with the environment. Fitted-Q Iteration is the offline learner here. See
[offline RL and OPE](offline-rl-and-ope.md).

**On-policy** — learning about the same policy that is generating the data. SARSA is on-policy: it
bootstraps from the action actually taken next, not the greedy max. See [the RL ladder](rl-ladder.md).

**Orchestration policy** — the learned decision rule that drives the agent: at each step it maps the
request state to one of `answer_direct`, `retrieve`, `clarify`, or `escalate`. This is the policy
every method on the ladder is trying to learn or beat. See
[locus of learning](locus-of-learning.md) and [showcase architecture](showcase-architecture.md).

## P

**Pareto frontier** — the set of options not dominated on every objective at once. In the cost
cascade, balancing total cost against reward, budgets 0, 2, and 4 are non-dominated while 1 and 3
are dominated; budget 4 has the best reward (1.16) *and* lower total cost (1.76) than budget 3
(1.8767), because escalation drops to 0 (`artifacts/cost_cascade/cost_quality_curve.csv`). See
[cost-aware cascade](cost-aware-cascade.md).

**Policy gradient** — directly optimizing a parameterized policy `pi(a|s)` by ascending the gradient
of expected return, instead of learning values first. The family includes REINFORCE and actor-critic
methods. See [the RL ladder](rl-ladder.md).

**Policy (pi)** — `pi(a|s)` is the rule mapping states to a distribution over actions. A
deterministic greedy policy is the special case that puts all mass on `argmax_a Q(s,a)`.

## Q

**Q-learning** — off-policy tabular TD control that updates `Q(s,a)` toward the Bellman optimality
target `r + gamma * max_a' Q(s-prime, a')`. Trained online for 400 episodes here, it over-escalates
(rate 0.65, avg reward 0.8525) and is governance-rejected; with ~5000 episodes it would approach the
DP ceiling. See [the RL ladder](rl-ladder.md).

## R

**Regret** — the cumulative gap between the reward an agent earned and the reward the best fixed
action would have earned. The bandit warm-up tracks regret as its learning signal. See
[exploration and bandits](exploration-and-bandits.md).

**REINFORCE** — the basic Monte-Carlo policy-gradient algorithm: scale the gradient of
`log pi(a|s)` by the realized return `G_t` (optionally minus a baseline). The tabular softmax
version is the policy-gradient rung on the ladder. See [the RL ladder](rl-ladder.md).

**Reward hacking** — when a policy scores well on a misspecified proxy reward while failing the true
objective. The showcase contrasts the aligned judge rubric with a hackable twin that overpays for
escalation and raw evidence, so a degenerate "always escalate" policy looks good. See
[reward design and hacking](reward-design-and-hacking.md).

**RLHF (Reinforcement Learning from Human Feedback)** — the pipeline that fits a reward model from
human preference comparisons (often Bradley-Terry) and then optimizes the policy against it with a
KL anchor to a reference. The toy version lifts expected quality from 0.49 to 0.9994 at `KL` 1.5995
(`artifacts/preference/method_comparison.csv`). See
[preference optimization](lane-b-preference-optimization.md).

**RLVR (Reinforcement Learning from Verifiable Rewards)** — preference/quality optimization driven
by an automatic checkable signal (a verifier) rather than human labels. In the toy study it reaches
expected quality 0.9988 (`artifacts/preference/method_comparison.csv`). See
[preference optimization](lane-b-preference-optimization.md).

## S

**SARSA** — on-policy TD control whose update target is `r + gamma * Q(s-prime, a-prime)`, using the
action `a-prime` actually taken next. The on-policy contrast to Q-learning on the same environment.
See [the RL ladder](rl-ladder.md).

**Shadow deployment** — running a candidate policy alongside production with human review and no
automated actioning, to gather evidence before trusting it. One of the three governance verdicts
(deploy / shadow / reject) in `artifacts/business/deploy_shadow_reject_memo.md`. See
[evaluation and governance](evaluation-and-governance.md).

## T

**TD error** — the temporal-difference prediction error `delta = r + gamma * Q(s-prime, a') -
Q(s,a)`: the gap between the bootstrapped target and the current estimate. It is the core update
signal for Q-learning and SARSA. See [the RL ladder](rl-ladder.md) and [math notes](math-notes.md).

## V

**Value iteration** — a dynamic-programming method that repeatedly applies the Bellman optimality
backup until values converge to `V*` (equivalently `Q*`). It is the planning counterpart to
learning; here backward induction plays this role for the finite-horizon MDP. See
[the RL ladder](rl-ladder.md) and [math notes](math-notes.md).

## W

**Weighted importance sampling (WIS)** — IS with self-normalized weights (divide by the sum of
weights instead of the trajectory count). It trades a small bias for much lower variance: on the
off-support `random` target it slashes the IS error from 0.5614 to 0.169
(`artifacts/ope/estimator_comparison.csv`). See [offline RL and OPE](offline-rl-and-ope.md).

## See also

- [Start here](00-start-here.md) — the showcase map and what is being learned.
- [The RL ladder](rl-ladder.md) — bandit to Q-learning to SARSA to policy gradients in context.
- [Offline RL and OPE](offline-rl-and-ope.md) — where most OPE terms above are taught.
- [Math notes](math-notes.md) — full derivations for the equations referenced here.
