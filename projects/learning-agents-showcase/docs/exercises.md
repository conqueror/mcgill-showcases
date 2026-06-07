# Exercises (with Solutions)

These exercises span the whole showcase: the tabular RL ladder, offline RL and
off-policy evaluation (OPE), the cost-aware cascade, reward design, governance,
and the three lanes (agent frameworks, preference optimization, multi-agent RL).
Each states a prompt, then a marked solution that uses the canonical numbers and
names the artifact or module to check against. Try to answer before reading on.

Notation: `gamma` is the discount, `Q(s,a)` an action value, `V(s)` a state
value, `pi(a|s)` a policy, `alpha` a step size, `epsilon` an exploration rate,
`r` a reward, `s` and `s-prime` the current and next state, `A` an advantage,
`beta` a KL weight, and `KL` the Kullback-Leibler divergence. The shared
environment is a student-support router: each step the agent answers at some
effort tier or escalates to a human, and reward trades quality against action
cost while penalizing unsafe shortcuts. The same world drives the offline and OPE
sections.

## Exercise 1 — Which policy does governance reject?

Prompt. Governance is a hard gate, not a score: a policy ships only if it solves
nearly every case AND does not over-escalate (escalation is the most expensive,
most human-intensive action). Using `artifacts/eval/policy_comparison.csv`, name
the policy governance rejects, and justify the call with its `escalation_rate`
and `avg_reward`. Then say which non-ceiling policy you would ship instead.

Solution. Governance rejects `q_learning`. In
`artifacts/eval/policy_comparison.csv` it posts `avg_escalation_rate` 0.65 —
roughly two of every three steps — while its `avg_reward` is only 0.8525, far
below the other solved policies. It does reach `solved_rate` 1.0, so it is not
failing the task; it fails the cost-and-trust constraint by reflexively kicking
work to humans. Contrast `heuristic_router`: `avg_reward` 1.16, `solved_rate`
1.0, `escalation_rate` 0.0. It never escalates and clears the gate, so it is the
policy you ship among non-ceiling options. (`random` fails harder — see Exercise
2 — but the catch here is that a high `solved_rate` does not rescue a policy that
games the cheapest-looking action.) The gate logic lives in
`src/learning_agents/evaluation.py`; see [evaluation and
governance](evaluation-and-governance.md).

## Exercise 2 — Read the floor

Prompt. The `random` policy is the floor baseline. From
`artifacts/eval/policy_comparison.csv`, what is its `avg_reward` and
`solved_rate`, and which two failure modes does it exhibit that the other
policies do not? Why does a random agent get an `escalation_rate` of exactly
1.0?

Solution. `random` has `avg_reward` -1.1817 and `solved_rate` 0.5333 — it
resolves barely over half of cases. It is the only policy with nonzero
`avg_unsafe_or_questionable_decisions` (0.4667) and nonzero `avg_over_effort_count`
(0.6333): it both takes unsafe shortcuts and burns effort it did not need. The
`escalation_rate` of 1.0 is an averaging artifact of the metric, not "escalate
every step": escalation is a terminal action, so almost every random trajectory
eventually stumbles into it before solving the case, and the rate is computed per
episode-that-escalated. The teaching point is that the floor is genuinely unsafe,
which is why governance (Exercise 1) must screen on safety counters and not just
reward. See [the RL ladder](rl-ladder.md).

## Exercise 3 — The planning ceiling vs. the best learner

Prompt. Two policies nearly tie at the top. Using
`artifacts/eval/policy_comparison.csv`, compare `dp_optimal` and `offline_fqi`
on `avg_reward`, `escalation_rate`, and `solved_rate`. Which is the true ceiling,
which is learned, and how close does the learner get?

Solution. `dp_optimal` is the planning ceiling: it computes the exact optimal
action values `Q*(s,a)` by backward induction over the finite-horizon MDP
(`src/learning_agents/dynamic_programming.py`), so no policy in this world can
beat it. It scores `avg_reward` 1.2142, `escalation_rate` 0.2833,
`solved_rate` 1.0. `offline_fqi` is learned — Fitted-Q Iteration trained only on
a logged dataset — yet it reaches `avg_reward` 1.2067, `escalation_rate` 0.30,
`solved_rate` 1.0. The reward gap to the ceiling is 1.2142 − 1.2067 = 0.0075,
under one percent. The lesson: with a decent log, a purely offline method can
land essentially on the planning optimum, and unlike `q_learning` it does not
over-escalate. See [offline RL and OPE](offline-rl-and-ope.md).

## Exercise 4 — Why does offline FQI beat online Q-learning here?

Prompt. Both are value-based and both can in principle reach `Q*`. So why does
offline `offline_fqi` (`avg_reward` 1.2067) crush online `q_learning`
(`avg_reward` 0.8525) in this showcase? Give the reason that is specific to how
each was trained, not a generic "offline is better" claim.

Solution. It is a sample-budget-and-exploration story, not a superiority of the
algorithm class. Online `q_learning` was trained for only 400 episodes on
purpose. Tabular Q-learning in this MDP needs roughly 5000 episodes to converge
to `Q*` (see `artifacts/dp/q_learning_gap.csv`, which tabulates the per-state gap
`|learned_q - optimal_q|`), so at 400 episodes its estimates are still noisy and
it has latched onto escalation as a deceptively safe action — hence
`escalation_rate` 0.65. FQI, by contrast, does not need to explore: it sweeps the
entire logged dataset and applies the Bellman backup in batch, reusing every
transition many times. With 1418 logged transitions
(`artifacts/offline_rl/dataset_summary.csv`) it converges in 6 sweeps
(`artifacts/offline_rl/training_curve.csv`) and inherits the heuristic log's good
behavior instead of having to rediscover it online under a tiny episode budget.
The residual online gap is itself the intended lesson — see [the RL
ladder](rl-ladder.md) and the update rule:

```text
Online Q-learning update:
  Q(s,a) <- Q(s,a) + alpha * ( r + gamma * max_a' Q(s-prime, a') - Q(s,a) )

FQI batch backup (per sweep, over all logged (s,a,r,s-prime)):
  target = r + gamma * max_a' Q_old(s-prime, a')
  fit Q_new to regress toward target on the whole dataset
```

Here `max_a'` ranges over actions `a'` available at `s-prime`. The online rule
sees each transition roughly once in order; the FQI rule sees all of them every
sweep. To watch the gap close yourself, read `artifacts/dp/q_learning_gap.csv`:
it lists per state and action the `learned_q_value`, the `optimal_q_value` from
dynamic programming, and their `abs_gap` — a concrete reminder that "it will
converge eventually" is not the same as "it has converged."

## Exercise 5 — FQI convergence by hand

Prompt. From `artifacts/offline_rl/training_curve.csv`, the FQI Bellman residual
per sweep is 2.0, 1.8, 1.62, 0.945, 0.3402, 0.0. (a) After how many sweeps has
FQI converged? (b) How many state-action pairs are updated each sweep, and why is
that count constant? (c) What does a residual of exactly 0.0 tell you about the
fixed point?

Solution. (a) FQI converges in 6 sweeps — the residual reaches 0.0 on the sixth.
The sequence is strictly decreasing, which is the contraction property of the
Bellman optimality operator at work. (b) 466 state-action pairs are updated each
sweep. The count is constant because FQI is a batch method: every sweep replays
the same fixed set of logged transitions and re-applies the backup to the same
466 pairs that appear in the data — nothing is added or removed between sweeps.
(c) A residual of exactly 0.0 means the value function no longer changes under
the Bellman backup, i.e. `Q = T Q` where `T` is the (empirical) Bellman
optimality operator. The iterate has reached the fixed point on the support of
the dataset. See [offline RL and OPE](offline-rl-and-ope.md) and
`src/learning_agents/offline_rl.py`.

## Exercise 6 — Coverage and the support problem

Prompt. From `artifacts/offline_rl/dataset_summary.csv`, the log holds 1418
transitions but covers only 196 of 371 decision states (`coverage_fraction`
0.5283), and was generated by `heuristic_router` made epsilon-soft with
`epsilon` = 0.6. (a) What does "covered 196 of 371" imply about states the
offline learner has never seen? (b) Why would you make a deterministic heuristic
epsilon-soft before logging? (c) Connect this to why OPE will struggle for some
target policies.

Solution. (a) For the 371 − 196 = 175 uncovered decision states, the dataset
contains zero transitions, so a pure offline method has no evidence there. Any
value it assigns to those states is extrapolation, not estimation — the source of
offline RL's distribution-shift risk. (b) A deterministic policy always picks one
action per state, so its log has no signal about the alternatives, and you can
never estimate `Q(s,a)` for the actions it skips. Injecting exploration with
`epsilon` = 0.6 (act greedily 40% of the time, uniformly at random 60%) widens
the action coverage so the log carries information about more `(s,a)` pairs. (c)
OPE re-weights logged data by how likely the target policy was to take the logged
actions. Where the behavior log has no overlap with a target — poor support — the
re-weighting is built on almost no data and the estimate becomes unreliable. That
is exactly the failure you see for the `random` target in Exercise 7. See
[offline RL and OPE](offline-rl-and-ope.md).

## Exercise 7 — Why WIS beats IS for the random target

Prompt. In `artifacts/ope/estimator_comparison.csv`, evaluate the `random`
target (true value -1.074). Importance sampling (IS) estimates -0.5126
(`abs_error` 0.5614); weighted importance sampling (WIS) estimates -0.905
(`abs_error` 0.169). Explain mechanically why WIS is so much more accurate here,
and why the same gap does not appear for the `heuristic_router` and `dp_optimal`
targets.

Solution. IS forms the estimate as an average of returns scaled by the
per-trajectory importance ratio, the product of `pi_target(a|s) / pi_behavior(a|s)`
over the trajectory:

```text
IS:   V_hat = (1/N) * sum_i  rho_i * G_i
WIS:  V_hat = ( sum_i rho_i * G_i ) / ( sum_i rho_i )
  rho_i = product over t of  pi_target(a_t | s_t) / pi_behavior(a_t | s_t)
  G_i   = discounted return of trajectory i,  N = number of trajectories
```

The `random` target has poor overlap with the heuristic behavior log
(Exercise 6), so a few trajectories get enormous ratios `rho_i` while most get
tiny ones. IS divides by the fixed count `N`, so those few exploding ratios
dominate the sum and the variance blows up — hence `abs_error` 0.5614. WIS
divides by `sum_i rho_i` (self-normalizes), so the same exploding ratios appear in
both numerator and denominator and largely cancel. WIS is slightly biased but
dramatically lower variance, cutting the error to 0.169. For `heuristic_router`
(the behavior policy itself) and `dp_optimal`, the targets are well inside the
log's support, so all ratios are moderate, variance is small, and every estimator
already has `abs_error` below 0.05 — there is no variance pathology for WIS to
fix. See [offline RL and OPE](offline-rl-and-ope.md) and
`src/learning_agents/ope.py`.

## Exercise 8 — When is the doubly robust estimator worth it?

Prompt. The doubly robust (DR) estimator combines a fitted Q-model (the direct
method, DM) with an IS-style correction. Using
`artifacts/ope/estimator_comparison.csv`, compare DM, DR, and IS on the `random`
target (true -1.074), then explain in one sentence what DR buys you that DM alone
does not.

Solution. On the `random` target: DM -0.5597 (`abs_error` 0.5143), DR -0.7201
(`abs_error` 0.3539), raw IS -0.5126 (`abs_error` 0.5614). DR is the most
accurate of the three because it starts from the fitted Q-model's prediction and
adds an importance-weighted correction on the model's residual, partially
repairing the model's bias on this off-support target. What DR buys over DM
alone, in one sentence: a correction term that stays approximately unbiased even
when the Q-model is wrong, as long as either the model or the behavior-policy
estimate is good (the "doubly" in doubly robust). For in-support targets DM and
DR are numerically identical here (`heuristic_router`: both 1.1507; `dp_optimal`:
both 1.192), because the correction term is negligible when overlap is good. See
[offline RL and OPE](offline-rl-and-ope.md).

## Exercise 9 — Find the dominated points on the cost-quality curve

Prompt. From `artifacts/cost_cascade/cost_quality_curve.csv`, each
`effort_budget` (0–4) yields a `total_cost` and an `avg_reward`. A point is
Pareto-dominated if another budget achieves at least as high a reward AND at
least as low a total cost, with one strictly better. (a) List the dominated
budgets. (b) Explain the counterintuitive fact that `total_cost` is not monotone
in budget. (c) State the non-dominated frontier. (d) A product owner needs
`solved_rate` = 1.0, the best reward available, and the lowest `total_cost` among
ties — which budget ships?

Solution. The data:

```text
budget  total_cost  avg_reward  escalation_rate
  0       1.375       0.2592      0.7167
  1       1.6667      0.2633      0.5333
  2       1.745       0.5183      0.2833
  3       1.8767      0.91        0.1667
  4       1.76        1.16        0.0
```

(a) Budgets 1 and 3 are dominated. Budget 1 (cost 1.6667, reward 0.2633) is
beaten by budget 0, which has lower cost (1.375) and essentially the same reward
(0.2592 vs 0.2633) — strictly cheaper at near-equal quality. Budget 3 (cost
1.8767, reward 0.91) is beaten by budget 4, which has both lower cost (1.76) AND
higher reward (1.16) — strictly better on both axes. (b) `total_cost` rises from
budget 0 to 3 but then falls at budget 4 (1.8767 → 1.76). The reason: escalation
is the most expensive tier, and its rate collapses to 0.0 at budget 4. Giving the
agent enough budget to actually solve cases itself removes the costly fallback,
so spending more on the cheap early tiers saves more on the expensive last one.
Cost is non-monotone because the action mix shifts. (c) The Pareto-non-dominated
frontier is budgets 0, 2, and 4 — cheap-and-weak, balanced, and
best-reward-yet-also-cheaper-than-3. (d) Ship budget 4: only budgets 3 and 4
reach `solved_rate` 1.0, and budget 4 beats budget 3 on both reward (1.16 vs
0.91) and cost (1.76 vs 1.8767), so there is no tie to break and the dominated
budget 3 is never the right pick. The habit to carry away: "more budget" is not
"more cost" once the expensive escalation tier drops out. See [the cost-aware
cascade](cost-aware-cascade.md) and `src/learning_agents/cost_cascade.py`.

## Exercise 10 — Predict the IQL team reward in the Climbing game

Prompt. Lane C runs the cooperative Climbing game (Claus and Boutilier, 1998),
relabeled researcher × responder, with payoff matrix (rows = researcher
[deep_research, search, skim], cols = responder [detailed, standard, brief]):

```text
              detailed  standard  brief
deep_research    11       -30       0
search          -30        7        6
skim              0        0        5
```

Independent Q-learning (IQL) has each agent learn alone, treating the other as
part of the environment. Predict the `final_team_reward` IQL converges to and the
joint action it settles on, then explain the mechanism. Check against
`artifacts/marl/coordination_comparison.csv`.

Solution. IQL converges to `final_team_reward` 5.0 with `final_joint_action`
`skim+brief`, against an `optimal_team_reward` of 11.0 and
`coordination_success_rate` 0.0, as recorded in
`artifacts/marl/coordination_comparison.csv`. Mechanism: the global optimum
`deep_research+detailed` pays 11, but that row also holds the catastrophic −30
penalties whenever one agent commits and the other does not match. Each
independent learner sees the other's choices only as nonstationary noise, so the
high-reward-but-high-variance equilibrium is too risky to discover; the agents
retreat to a safer, miscoordinated region worth far less than 11. This is the
classic relative-overgeneralization / shadowed-equilibrium failure of independent
learners in cooperative games. With the labels above, the `skim+brief` cell reads
5, matching the recorded `final_team_reward` of 5.0. See [Lane C: multi-agent
RL](lane-c-marl.md) and `src/learning_agents/marl.py`.

## Exercise 11 — Why does centralized JAL reach the optimum?

Prompt. Joint Action Learning (JAL) is centralized: it learns values over the
joint action of both agents. From `artifacts/marl/coordination_comparison.csv`,
what `final_team_reward` and `coordination_success_rate` does `joint` reach, and
why can it find the optimum that IQL (Exercise 10) cannot?

Solution. `joint` reaches `final_team_reward` 11.0 — exactly the
`optimal_team_reward` — with `coordination_success_rate` 1.0 and joint action
`deep_research+detailed`. It succeeds because it optimizes over the joint action
space directly: it can evaluate the cell `(deep_research, detailed)` as a single
coordinated choice and see its value of 11 without exposure to the −30 off-diagonal
penalties that doom an uncoordinated learner. The −30 entries punish only
mismatched unilateral moves; a centralized learner never has to take a one-sided
gamble. The cost of this is the obvious one: the joint action space grows
multiplicatively with the number of agents, so JAL does not scale the way
independent learning does. The tension — coordination quality versus
scalability — is the lane's core lesson. See [Lane C: multi-agent
RL](lane-c-marl.md).

## Exercise 12 — Modify and predict: raise epsilon in the logging policy

Prompt. The offline dataset was logged by `heuristic_router` made epsilon-soft
with `epsilon` = 0.6, giving `coverage_fraction` 0.5283 over 371 decision states
(`artifacts/offline_rl/dataset_summary.csv`). Suppose in `scripts/run_offline_rl.py`
you raise `epsilon` toward 1.0 and regenerate the log at the same number of
episodes. (a) Predict the direction of change in `coverage_fraction`. (b) Predict
what happens to FQI's ability to recover a near-optimal policy. (c) Predict the
effect on the IS estimate for the `random` target in OPE. State the trade-off in
one sentence.

Solution. (a) `coverage_fraction` increases. Higher `epsilon` means more uniform
random actions, so more of the 371 decision states and more `(s,a)` pairs appear
in the log; in the limit `epsilon` = 1.0 the behavior policy is uniform random
and coverage is maximal for a fixed budget. (b) FQI's coverage improves, so it
has evidence in more states and its extrapolation risk on previously-unseen states
shrinks — its recovered policy should stay near-optimal or improve on the margins,
since FQI is off-policy and does not care that the data came from a worse behavior
policy. (c) For OPE of the `random` target, raising `epsilon` makes the behavior
policy look more like uniform-random, improving overlap with the `random` target,
which shrinks the importance ratios `rho_i` and therefore reduces IS variance —
the `abs_error` of 0.5614 (Exercise 7) would come down. The trade-off in one
sentence: a more exploratory logging policy buys broader coverage and better
overlap (good for offline learning and for OPE) at the price of a lower-quality,
less realistic behavior policy whose own on-policy reward is worse. Verify by
editing `scripts/run_offline_rl.py`, re-running, and diffing
`artifacts/offline_rl/dataset_summary.csv`. See [offline RL and
OPE](offline-rl-and-ope.md).

## Exercise 13 — Reward design: would more KL budget help the toy methods?

Prompt. Lane B compares four preference-optimization methods on a toy 4×5 quality
matrix (`artifacts/preference/method_comparison.csv`). RLHF, DPO, GRPO, and RLVR
all lift `expected_quality` from the reference's 0.49 to about 0.999 with a `KL`
to the reference of about 1.6. (a) Why are all four so close? (b) The objective
maximizes reward minus `beta` · `KL` to the reference. If you lowered `beta`
(allowing more KL drift), would `expected_quality` rise meaningfully here? (c)
Why must you not read these numbers as a verdict on real language models?

Solution. (a) On a tiny 4×5 quality matrix the optimum is trivially reachable:
the four methods are different routes up the same small hill, so they converge to
nearly the same `expected_quality` (0.9988–0.9995), the same controlled `KL`
(about 1.59–1.60), and the same `win_rate_vs_reference` (~0.899 versus the
reference's 0.5). (b) Lowering `beta` would not help meaningfully:
`expected_quality` is already pinned at ~0.999, the ceiling of this toy matrix,
so loosening the KL leash only lets the policy drift farther for no quality gain.
The objective trades quality against drift:

```text
maximize over pi:   E[ r ]  -  beta * KL( pi || pi_reference )
```

where `r` is the toy quality reward, `pi_reference` is the starting policy, and
`beta` weights how hard you penalize divergence. Once `E[r]` saturates, spending
KL is pure waste. (c) These are concepts at small scale, not a language-model
benchmark: a 4×5 matrix has no generalization, no tokenization, no reward-model
noise, and no distribution shift, so the near-ties say nothing about how RLHF,
DPO, GRPO, or RLVR rank on real models. The honest framing — toy-scale — is the
point. See [reward design and hacking](reward-design-and-hacking.md) and [Lane B:
preference optimization](lane-b-preference-optimization.md).

## Exercise 14 — Synthesis: rank the five policies and defend the ranking

Prompt. Using only `artifacts/eval/policy_comparison.csv`, rank all five policies
by `avg_reward`, then state for each whether governance would pass or reject it
and the single number that decides the call.

Solution. By `avg_reward`, descending:

```text
rank  policy            avg_reward  governance   deciding signal
 1    dp_optimal         1.2142     pass*        ceiling; solved_rate 1.0
 2    offline_fqi        1.2067     pass         solved_rate 1.0, escalation 0.30
 3    heuristic_router   1.16       pass         escalation_rate 0.0
 4    q_learning         0.8525     REJECT       escalation_rate 0.65
 5    random            -1.1817     REJECT       solved_rate 0.5333 (and unsafe 0.4667)
```

`dp_optimal` is the planning ceiling (computed, not deployed as a learner), shown
for reference. `offline_fqi` and `heuristic_router` both pass: high reward, full
`solved_rate`, no over-escalation. `q_learning` is rejected on its
`escalation_rate` of 0.65 despite solving every case. `random` is rejected on its
`solved_rate` of 0.5333 — it cannot even reliably finish — compounded by unsafe
decisions (0.4667). The single most important habit: governance is a gate on the
right counter, not a ranking by reward alone. See [evaluation and
governance](evaluation-and-governance.md) and [the RL ladder](rl-ladder.md).

## See also

- [The RL ladder](rl-ladder.md) — the tabular methods and the q_learning gap behind Exercises 3 and 4.
- [Offline RL and OPE](offline-rl-and-ope.md) — FQI, coverage, and the estimators behind Exercises 5–8 and 12.
- [Evaluation and governance](evaluation-and-governance.md) — the hard gate behind Exercises 1, 2, and 14.
- [The cost-aware cascade](cost-aware-cascade.md) — the Pareto reasoning behind Exercise 9.
