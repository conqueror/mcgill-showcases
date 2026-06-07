# Results Dashboard

One place to see what every lane of the showcase actually produced. Each chart is a plain-text
rendering of a checked artifact — run `make run` (or `make smoke` for the fast path, plus
`make run-drl` for the optional deep-RL lane) to regenerate the underlying CSVs, then read them
directly if you want the full precision. Numbers here are the full-run values; `--quick` mode
produces the same qualitative story with softer numbers.

This page is a synthesis; for the *why* behind each result, follow the per-lane guides linked at the
end and start from [where learning lives](locus-of-learning.md).

## Scorecard: every method at a glance

| Method | Locus / lane | Headline result | Lesson |
| --- | --- | --- | --- |
| Contextual bandit | orchestration (warm-up) | regret flattens as exploration pays off | exploration vs exploitation |
| Q-learning (tabular) | orchestration | avg reward 0.85, escalates 0.65 | under-trained online control over-escalates |
| SARSA | orchestration | on-policy contrast to Q-learning | on- vs off-policy backup |
| Dynamic programming | orchestration (planning) | `dp_optimal` 1.2142 — the exact ceiling | planning gives ground-truth `Q*` |
| REINFORCE | orchestration | tabular policy gradient | optimize the policy directly |
| Offline FQI | orchestration (offline) | 1.2067 — nearly matches the ceiling | learn from a good log without new data |
| OPE (IS/WIS/DM/DR) | orchestration (offline) | in-support err < 0.05; off-support IS err 0.56 | overlap is everything |
| Cost-aware cascade | orchestration | best reward 1.16 at *lower* cost than budget 3 | more effort is not monotonically more costly |
| DQN | orchestration (deep) | 1.1783 — recovers the ceiling | value approximation reaches `Q*` |
| PPO | orchestration (deep) | 0.6933 — safe local optimum | policy gradient can settle for "safe" |
| Lane A — Agents SDK | orchestration (framework) | learned policy drives tool/handoff calls | the framework executes, it does not learn |
| Lane B — RLHF/DPO/GRPO/RLVR | LLM weights | quality 0.49 → ~0.999 | learning the token policy (toy scale) |
| Lane C — IQL vs JAL | multi-agent | coordination 0.0 vs 1.0 | decentralised learners miscoordinate |

## Orchestration policies — average reward

Source: `artifacts/eval/policy_comparison.csv` (bar scale −1.25 … +1.25).

```text
dp_optimal    ██████████████████████████████   1.2142   esc 0.2833 ceiling
offline_fqi   █████████████████████████████    1.2067   esc 0.30   offline RL
heuristic     █████████████████████████████    1.1600   esc 0.00   baseline
q_learning    █████████████████████████        0.8525   esc 0.65   REJECTED
random        █                               -1.1817   esc 1.00   floor
```

The learned online policy (`q_learning`) sits *below* the hand-written baseline and over-escalates,
which is why governance rejects it; offline FQI and DQN are the learners that actually reach the
planning ceiling. See [evaluation and governance](evaluation-and-governance.md).

## Deep RL — value-based vs policy-gradient

Source: `artifacts/drl_optional/rl_family_comparison.csv` (bar scale 0 … 1.25).

```text
dqn   (value-based)        ████████████████████████████   1.1783   recovers the ceiling
q_learning (tabular)       ████████████████████           0.8417   over-escalates
ppo   (actor-critic)       █████████████████              0.6933   safe local optimum
```

Value approximation (DQN) reaches essentially the optimum; the policy-gradient method (PPO) settles
into a safe, over-escalating policy — a genuine, well-known contrast, not a bug. See
[deep RL: DQN and PPO](deep-rl.md).

## Cost-aware cascade — quality climbs, cost does not

Source: `artifacts/cost_cascade/cost_quality_curve.csv` (reward bar scale 0 … 1.25).

```text
budget 0   ██████          reward 0.2592   total_cost 1.375    (frontier)
budget 1   ██████          reward 0.2633   total_cost 1.6667   dominated
budget 2   ████████████    reward 0.5183   total_cost 1.745    (frontier)
budget 3   ██████████████████████   reward 0.91     total_cost 1.8767   dominated
budget 4   ████████████████████████████   reward 1.16   total_cost 1.76   (frontier)
```

Budget 4 has the best reward *and* a lower total cost than budget 3, because escalation (the
expensive last tier) falls to zero — the Pareto-non-dominated points are budgets 0, 2, and 4. See
[cost-aware cascades](cost-aware-cascade.md).

## Off-policy evaluation — overlap is everything

Source: `artifacts/ope/estimator_comparison.csv`, absolute error for the off-support `random`
target (bar scale 0 … 0.6). For the in-support `heuristic_router` and `dp_optimal` targets every
estimator lands within 0.05.

```text
importance_sampling            ████████████████████████████   0.5614   variance explodes
direct_method                  ██████████████████████████     0.5143
doubly_robust                  ██████████████████             0.3539
weighted_importance_sampling   ████████                       0.1690   variance tamed
```

When the target policy strays from the behaviour log (poor overlap), plain importance sampling
blows up; weighting (WIS) slashes the variance. See
[offline RL and off-policy evaluation](offline-rl-and-ope.md).

## Preference optimization — quality lift (toy scale)

Source: `artifacts/preference/method_comparison.csv` (expected-quality bar scale 0 … 1.0).

```text
reference   ███████████████                0.4900   starting policy
rlhf        ██████████████████████████████ 0.9994   KL 1.5995
dpo         ██████████████████████████████ 0.9995   KL 1.5986
grpo        ██████████████████████████████ 0.9988   KL 1.593
rlvr        ██████████████████████████████ 0.9988   KL 1.5927
```

All four methods lift the toy LM's expected quality from 0.49 to about 0.999 with a controlled KL
leash (~1.6) to the reference. See [Lane B: preference optimization](lane-b-preference-optimization.md).

## Multi-agent coordination — IQL vs JAL

Source: `artifacts/marl/coordination_comparison.csv` (team-reward bar scale 0 … 11; optimum = 11).

```text
independent (IQL)   ██████████             team reward 5.0    success 0%    skim+brief
joint (JAL)         ██████████████████████ team reward 11.0   success 100%  deep_research+detailed
```

Independent learners miscoordinate onto a safe, suboptimal equilibrium; the centralised joint-action
learner reaches the optimum every time. See [Lane C: multi-agent coordination](lane-c-marl.md).

## See also

- [Where learning lives](locus-of-learning.md) — the taxonomy these results populate.
- [The RL ladder](rl-ladder.md) and [deep RL](deep-rl.md) — the orchestration-policy methods.
- [Evaluation and governance](evaluation-and-governance.md) — how these numbers gate a deployment.
