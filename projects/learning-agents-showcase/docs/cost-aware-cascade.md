# Cost-Aware Effort Cascades

A deployed agent does not pay for its most expensive option on every request. It runs
a *cascade*: try the cheap moves first, and fall through to the expensive last tier only
when the cheap tiers cannot resolve the request. How far up that cost ladder you are
willing to climb before committing is an operating choice, and different choices trade
money and latency for answer quality. This page makes that trade-off measurable, reads
the realised numbers off `artifacts/cost_cascade/cost_quality_curve.csv`, and teaches
the Pareto frontier you actually pick an operating point from.

The code lives in `src/learning_agents/cost_cascade.py`. This is an
*evaluation and deployment* concern layered on top of the learned and heuristic policies,
not a new learning algorithm: the same cost and quality metrics the governance rung
reports, swept across one cost knob to expose the frontier.

## The student-support task in one paragraph

The agent answers student questions. On each step it observes a state `s` and picks one
of four actions: `answer_direct` (free, but only correct when the request is grounded
and unambiguous), `retrieve` (cheap grounding), `clarify` (cheap disambiguation), or
`escalate` (hand off to a human — the expensive last tier). The reward `r` rewards
solving the request and penalises both wasted effort and unsafe answers, so the policy
is judged on quality net of what it spent. See [the locus-of-learning map](locus-of-learning.md)
for where this environment sits relative to the rest of the showcase.

## Total cost: money plus latency

A single quality number hides the trade-off, because "expensive" has two faces. The
sweep in `cost_cascade_curve` records both and combines them.

```text
total_cost = avg_action_cost + latency_cost_per_step * avg_steps
```

Symbols:

- `avg_action_cost` — the average resource/money price of the actions taken in an
  episode. A human `escalate` is by far the dearest action; cheap grounding and a direct
  answer cost little or nothing.
- `avg_steps` — the average number of orchestration steps an episode took. This is the
  *latency* face of cost: more steps means a slower answer.
- `latency_cost_per_step` — the weight that converts one extra step into the reward's
  cost units. In the showcase it is fixed at `0.3` (the `latency_cost_per_step` default
  in `cost_cascade_curve`). Tune it to how much latency matters in your deployment; the
  *shape* of the frontier depends on this weight.

Reporting both faces is what creates a real frontier. Spending cheap grounding lets the
agent answer hard requests itself, so it pays for fewer human escalations (money falls) —
but each grounding step adds latency (steps rise). The cheapest-fastest setting and the
highest-quality setting are therefore different operating points, and you must choose
between them.

## The effort-budget policy

`EffortCascadePolicy` is the cascade with a single knob, `effort_budget`: the maximum
number of cheap grounding steps (`clarify` or `retrieve`) it will spend before it must
commit. Its `select_action(s)` logic, read directly from `cost_cascade.py`, is:

1. **Cheap tier (grounding).** While `state.step < effort_budget` *and* the request
   still needs work (`ambiguity > 0` or evidence is not yet adequate) *and* the action
   is affordable within the remaining budget and horizon: `clarify` first when the
   request is ambiguous, otherwise `retrieve` to add grounding. `clarify` is tried
   before `retrieve` because it is the cheaper disambiguation.
2. **Commit tier.** Once the budget is spent — or the request is already grounded and
   unambiguous — commit to the *cheapest adequate terminal action*: `answer_direct` when
   grounded and unambiguous (free and correct); otherwise `escalate` when the request is
   hard (`difficulty >= 2`) or still ambiguous; otherwise answer directly.

Escalation is the expensive last tier by construction: the policy only reaches it after
the cheap tiers are exhausted or disallowed. Sweeping `effort_budget` slides the policy
along the cost/quality frontier. A small budget leans on cheap immediate commits and
pays for escalations it could have avoided; a larger budget pays for grounding so it can
answer hard requests itself.

The affordability gate (`_can_afford`) checks that a grounding action both fits the
remaining `budget` and leaves the step counter below the `horizon` — so the cascade never
spends effort it cannot pay for or steps it does not have. The policy is stateless: it is
a pure function of the observed state, so `reset()` does nothing.

## The cost/quality curve

`cost_cascade_curve` evaluates one `EffortCascadePolicy` per effort level on the fixed
scenarios (12 seeded rollouts per scenario, horizon 5) and reads the metrics straight
from the shared evaluation harness. The realised numbers from
`artifacts/cost_cascade/cost_quality_curve.csv` are below. `total_cost` is rounded to
four decimals as the artifact stores it.

| effort_budget | avg_action_cost | avg_steps | total_cost | avg_reward | escalation_rate | solved_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.075 | 1.0 | 1.375 | 0.2592 | 0.7167 | 0.85 |
| 1 | 1.1067 | 1.8667 | 1.6667 | 0.2633 | 0.5333 | 0.8833 |
| 2 | 0.99 | 2.5167 | 1.745 | 0.5183 | 0.2833 | 0.9 |
| 3 | 1.0067 | 2.9 | 1.8767 | 0.91 | 0.1667 | 1.0 |
| 4 | 0.84 | 3.0667 | 1.76 | 1.16 | 0.0 | 1.0 |

Read the columns together and the mechanism is visible. As `effort_budget` rises from 0
to 4, `escalation_rate` falls monotonically (`0.7167 -> 0.5333 -> 0.2833 -> 0.1667 -> 0.0`)
because the agent increasingly grounds and answers hard requests itself. `solved_rate`
climbs to a perfect `1.0` once the budget is large enough (budgets 3 and 4), and
`avg_reward` rises steadily to its maximum of `1.16` at budget 4. Meanwhile `avg_steps`
rises monotonically (`1.0 -> 3.0667`): more grounding is genuinely slower.

## The Pareto frontier and dominated points

A practitioner does not pick a budget by eyeballing one column. They look at the
cost/reward plane — `total_cost` on one axis, `avg_reward` on the other — and keep only
the **non-dominated** points. `cost_efficient_frontier` encodes the rule exactly: an
operating point is *dominated* if some other point is at least as cheap on `total_cost`
**and** at least as rewarding, while strictly better on one of the two. Nobody would
ever choose a dominated point, so filtering them out leaves the menu of sensible choices.

Applying that rule to the table:

- **Budget 3 is strictly dominated by budget 4.** Budget 4 is both cheaper
  (`total_cost` `1.76 < 1.8767`) *and* more rewarding (`avg_reward` `1.16 > 0.91`). This
  is the dominance the frontier filter catches outright — budget 4 beats budget 3 on
  both axes at once.
- **Budget 1 is dominated in practice.** No single point beats it on both axes
  simultaneously, but it is never the rational choice: budget 0 is cheaper
  (`1.375 < 1.6667`) at essentially the same reward (`0.2592` versus `0.2633`), while
  budget 2 buys far more reward (`0.5183` versus `0.2633`) for only a little more cost
  (`1.745` versus `1.6667`). Budget 1 is squeezed out from both sides, so it is left off
  the practical frontier.

That leaves **budgets 0, 2, and 4 as the non-dominated operating points** — the real menu.
Budget 0 is the cheapest-fastest corner (lowest `total_cost`, lowest reward); budget 4 is
the highest-quality corner; budget 2 is the genuine middle ground that trades a little
cost for a real reward gain over budget 0.

## The counterintuitive result: more effort, lower cost

The headline of this page is that **`total_cost` is not monotonic in `effort_budget`**.
Reading the `total_cost` column top to bottom:

```text
budget 0: 1.375
budget 1: 1.6667
budget 2: 1.745
budget 3: 1.8767   <- the most expensive setting
budget 4: 1.76     <- MORE effort, yet CHEAPER than budget 3
```

Budget 4 spends the *most* effort, yet it is **cheaper than budget 3** (`1.76 < 1.8767`)
*and* delivers the **best reward in the whole sweep** (`1.16`). Both facts hold at once.

The cause is the expensive last tier. At budget 4 the cascade has enough grounding
allowance to answer every hard request itself, so `escalation_rate` drops to exactly
`0.0`. Eliminating the dearest action lowers `avg_action_cost` (`0.84`, the lowest in the
table) by more than the extra latency steps add back, so the *combined* `total_cost`
actually falls. In other words: paying for more cheap grounding bought its way out of the
single most expensive action, and the net bill went down.

The lesson generalises. More effort is **not** monotonically more expensive when the
effort substitutes for an even costlier fallback. Whenever your cascade has an expensive
escalation tier, the cost curve can be non-monotonic, and the cheapest setting may not be
the one that does the least work. You have to measure `total_cost` end to end — counting
both money and latency — rather than assume "less effort is cheaper."

## Recommended operating point

`recommended_operating_point` first computes the frontier, then returns the
**highest-reward point on it** (ties broken toward the cheaper point so the choice is
deterministic). On this curve that is **budget 4**: `avg_reward` `1.16`, `total_cost`
`1.76`, `solved_rate` `1.0`, `escalation_rate` `0.0`.

Budget 4 is an unusually clean recommendation here because it is *not* a compromise — it
is the best reward and it strictly dominates the next-highest-effort setting on cost, so
choosing it costs you nothing relative to budget 3. This is a defensible default when
reward is the objective and cost is a soft constraint. A real deployment with a hard
budget cap would instead take the best reward *under* the cap (for example, settling for
budget 2 if latency near `2.5` steps were the most you could tolerate); this picker
returns the unconstrained reward-maximiser on the frontier.

One honest caveat: the recommendation is sensitive to `latency_cost_per_step`. At `0.3`,
budget 4 wins outright. Raise the latency weight enough and the extra steps budget 4
spends would eventually outweigh the escalation savings, pushing the recommended point
back down the ladder. The frontier is a function of how much *you* price latency, which
is why the sweep reports both cost faces rather than collapsing them prematurely.

## How this connects to the rest of the showcase

The cost knob here is orthogonal to *which* policy you run. The
[evaluation-and-governance](evaluation-and-governance.md) rung reports the same cost and
quality metrics for every policy, and it is the reason the online `q_learning` agent is
governance-rejected: it over-escalates (escalation rate `0.65`), spending the expensive
last tier when cheaper tiers would do — exactly the failure this cascade is designed to
avoid. The cascade shows the *deployment-time* dial for trading cost against quality;
governance is the gate that rejects a policy whose cost profile is unsafe regardless of
the dial.

## See also

- [Locus of learning](locus-of-learning.md) — where this environment and its actions sit in the showcase.
- [Evaluation and governance](evaluation-and-governance.md) — the cost/quality metrics this curve reuses, and the over-escalation rejection.
- [Offline RL and OPE](offline-rl-and-ope.md) — estimating a policy's value (and cost) from logged data before deploying it.
- [Math notes](math-notes.md) — notation reference for the symbols used above.
