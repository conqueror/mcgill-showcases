# Reward Design and Reward Hacking

An agent does not optimize what you *mean*. It optimizes the scalar number you
hand it after each step. In a Markov decision process (MDP) that number is the
reward `r` (the project writes it `R_{t+1}`, the reward received after acting at
time `t`), and the agent's whole job is to maximize the discounted return:

```text
G_t = sum_{k>=0} gamma^k * R_{t+k+1}
```

where `gamma` in `[0, 1)` is the discount factor and `G_t` is the return from
time `t`. Every method on the ladder -- bandits, dynamic programming,
Q-learning, SARSA, REINFORCE, DQN -- inherits this reward unchanged. The reward
is not a tuning knob bolted onto a solved problem. **The reward is the
specification.** Get it wrong and every learner downstream faithfully optimizes
the wrong thing. This page reads the aligned judge rubric in
`src/learning_agents/reward.py`, contrasts it with a deliberately broken twin,
and shows -- with measured numbers -- how a degenerate policy can win under the
broken reward while solving almost nothing.

## The agent and its action costs

The agent answers student requests. At each step it picks one of four actions,
each with a cost `c(a)` drawn from `ACTION_COSTS` in
`src/learning_agents/reward.py`:

| action `a` | meaning | cost `c(a)` |
| --- | --- | --- |
| 0 | `answer_direct` | 0.0 |
| 1 | `retrieve` (gather evidence) | 0.5 |
| 2 | `clarify` (resolve ambiguity) | 0.3 |
| 3 | `escalate` (hand off to a human) | 1.5 |

`escalate` is the most expensive action because it consumes a scarce human.
`answer_direct` is free to *attempt* -- but whether the attempt is good depends
on the state, which is the entire point of the rubric below.

A state `s` carries (among other fields) the request's `difficulty`, its
`ambiguity`, and the `evidence` gathered so far. The shared quality predicate
`evidence_is_adequate(evidence, difficulty)` returns true iff
`evidence >= difficulty`: harder requests demand strictly more grounding. An
answer is *well-grounded* (`_is_well_grounded`) iff evidence is adequate for the
difficulty **and** ambiguity is fully resolved (`ambiguity == 0`).

## The reward is the specification: the judge rubric

The agent never sees a ground-truth answer. Instead a *judge rubric* scores its
committed action against several criteria, exactly as an LLM-as-judge or a
learned reward model would in practice. This is `judge_reward` in
`src/learning_agents/reward.py`, and its per-criterion decomposition is
`rubric_breakdown`. The reward is the sum of five named terms:

```text
R_{t+1} = answer_quality
        + hallucination_penalty
        + escalation_value
        + effort_penalty
        + action_cost
```

Each term encodes one design intent (see `artifacts/reward/reward_spec_good.md`):

- **`answer_quality` = +2.0** for a well-grounded `answer_direct`, else `0`.
  Reward a *correct, grounded* answer.
- **`hallucination_penalty` = -1.5** for an under-grounded `answer_direct`, else
  `0`. Penalize answering without adequate evidence or with unresolved
  ambiguity -- the wrong-answer / hallucination risk.
- **`escalation_value`** for `escalate`, scaled by genuine need. With
  `need = difficulty + ambiguity` of the pre-action state:

  ```text
  escalation_value = 0.6 + 0.45 * need
  ```

  Escalating a hard, ambiguous request earns more than escalating an easy one.
- **`effort_penalty` = -0.2`** for each `retrieve`/`clarify` that was *not*
  needed (evidence already adequate, or ambiguity already `0`). Discourage
  busywork.
- **`action_cost` = -c(a)`**, the raw price of the action.

The rubric is multi-objective: it bundles *correct*, *grounded*, *safe*, and
*efficient* into one scalar, and `rubric_breakdown` keeps the trade-offs
auditable rather than buried in a single number. By construction the aligned
optimum is the genuinely good policy. A well-grounded answer scores
`+2.0` at zero action cost. That strictly beats an under-grounded answer
(`-1.5`), and it beats escalating a request that did not need a human: on an
easy, unambiguous request (`need = 0`) escalation nets
`0.6 - 1.5 = -0.9`. Escalation only pays off when the request genuinely
warrants a human.

## What reward hacking is: optimizing a proxy

We almost never get to write the *true* objective directly. We write a *proxy*:
a rule, a learned reward model, an LLM judge -- something we hope correlates
with what we actually want. **Reward hacking** is what happens when an optimizer
drives that proxy up while the true objective stays flat or falls. The policy
isn't broken; it is doing exactly its job. The *measurement* was broken.

This is **Goodhart's law**: "when a measure becomes a target, it ceases to be a
good measure" (Goodhart, 1975; sharpened for ML by Manheim and Garrabrant,
2019). A proxy that merely *correlates* with intent on the data you used to
design it can diverge sharply once a learner is free to seek out the corners
where proxy and intent disagree. Optimization pressure is an adversary hunting
for exactly those corners.

## A hackable twin, and the rank reversal

To make the failure measurable, `src/learning_agents/reward.py` ships a
deliberately misspecified twin, `hackable_reward`. It keeps the same *shape* as
the judge rubric (a quality term, an escalation term, an evidence term, minus
cost) but introduces two classic mistakes (see
`artifacts/reward/reward_spec_bad.md`):

```text
answer_direct: +1.0 if well-grounded else -0.2
retrieve:      +0.8   (always, no redundancy penalty)
clarify:       +0.1
escalate:      +3.0   (flat, regardless of need)
then subtract c(a) in every case.
```

Two bugs, each a real design smell:

1. **It overpays escalation.** `escalate` earns a flat `+3.0` regardless of
   need; net of its `1.5` cost that is `+1.5` for doing nothing useful. A
   trivial "always escalate" policy farms this.
2. **It pays for raw evidence regardless of need.** Every `retrieve` earns
   `+0.8` with no redundancy penalty, so "always retrieve" piles up grounding
   it does not need.

And critically, it **under-credits the good behavior**: a well-grounded answer
earns only `+1.0`, less than a couple of needless retrievals or a single
escalation. The result is a *rank reversal* -- degenerate policies beat the
genuinely good one. That reversal is the diagnostic signature of reward hacking.

### Measuring it: the controlled swap

`src/learning_agents/reward_study.py` turns the two reward definitions into a
measurement. `compare_reward_models` holds the policies, scenarios, and horizon
fixed and swaps **only** the reward function, so any change in the policy
ranking must come from the objective, not the dynamics. It re-scores two foil
policies from `src/learning_agents/policies.py`:

- `always_escalate` (`AlwaysEscalatePolicy`): the degenerate strategy -- hand
  every request to a human, solve almost nothing yourself.
- `heuristic_router` (`HeuristicRouterPolicy`): the strong hand-written
  baseline -- it grounds, disambiguates, and answers.

`reward_hacking_report` lays out the four `avg_reward` cells. From
`artifacts/reward/reward_hacking_report.md`:

| reward model | `always_escalate` avg_reward | `heuristic_router` avg_reward |
| --- | --- | --- |
| bad (hackable proxy) | **1.5** | 1.14 |
| good (judge rubric) | **-0.09** | **1.26** |

Read the table by column-comparison within each row:

- **Under the proxy**, `always_escalate` (1.5) *outscores* the genuinely good
  `heuristic_router` (1.14). The proxy declares the do-nothing policy the
  winner.
- **Under the judge rubric**, the ranking flips: `heuristic_router` (1.26) beats
  `always_escalate` (-0.09 -- a *net loss*, because escalating requests that did
  not need a human costs more than its small payoff returns).

### The hack, exposed by a second metric

The reason the proxy "win" is hollow shows up the moment you look past the
scalar reward. The same report records `solved_rate`:

- `always_escalate` solved rate: **0.6**
- `heuristic_router` solved rate: **1.0**

The always-escalate policy that "won" the proxy actually resolves only 60% of
requests on its own (the rest are punted to a human), while the heuristic router
solves everything. This is the general lesson: **always pair the optimization
target with an orthogonal sanity metric.** A reward model is a hypothesis about
what good behavior is, and `solved_rate` is the held-out check that catches the
hypothesis lying.

## A milder, real exploitation story: q_learning over-escalates

The hackable twin is a designed extreme. But reward exploitation also shows up
in a *correct* reward when a learner is under-trained, and the showcase has a
live example. In `artifacts/eval/policy_comparison.csv`, online tabular
Q-learning (400 episodes) lands at:

| policy | avg_reward | escalation_rate | solved_rate |
| --- | --- | --- | --- |
| dp_optimal | 1.2142 | 0.2833 | 1.0 |
| heuristic_router | 1.16 | 0.0 | 1.0 |
| q_learning | 0.8525 | **0.65** | 1.0 |
| random | -1.1817 | 1.0 | 0.5333 |

`q_learning` escalates on **65%** of decisions -- more than twice the
planning-ceiling rate (`dp_optimal` escalates 28.33% of the time) and far above
the heuristic router, which never escalates. Escalation under the *aligned*
rubric does carry a real payoff (`0.6 + 0.45 * need`), so a learner that has not
yet learned to ground and answer can lean on escalation as a safe-ish default
that still collects some reward. It is over-using a legitimately-rewarded action
rather than gaming a broken one -- a milder cousin of reward hacking. The judge
rubric still penalizes it correctly: `q_learning` earns only 0.8525, well below
the 1.16 of `heuristic_router` and the 1.2142 planning ceiling.

Two honest footnotes the showcase deliberately preserves. **First, this is not a
reward bug.** Online Q-learning converges to the exact optimal action value
`Q*(s, a)` only around 5000 episodes; the showcase trains 400 on purpose, so the
residual gap (and the over-escalation it produces) *is* the lesson -- see
`artifacts/dp/q_learning_gap.csv`. **Second, governance catches it.** The same
over-escalation the reward correctly scores down also trips the deployment gate,
and `q_learning` is **rejected**: the reward ranks honestly, and an independent
governance check refuses to ship a policy that leans on the human-escalation
escape hatch. The contrast is instructive -- offline Fitted-Q learned from the
same heuristic log scores 1.2067 and *nearly matches the planning ceiling*, so
the failure is specific to the under-trained online learner, not to learning
itself.

## Concrete reward-design guidance

The two rubrics in this showcase distill into a checklist you can reuse.

1. **Reward outcomes, not effort.** The hackable proxy paid `+0.8` per
   `retrieve` -- effort -- and got farmed. The judge rubric pays `+2.0` only for
   a *well-grounded answer* -- an outcome -- and *charges* for needless effort
   (`effort_penalty = -0.2`). Pay for the result; price the steps.

2. **Make the costly escape hatch cost something proportional.** Escalation is
   the expensive action. The proxy's flat `+3.0` made it a free lunch; the
   rubric's `0.6 + 0.45 * need` ties the payoff to genuine need and nets
   *negative* when a human was not warranted. Any "safe" fallback (escalate,
   refuse, defer) must be priced so it is not the lazy global optimum.

3. **Do not under-credit the behavior you actually want.** The proxy's subtle
   bug was paying the good answer only `+1.0` -- the rank reversal needed
   *both* the overpaid hack and the under-credited target. Audit the ceiling on
   the desired action relative to every shortcut.

4. **Decompose and audit the reward.** `rubric_breakdown` returns per-criterion
   terms that sum (after rounding) to `judge_reward`, asserted in the test
   suite. A reward you can read term by term is a reward you can debug; a single
   opaque scalar hides exactly the trade-off that gets gamed.

5. **Always co-monitor an orthogonal metric.** `solved_rate` is the check that
   exposed the proxy's hollow win and the over-escalation alike. Never let the
   number you optimize be the only number you watch.

6. **Anticipate the optimizer as an adversary (Goodhart).** Before deploying,
   ask: "what is the laziest policy that maximizes this number?" If the answer
   is a degenerate policy you would be embarrassed to ship, the proxy is
   misspecified. Fix the reward, not the policy.

A note of honesty on scope: this is a small, fully specified MDP, so we can
literally enumerate the hack and prove the rank reversal. Real reward models for
language agents are far higher-dimensional and the hacks are subtler -- but the
mechanism is identical, which is why this toy is worth studying. Preference-based
methods that *learn* the reward from comparisons (RLHF, DPO, GRPO, RLVR) face
exactly this hazard at scale, with the KL-divergence term `KL` (weighted by
`beta`) acting partly as a leash against drifting too far from a trusted
reference policy `pi`; see the toy treatment in
[Lane B preference optimization](lane-b-preference-optimization.md).

## See also

- [Evaluation and governance](evaluation-and-governance.md) -- the orthogonal
  metrics and deployment gate that caught the over-escalating `q_learning`.
- [The RL ladder](rl-ladder.md) -- every learner on the ladder inherits this
  reward, including the under-trained `q_learning`.
- [Lane B preference optimization](lane-b-preference-optimization.md) -- learning
  a reward from preferences, and the KL leash that contains it.
- [Glossary](glossary.md) -- definitions of return, reward, Goodhart, and proxy
  objective.
