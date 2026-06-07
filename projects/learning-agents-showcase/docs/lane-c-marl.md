# Lane C: Multi-Agent Coordination (Independent vs Joint-Action Learning)

The first two loci of learning in this showcase put a *single* policy under pressure: a lone
agent learning to route requests, or a lone model learning to match a preference. Lane C moves to
the third locus, where the thing that learns is no longer one policy but the **coordination between
several co-adapting policies** -- a researcher and a responder, a planner and a solver, a team of
specialists. This is multi-agent reinforcement learning (MARL), and it has a failure mode that has
no counterpart in the single-agent setting. This page builds the smallest faithful example of that
failure, names the two effects behind it, and measures the gap between a decentralised learner that
falls into the trap and a centralised one that escapes it.

All numbers below come from `artifacts/marl/coordination_comparison.csv` and
`artifacts/marl/training_curves.csv`, both produced by `src/learning_agents/marl.py`.

## Where this sits in the ladder

In the [locus-of-learning](locus-of-learning.md) map, Lane C is **locus C: the coordination
itself**. Loci A and B ask "can a single policy get better?" Locus C asks a harder question: "can
several policies that are each changing at the same time agree on what to do?" The algorithms here
are still built from the value-iteration and temporal-difference machinery introduced in the
[rl-ladder](rl-ladder.md) -- each agent still keeps a `Q(s,a)` table and still does TD updates --
but pointing that machinery at *each other* breaks an assumption the ladder quietly relied on.

## The assumption that breaks: stationarity

Single-agent RL assumes the environment is **stationary**: the transition and reward rules do not
change while you learn. That is what lets a value estimate converge -- you are estimating a fixed
target. The standard tabular TD update for one agent is

```text
Q(s, a) <- Q(s, a) + alpha * ( r + gamma * max_a' Q(s', a') - Q(s, a) )
```

where `Q(s,a)` is the agent's value estimate for taking action `a` in state `s`, `alpha` is the
step size (learning rate), `r` is the reward, `gamma` is the discount factor weighting future
reward, and `s'` is the next state. The term `r + gamma * max_a' Q(s', a')` is the TD *target*.
Convergence guarantees for this update assume that target is anchored to a fixed environment.

Now put two such learners in the same world and let each one treat the other as "part of the
environment." Agent 0's effective environment now includes agent 1's policy `pi_1(a|s)` -- and
agent 1 is *learning*, so `pi_1` is moving. From agent 0's point of view the reward and transition
structure drifts every episode. **Each agent is a moving target for the others.** This is
*non-stationarity*, and it means the convergence story from the single-agent ladder no longer
applies: you are chasing a target that runs away as you approach it.

## The second effect: relative overgeneralization

Non-stationarity by itself would just make learning noisy. The reason independent learners do
something specifically *bad* -- not just slow -- is a related effect called **relative
overgeneralization**.

An independent learner evaluates one of its own actions by averaging the reward it received over
all the things the *partner* happened to do while that action was tried, including the partner's
random exploration. So the value an agent assigns to action `a` is roughly

```text
Q_0(a)  ~=  E over partner action b of [ team_reward(a, b) ]
```

where the expectation is taken over the partner's behaviour distribution `pi_1`. A high-payoff
action whose reward depends on the partner *also* picking the matching action will, under a partner
who is still exploring, have most of its samples land on *mismatched* partner actions. If those
mismatches are heavily penalised, the average drags the good action's estimate *below* that of a
mediocre-but-safe action. Both agents independently conclude the bold action is bad, and the pair
"generalizes" toward a safe, suboptimal joint action. The optimum is invisible to them not because
it pays poorly but because reaching it requires a coordinated bet that neither agent will make
unilaterally.

To see this bite, we need a game whose payoffs are shaped exactly so the safe choice looks better
than the bold one *on average* -- even though the bold one is jointly optimal.

## The game: the cooperative Climbing game

The testbed is the **cooperative Climbing game** (Claus and Boutilier, 1998), a two-agent,
single-shot game with a single shared (team) reward. We relabel it to the agent domain. Agent 0 is
a **researcher** choosing an effort level; agent 1 is a **responder** choosing an answer depth.
Both act simultaneously and both receive the *same* reward, read from a payoff matrix indexed by
their joint action. Using a shared reward makes the game purely *cooperative*: there is no
competition to muddy things, so the only difficulty left is agreeing on a joint action -- precisely
what independent learners struggle with.

The canonical payoff matrix is `((11,-30,0),(-30,7,6),(0,0,5))` from
`src/learning_agents/marl.py`. Rows are the researcher's effort; columns are the responder's depth.
Each cell is the shared team reward `r` for that joint action.

| researcher \ responder | detailed | standard | brief |
| --- | --- | --- | --- |
| **deep_research** | 11  | -30 | 0 |
| **search**        | -30 | 7   | 6 |
| **skim**          | 0   | 0   | 5 |

Read the shape carefully, because the shape *is* the lesson:

- The global optimum is `(deep_research, detailed) = 11` -- thorough research paired with a detailed
  answer pays the most.
- But the two cells *adjacent* to the optimum are catastrophic: `(deep_research, standard) = -30`
  and `(search, detailed) = -30`. Reaching toward the optimum with a partner who is even slightly
  miscoordinated is severely punished.
- Meanwhile `(skim, brief) = 5` is a safe little plateau: low effort, low depth, modestly positive,
  and forgiving of what the partner does (the whole `skim` row and `brief` column avoid the -30
  traps).

So if the researcher cannot count on the responder choosing `detailed`, then "average over the
partner's behaviour" makes `deep_research` look terrible -- it is one cell of +11 against a cell of
-30 and a cell of 0. The same logic makes `detailed` look terrible to the responder. Relative
overgeneralization pushes both toward the safe `(skim, brief)` plateau. The payoff matrix was
engineered to trip up naive multi-agent learning, and it does.

## Independent Q-learning (IQL): the decentralised baseline

`train_independent_q_learning` in `src/learning_agents/marl.py` is the decentralised learner. Each
agent keeps its *own* action-value vector over only its *own* actions, both act epsilon-greedily
(with exploration rate `epsilon`), both receive the shared reward `r`, and -- the defining move --
each agent updates only its own table:

```text
q0[a0] <- q0[a0] + alpha * ( r - q0[a0] )      # researcher updates only its row-action value
q1[a1] <- q1[a1] + alpha * ( r - q1[a1] )      # responder updates only its column-action value
```

Here `q0` and `q1` are the two agents' independent value vectors, `a0` and `a1` are the actions
they chose this round, `alpha` is the step size, and `r` is the shared team reward. There is no
`gamma` term because the game is single-shot (one decision, then reward), so the TD target is just
`r`. Crucially, neither agent's table is indexed by the *other's* action -- that is what "treats
the partner as part of the environment" means in code, and it is exactly the ingredient that
produces relative overgeneralization.

The result is in `artifacts/marl/coordination_comparison.csv`:

- final joint action: `skim+brief`
- final team reward: `5.0`
- coordination success rate: `0.0`
- optimal team reward: `11.0`

The two learners miscoordinate to the safe equilibrium every time. The convergence trace in
`artifacts/marl/training_curves.csv` makes the trap vivid: at episode 80 the greedy joint action
briefly *touches* the optimum at `greedy_team_reward = 11.0`, but it cannot hold there -- by
episode 160 it has collapsed to `5.0` and stays parked at `5.0` (with occasional dips to the
`6.0` cell) all the way to episode 4000. The agents found the peak, could not coordinate to stay on
it, and retreated to the plateau. That collapse is non-stationarity and relative overgeneralization
made visible.

## Joint-action learning (JAL): the centralised contrast

`train_joint_action_learner` in `src/learning_agents/marl.py` changes one thing: instead of two
separate tables, a *single* learner holds a value for every **joint** action `(a0, a1)` and acts
epsilon-greedily over the whole grid. Its update is

```text
joint_q[a0][a1] <- joint_q[a0][a1] + alpha * ( r - joint_q[a0][a1] )
```

where `joint_q[a0][a1]` is the value of the *pair* of actions. Because the value is keyed on the
joint action, there is no averaging over the partner's behaviour -- the learner sees the true value
of the risky optimum `(deep_research, detailed) = 11` directly, distinct from the adjacent -30
cells. This is the simplest possible form of **centralised training**: learn over the joint action
space with full visibility of what everyone did.

The result, again from `artifacts/marl/coordination_comparison.csv`:

- final joint action: `deep_research+detailed`
- final team reward: `11.0`
- coordination success rate: `1.0`
- optimal team reward: `11.0`

The training trace confirms it: in `artifacts/marl/training_curves.csv` the joint learner reports
`greedy_team_reward = 11.0` at *every* checkpoint from episode 80 through 4000 -- it finds the
optimum early and never leaves. On the same game, IQL almost never coordinates to the optimum and
JAL almost always does. That is the whole point of Lane C in one comparison.

## Side-by-side

| learner | locus | final joint action | final team reward | optimal | coordination success rate |
| --- | --- | --- | --- | --- | --- |
| independent (IQL) | decentralised | skim+brief | 5.0 | 11.0 | 0.0 |
| joint (JAL) | centralised | deep_research+detailed | 11.0 | 11.0 | 1.0 |

Source: `artifacts/marl/coordination_comparison.csv`.

The honest tension to surface: JAL "wins" here only because the joint table is tiny. With two
agents of three actions each it has nine joint entries. The joint action space grows
*multiplicatively* with the number of agents -- `k` agents with `m` actions each gives `m^k` joint
actions -- so a literal joint table is exponential and does not scale. JAL is not the deployable
answer; it is the proof that *information* about the partner is what independent learning lacks.

## CTDE: the bridge between them

The two learners above are the extremes of a spectrum, and they trade off against each other:

- **IQL** scales perfectly (each agent is small and decentralised) but miscoordinates, because no
  agent can see the joint structure.
- **JAL** coordinates perfectly but does not scale, because the joint table is exponential.

**Centralised training, decentralised execution (CTDE)** is the bridge that keeps the good half of
each. The idea: *during training* allow centralised information -- the other agents' actions,
observations, or a value function defined over the joint action -- so learners can see the joint
structure and avoid relative overgeneralization. But *at execution time* each agent acts using only
its own local policy `pi_i(a|s)`, with no need for the joint table. You pay the coordination cost
once, in the lab, and ship cheap decentralised policies. The JAL run here is the most literal
("training is the full joint table") sketch of the "centralised training" half; production CTDE
methods such as QMIX, MAPPO, and COMA replace the exponential joint table with a learned, factored
joint value function or a centralised critic, recovering most of JAL's coordination at IQL-like
execution cost. This showcase stops at the JAL contrast on purpose -- it is the minimal artifact
that shows *why* the centralised half of CTDE is worth its complexity.

## What to take away

- Adding agents adds a new failure mode, not just more of the same. **Non-stationarity** (each agent
  is a moving target) and **relative overgeneralization** (averaging the partner's exploration drags
  down risky-but-optimal actions) are specific to the multi-agent locus and have no single-agent
  analogue.
- The cooperative Climbing game makes both effects measurable in nine numbers. Its payoff shape
  rewards a coordinated bet and punishes a half-coordinated one, which is exactly what naive
  independence cannot handle.
- The decentralised-vs-centralised gap is stark and reproducible: IQL lands at `skim+brief` (team
  reward `5.0`, success `0.0`) while JAL reaches `deep_research+detailed` (team reward `11.0`,
  success `1.0`).
- The resolution is not "always centralise." It is **CTDE**: use centralised information to *learn*
  coordination, then *execute* with cheap local policies. JAL is the toy that motivates it; QMIX,
  MAPPO, and COMA are how it is done at scale.

## See also

- [Locus of learning](locus-of-learning.md) -- the map of where learning can live in an agent
  system; Lane C is locus C.
- [The RL ladder](rl-ladder.md) -- the value-iteration and TD foundations these MARL learners are
  built from.
