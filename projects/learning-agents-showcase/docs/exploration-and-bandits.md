# Exploration and the Contextual Bandit Warm-up

This is the first rung of the RL ladder. Before we touch transitions, discounting, or
credit assignment, we isolate the one tension that never goes away in reinforcement
learning: **exploration versus exploitation**. A *contextual bandit* is the cleanest
place to meet it, because it strips an agent's decision down to a single shot.

The runnable warm-up lives in `src/learning_agents/bandit.py`, and its two diagnostic
logs are `artifacts/bandit/reward_trace.csv` and `artifacts/bandit/regret_trace.csv`.
Everything below is grounded in that code and those two files.

## A bandit is the horizon-1 special case of the MDP

A full reinforcement-learning problem is a Markov decision process (MDP): the agent is in
a state `s`, picks an action `a` under its policy `pi(a|s)`, receives a reward `r`, and
*moves to a next state* `s-prime` that depends on what it just did, so it must reason about
long-horizon consequences discounted by `gamma` (the discount factor).

A contextual bandit is what you get when you delete the consequences — the MDP with
**horizon 1 and no transitions**:

```text
Full MDP (horizon H):   s_0 --a_0--> s_1 --a_1--> s_2 --...--> s_H     (s-prime depends on a)
Contextual bandit:      x  --a-->  r                                     (no s-prime at all)
```

Each round the learner sees a *context* `x` (the request situation), picks one action,
collects an immediate reward, and the round ends. The next context is drawn independently
of the action just taken, so there is nothing to plan for across rounds. In
`bandit.py` the contexts simply cycle through the environment's fixed scenario catalog:

```text
scenario_t = scenarios[(t - 1) mod len(scenarios)]
```

Because the next `x` does not depend on the chosen action, there is no `gamma`, no
bootstrapping, and no credit-assignment problem. What remains is pure: in this context,
which single action pays best, and how do I find out without wasting too many rounds
trying losers? That last clause is the whole subject of this page.

The bandit reuses the same four orchestration actions and the same scenario catalog as
the multi-step environment the rest of the showcase learns over, so it is genuinely the
one-shot ancestor of the later policies, not a toy on the side.

## The decision problem: context, actions, and a hidden reward model

The four actions are the agent-orchestration moves shared across the showcase, indexed
`0..3`: `answer_direct`, `retrieve`, `clarify`, `escalate`.

The context `x` is an 8-dimensional vector built in `_context_vector`: a leading constant
`1.0` bias term followed by the request's normalized start state (`step`, `intent`,
`difficulty`, `ambiguity`, `evidence`, `attempts`, `budget`), each scaled to `[0, 1]`.
Reusing the MDP's own state encoding is deliberate: the bandit's notion of "context" is
identical to the features the deep-RL rungs consume.

There is a hidden, synthetic ground-truth reward model. Each action `a` has a true weight
vector `w_a` (the constant `CONTEXTUAL_REWARD_WEIGHTS` in the code). The expected reward
of action `a` in context `x` is the logistic-squashed linear score, and the realized
reward is a coin flip with that mean:

```text
score_a(x) = w_a . x
mu_a(x)    = sigma(score_a(x)) = 1 / (1 + exp(-score_a(x)))      expected reward in (0, 1)
R          ~ Bernoulli(mu_a(x))                                   sampled reward, 0 or 1
```

Symbols:
- `x` — the 8-dim context vector for the current round.
- `w_a` — the true (hidden) weight vector for action `a`.
- `sigma(z)` — the logistic sigmoid, `1 / (1 + exp(-z))`, mapping any real score to a
  probability in `(0, 1)`.
- `mu_a(x)` — the **expected** reward of action `a` in context `x`; the mean of the
  Bernoulli draw, not a sampled value.
- `R` — the **realized** reward actually observed this round, either `0` or `1`.

The weights are hand-tuned so the best action genuinely depends on the context:
`answer_direct` wins clean low-difficulty, low-ambiguity asks; `clarify` wins
under-specified (high-ambiguity) ones; `retrieve` wins hard ones that need grounding;
`escalate` is a costly fallback for when difficulty and ambiguity are both high. You can
see this in `artifacts/bandit/reward_trace.csv`: the `optimal_action_label` column is
`answer_direct` for `easy_factual`, `clarify` for `ambiguous_query`, and `retrieve` for
`hard_debug`.

The learner never sees `w_a`. It only sees the rewards it earns and must reconstruct which
action is best per context. The synthetic model exists for one reason: because we know
`mu_a(x)` exactly, we can compute the agent's *regret* against a perfect oracle, which is
the canonical bandit performance metric (defined below).

## Exploration vs exploitation, and the algorithm actually implemented

Every round poses the same dilemma. **Exploit**: play the action your estimates say is
best, banking reward now. **Explore**: try a different action to improve those estimates,
paying a possible short-term cost for better long-term decisions. Pure exploitation can
lock onto a wrong early guess forever; pure exploration never cashes in what it learned.

**What `bandit.py` actually does is epsilon-greedy on per-action ridge-regression
estimates.** Be precise here, because the module's docstring is emphatic about it: this is
**not** LinUCB. There is no upper-confidence-bound term, no optimism-under-uncertainty
bonus added to the predictions. Exploration comes entirely from the `epsilon` coin flip
(plus a one-pull-per-arm warm-up). If you came looking for UCB, it is deliberately absent.

The learner estimates each action's payoff with its own ridge (L2-regularized linear)
regression. Per action `a` it maintains a design matrix `A_a` and a reward-feature vector
`b_a`, updated online after every pull of that arm:

```text
A_a = lambda * I + sum_t x_t x_t^T            (lambda = 1.0, the ridge regularizer)
b_a = sum_t r_t x_t
theta_a = A_a^{-1} b_a                         (solve the ridge normal equations)
estimated payoff of a in context x = theta_a . x
```

Symbols:
- `A_a` — action `a`'s design matrix, seeded as `lambda * I` so it is invertible before
  any data arrives.
- `lambda` — the ridge regularizer, `1.0` here; it shrinks `theta_a` toward zero and keeps
  early estimates stable.
- `I` — the identity matrix.
- `b_a` — the running sum of `reward x context` for the pulls of action `a`.
- `theta_a` — action `a`'s estimated weight vector, the learner's stand-in for the unknown
  `w_a`.
- `x_t`, `r_t` — the context and realized reward at round `t`.

The action rule each round, with exploration rate `epsilon`:

```text
draw u ~ Uniform(0, 1)
if u < epsilon  OR  some arm has never been pulled:
    a_t = uniform random action               (EXPLORE)
else:
    a_t = argmax_a  theta_a . x_t             (EXPLOIT: greedy, no UCB bonus)
```

Symbols:
- `epsilon` — exploration rate in `[0, 1]`; the default run uses `epsilon = 0.2`, so about
  one round in five is a deliberate random probe.
- `a_t` — the action chosen at round `t`.
- `u` — a uniform random draw deciding explore vs exploit.

The forced "pull every arm at least once" warm-up guarantees no action is judged on zero
evidence. The default experiment runs `steps = 600` rounds with `seed = 7`, which makes
the whole trajectory deterministic and reproducible — the explore/exploit behavior is
literally executable documentation.

## Regret: the yardstick

Accuracy on a bandit is measured by **regret** — how much expected reward you forfeited by
not playing the optimal action every round. The oracle's per-round best is the action with
the highest *expected* reward, knowable only because the reward model is synthetic:

```text
a*(x)  = argmax_a mu_a(x)                              the oracle-optimal action in context x
instantaneous_regret_t = mu_{a*}(x_t) - mu_{a_t}(x_t) the expected-reward gap this round
cumulative_regret_T    = sum_{t=1..T} [ mu_{a*}(x_t) - mu_{a_t}(x_t) ]
```

Symbols:
- `a*(x)` — the optimal action for context `x` under the true model.
- `mu_{a*}(x)` — the expected reward of that optimal action (the best achievable mean).
- `mu_{a_t}(x_t)` — the expected reward of the action the learner *actually* chose at
  round `t`.
- `instantaneous_regret_t` — the one-round gap; it is always `>= 0`, and exactly `0`
  whenever the learner picks an optimal action.
- `cumulative_regret_T` — the running total over the first `T` rounds; the headline metric.

Two things are worth internalizing. First, regret is defined on *expected* rewards
`mu`, not on the noisy sampled `R`, so a learner can pick the right action yet still
collect `R = 0` on an unlucky Bernoulli draw — that draw costs nothing in regret. Second,
the goal is **sublinear** cumulative regret: as the learner figures out each context, its
per-round regret should trend toward zero, so the cumulative curve **flattens** instead of
growing as a straight line. A flattening regret curve is the visual signature of
exploration paying off.

## Reading the artifacts

### `artifacts/bandit/reward_trace.csv` — what the learner earned and believed

One row per round (`600` rows). Key columns: the chosen `action_label`, the realized
`reward` (0 or 1), `expected_reward` (the true `mu_{a_t}(x)`), `cumulative_reward`, the
learner's `estimated_value` (`theta_a . x` for the chosen arm), and the oracle's
`optimal_action_label`.

Watch the estimates converge to the truth:

- **Round 1** picks `retrieve` on `easy_factual` and its `estimated_value` is `0.0` —
  the ridge model has seen no data, so it is shrunk to zero and the choice is a blind
  warm-up pull. The oracle action there is `answer_direct`.
- **By the late rounds** the estimates closely track the true means. At round 594
  (`hard_debug`) the learner plays the optimal `retrieve` with `estimated_value` `0.7969`
  against the true `expected_reward` of `0.7408`; at round 595 (`needs_escalation`) it
  plays `clarify`, estimating `0.8531` against the true `0.7858`.

The final `cumulative_reward` after all `600` rounds is **`443.0`**. Because rewards are
Bernoulli, the ceiling is well below `600`: even a perfect oracle averages only
`mu_{a*}(x)` per round (often `0.6` to `0.8` for these contexts), so `443` reflects a
learner that is mostly, but not always, optimal — just what an `epsilon = 0.2` policy that
keeps probing should produce.

### `artifacts/bandit/regret_trace.csv` — the gap to the oracle, flattening over time

One row per round (`600` rows) with `instantaneous_regret` and `cumulative_regret`. This
is the file that shows learning most clearly.

- **Early rounds accumulate regret quickly.** The warm-up and a cold model mean several
  wrong picks up front: by round 5 the cumulative regret is already `1.9959`, having taken
  hits of `0.4704` (round 1), `0.3395` (round 3), `0.5093` (round 4), and `0.6767`
  (round 5).
- **The curve then flattens hard.** Once the per-context winners are learned, most rounds
  add exactly `0.0`. The trace shows long flat plateaus — for example
  `cumulative_regret` sits at `32.5503` unchanged across roughly rounds 460 through 492,
  and again holds at `35.2668` from about round 561 onward.
- **The residual bumps are the price of exploration.** The small non-zero increments late
  in the run (e.g. a `0.0959` step when `needs_escalation` is answered with `retrieve`
  instead of the optimal `clarify`, or `0.4704` when an `epsilon` probe plays `retrieve`
  on `easy_factual`) come from the `epsilon = 0.2` random draws, not from a confused
  model. With a fixed, non-decaying `epsilon`, regret can never reach a perfectly flat
  line — there is always a `0.2` chance of a deliberate suboptimal probe.

The final `cumulative_regret` after `600` rounds is **`36.283`**. Compare the shapes:
roughly `2.0` of regret was spent in the first 5 rounds learning, but the next ~600 rounds
add only about `34` more — and much of that is the unavoidable exploration tax. That
sharply decelerating, flattening curve, against a `cumulative_reward` that keeps climbing
to `443.0`, is the textbook picture of exploration converting into exploitation.

## Bridging to the rest of the ladder

The bandit teaches the agent which single action is best **in a context**. That is the
whole game when actions have no consequences. Real agents are not so lucky: a `clarify`
now changes what the user says next; an early `retrieve` builds evidence that makes a later
`answer_direct` safe. Restoring those dependencies turns the bandit back into a full MDP
and adds exactly the two ingredients we deleted:

1. **State transitions** — `s-prime` now depends on `a`, so the agent must reason about how
   today's action shapes tomorrow's situation.
2. **Long horizons and discounting** — value is a discounted sum of future rewards
   (governed by `gamma`), so we need the action value `Q(s,a)` and the state value `V(s)`,
   and the credit-assignment machinery to learn them.

That is precisely where [the RL ladder](rl-ladder.md) goes next: from this one-shot warm-up
up through value iteration, Q-learning, SARSA, and policy gradients over the very same
actions and scenarios. The explore/exploit instinct carries straight over — `epsilon`-greedy
is the same exploration rule online Q-learning uses. But more exploration is not
automatically better downstream: in the multi-step setting the online `q_learning` policy
over-explores into over-escalation and is governance-rejected, while a careful offline
method nearly matches the planning ceiling. The single-step bandit is just where that
trade-off is easiest to see in isolation.

For the complementary framing — *where* in an agent the learning actually happens, and how
a learned action-value model sits beside hand-written tools and planners — see
[the locus of learning](locus-of-learning.md).

## See also

- [The RL ladder](rl-ladder.md) — the full progression from this bandit up to deep-RL
  control over the same environment.
- [The locus of learning](locus-of-learning.md) — what part of an agent is learned versus
  engineered, and where bandit/RL value estimates fit.
- [Math notes](math-notes.md) — fuller derivations of expected reward, regret, and the
  value functions used across the showcase.
- [Exercises](exercises.md) — practice problems, including varying `epsilon` and reading
  the bandit traces.

## Citations

The exploration-versus-exploitation framing and the regret metric for (contextual) bandits
follow the standard treatment in Sutton and Barto, 2018, and Lattimore and Szepesvari,
2020.
