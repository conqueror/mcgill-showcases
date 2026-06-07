# Lane B: Learning the LLM Weights (RLHF, DPO, GRPO, RLVR)

Most of this showcase learns an agent's **orchestration policy** -- which tool to
call, when to escalate, how hard to try. That is one locus of learning. This page
is about the *other* major one: the **token policy itself**, the weights inside
the language model that decide which response to emit. When people say a model was
"instruction-tuned", "aligned", or "trained to reason", they mean Lane B -- the
weights were moved by reinforcement learning from preferences or from a verifier.

This page walks the four methods behind modern instruction-tuned and reasoning
models -- RLHF, DPO, GRPO, RLVR -- as one family with one shared idea, each with
its objective written out and every symbol defined. The code lives in
`src/learning_agents/preference_optimization.py`; the numbers come from
`artifacts/preference/method_comparison.csv` and
`artifacts/preference/training_curves.csv`.

A blunt warning up front, repeated at the end because it matters: the "language
model" here is a **toy 4x5 quality matrix**, not a transformer. There is no text,
no tokenizer, no GPU, no neural net. Four "prompts", five "responses" each, a
softmax over five logits per prompt. We do this on purpose. At this scale the
Bradley-Terry loss, the KL-penalised policy gradient, and the DPO logistic loss
are *visible* -- you can read every gradient. What transfers is the **mechanism
and the relationships between methods**, not the scale.

## The setup: a tabular "language model"

For each prompt `p`, the policy is a softmax over a small response vocabulary:

```text
pi_theta(r | p) = softmax(theta[p])[r]
```

Symbols:

- `pi_theta(r | p)` -- the policy: probability of emitting response `r` for prompt
  `p`. This is the thing we are training.
- `theta[p]` -- the **weights** (logits) for prompt `p`. In a real LM these are
  billions of transformer parameters; here it is a length-5 vector per prompt.
- `pi_ref(r | p)` -- the **reference policy**: the pretrained model we start from
  and must not stray too far from. Here `pi_ref` is uniform (all-zero logits), the
  maximally uncertain starting point. See `reference_logits` in the module.

A hidden quality matrix `QUALITY[p][r]` in `[0, 1]` plays the role of latent human
judgement; each prompt has one best response. The policy **never sees this matrix
directly** -- it only ever sees *preferences derived from it* (or, for the
verifier methods, a binary correctness check). The headline metric is expected
quality under the policy:

```text
expected_quality(theta) = mean_p  sum_r  pi_theta(r | p) * QUALITY[p][r]
```

A uniform policy scores about `0.49` on this matrix; the optimum is `1.0`. Every
method below has the same job: push that number up using only relative judgements,
without ever reading `QUALITY`.

## The shared idea: maximise reward, but stay near the reference

All four methods optimise the same underlying objective. Maximise expected reward,
minus a penalty for drifting from the reference policy:

```text
max_theta   E_{r ~ pi_theta}[ R(r) ]  -  beta * KL( pi_theta || pi_ref )
```

Symbols:

- `R(r)` -- a reward for response `r`. The methods differ entirely in **where `R`
  comes from**: a learned reward model (RLHF), an implicit reward folded into the
  loss (DPO), the response's own group-relative standing (GRPO), or a programmatic
  verifier (RLVR).
- `beta` -- the **KL weight**, the strength of the leash. Larger `beta` keeps the
  policy closer to `pi_ref`; smaller `beta` lets it chase reward harder.
- `KL(pi_theta || pi_ref)` -- the Kullback-Leibler divergence from the tuned
  policy to the reference, defined per prompt as
  `sum_r pi_theta(r | p) * log( pi_theta(r | p) / pi_ref(r | p) )` and averaged
  over prompts. It is `>= 0`, and `0` only when the two policies are identical.
  See `policy_kl` in the module.

**Why the KL leash is the whole game.** Without it, reward maximisation collapses
the policy onto the single argmax response and throws away everything the
pretrained model knew -- fluency, calibration, the diversity that makes a model
useful. The KL term is what keeps preference-tuning from devouring the model it
started from. Every plot in this lane reports KL alongside quality precisely so
you can watch the leash hold: all four methods finish near-optimal at a
*controlled* KL of about `1.6`, not at a blown-up one. A method that races quality
up while letting KL explode has overfit its reward and drifted off the pretrained
manifold -- which, in a real model, shows up as repetition, mode collapse, and
confident nonsense.

## RLHF: reward model plus KL-penalised policy gradient

Classic RLHF (the original ChatGPT-style recipe) is **two stages**.

**Stage one: fit a reward model from preferences (Bradley-Terry).** Humans rarely
give absolute scores; they say "this answer is better than that one". The
Bradley-Terry model turns such pairwise judgements into a scalar reward `r_phi`.
For every preferred pair (`c` chosen over `l` rejected) we minimise:

```text
L_RM(phi) = - sum_{(c, l)}  log sigma( r_phi(c) - r_phi(l) )
```

Symbols:

- `r_phi(r)` -- the learned reward (a scalar per response), parameters `phi`. Here
  it is a per-(prompt, response) table; in practice a neural net sharing the base
  model's body. See `train_reward_model`.
- `sigma(x) = 1 / (1 + e^-x)` -- the logistic sigmoid; `sigma(r_phi(c) - r_phi(l))`
  is the model's predicted probability that `c` beats `l`.
- `(c, l)` -- a (chosen, rejected) pair from the preference dataset
  (`build_preference_pairs`), where `QUALITY[p][c] > QUALITY[p][l]`.

Minimising this loss makes the chosen response score above the rejected one. The
reward model has converted *relative* preferences into an *absolute* reward the
policy can be optimised against.

**Stage two: improve the policy against the reward, with a KL penalty.** Starting
from `pi_ref`, we ascend the shared objective with `R = r_phi`. The KL penalty is
applied the way real PPO-style RLHF applies it -- folded into a per-response
effective reward:

```text
R_eff(r) = r_phi(r) - beta * ( log pi_theta(r | p) - log pi_ref(r | p) )
```

and the policy is updated by a score-function (REINFORCE-style) gradient that uses
the policy's own mean reward as a baseline:

```text
theta[p][r]  +=  alpha * pi_theta(r | p) * ( R_eff(r) - baseline_p )
baseline_p   =  sum_r pi_theta(r | p) * R_eff(r)
```

Symbols:

- `alpha` -- the step size (learning rate).
- `baseline_p` -- the mean effective reward under the current policy for prompt
  `p`; subtracting it reduces gradient variance without biasing the update.

The KL term keeps the policy from collapsing onto the reward model's argmax. In
the curve (`artifacts/preference/training_curves.csv`) RLHF jumps fastest of all
four early -- by epoch 1 it is already at quality `0.6514` with KL `0.1405`,
because it optimises a dense, fully-known reward table rather than waiting on
sampling. By epoch 200 it reaches quality `0.9994` at KL `1.5995`. See
`train_rlhf`.

## DPO: the same objective, with the reward model deleted

RLHF's reward model is expensive and fragile -- it is a second model to train,
serve, and (as Lane B's sibling page on reward hacking shows) *exploit*. Direct
Preference Optimization (Rafailov and colleagues, 2023) makes a sharp observation:
under the Bradley-Terry assumption, the KL-penalised objective has a *closed-form*
optimal policy, and you can invert it to express the reward in terms of the policy
itself. Substitute that back into the Bradley-Terry loss and the reward model
disappears. You optimise the **policy directly** on the preference pairs:

```text
L_DPO(theta) = - log sigma( beta * [ ( log pi_theta(c) - log pi_ref(c) )
                                     - ( log pi_theta(l) - log pi_ref(l) ) ] )
```

Symbols are as above; `c` and `l` are the chosen and rejected responses, and
`beta` now plays the role of the implicit KL strength (the temperature of the
implied reward). The bracketed quantity is the difference of **log-ratios** to the
reference -- DPO's implicit reward is `beta * log( pi_theta(r) / pi_ref(r) )`.

DPO is "RLHF without the reward model": same data, same objective, but **no
separate reward model, no sampling, and no RL loop** -- just a supervised logistic
loss. For the tabular softmax the per-logit gradient is strikingly clean (the
softmax terms cancel): each step nudges the chosen logit up and the rejected logit
down by `beta * (1 - sigma(margin))`, scaled by `alpha`. See `train_dpo`.

The trade-off shows in the curve. DPO starts *gently* -- epoch 1 is quality
`0.5106` at KL `0.0024`, far slower off the line than RLHF -- because its gradient
is throttled by `beta` and the still-small margins. But it climbs steadily and
ends in a dead heat: quality `0.9995`, KL `1.5986` at epoch 200, the best final
quality of the four. Simpler machinery, same destination.

## GRPO: critic-free, group-relative advantage

Both methods above lean on extra machinery -- RLHF on a reward model, and standard
policy-gradient RL on a *value model* (a critic) to compute the baseline. Group
Relative Policy Optimization (Shao and colleagues, 2024), the method behind recent
reasoning models, throws out **both**. It replaces the critic with the group
itself: for each prompt, sample a *group* of responses from the current policy,
score each, and use the group's own statistics as the baseline.

The advantage of a sampled response is its reward standardised within the group:

```text
A_i = ( R(r_i) - mean_j R(r_j) ) / ( std_j R(r_j) + eps )
```

and the policy update is the advantage-weighted score function over the group:

```text
theta[p][a]  +=  alpha * mean_i [ A_i * ( 1[a = r_i] - pi_theta(a | p) ) ]
```

Symbols:

- `R(r_i)` -- the reward of the `i`-th sampled response. In plain GRPO this is the
  graded quality; in RLVR (next section) it is a binary verifier.
- `A_i` -- the **group-relative advantage**: how far response `i` beats its
  group's mean reward, in units of the group's spread. Above-average responses get
  `A_i > 0` (made more likely); below-average get `A_i < 0` (made less likely).
- `mean_j` / `std_j` -- the mean and standard deviation of rewards **over the
  sampled group**, not over a learned value function. This is the critic-free
  baseline.
- `eps` -- a tiny constant (`1e-8`) added to the std for numerical safety.
- `1[a = r_i]` -- the indicator that action `a` is the sampled response `r_i`.

Subtracting the group mean centres the advantage; dividing by the group std
normalises it. No critic, no reward model -- which is exactly what makes GRPO cheap
and stable enough to train very large policies. See `train_grpo`.

The cost of sampling shows up as **noise** in the curve. Unlike RLHF's and DPO's
glassy-smooth ascents, GRPO's curve has flat plateaus where several consecutive
epochs report the identical quality (for example epochs 116 through 131 all sit at
quality `0.9978`, KL `1.58`) -- stretches where the sampled group happened not to
move the logits. It still arrives: quality `0.9988`, KL `1.593` at epoch 200.

## RLVR: GRPO with a reward you cannot fake

RL from Verifiable Rewards is **GRPO with one change**: the reward `R(r)` is a
deterministic, programmatic **verifier** instead of a learned reward model or a
graded preference. A response earns reward `1` only if it is verifiably correct,
`0` otherwise:

```text
R(r) = 1  if response_is_correct(p, r)  else  0
```

Everything else -- the group sampling, the group-relative advantage, the update --
is identical to GRPO. In the code, RLVR is literally `train_grpo` with
`use_verifier=True`, surfaced as `train_rlvr` for clarity. The verifier here is
`response_is_correct`, true exactly for the single highest-quality response.

This is how reasoning models are trained on math and code: run the code, check the
proof, grade the answer -- reward `1` for correct, `0` for wrong -- and let GRPO's
group-relative advantage turn that sparse binary signal into a gradient. The
decisive property is that **a verifier cannot be gamed the way a learned reward
model can.** A reward model is itself a learned approximation with exploitable
seams; a unit test either passes or it does not. RLVR therefore sidesteps reward
hacking -- at the price of needing a real verifier, which only exists for
checkable domains. The reward-hacking page picks up exactly this tension.

On the toy, RLVR tracks GRPO almost step for step (same sampling, same plateaus)
and finishes at quality `0.9988`, KL `1.5927` -- effectively tied with GRPO, as
expected when the verifier and the graded reward agree on which response is best.
See `train_rlvr`.

## The comparison: four routes, one destination

All numbers below are from `artifacts/preference/method_comparison.csv`, produced
by `compare_preference_methods` with every method trained for the same 200 epochs
from the same uniform reference (a fair, like-for-like run).

| method      | expected_quality | win_rate_vs_reference | kl_to_reference |
|-------------|------------------|-----------------------|-----------------|
| `reference` | 0.49             | 0.5                   | 0.0             |
| `rlhf`      | 0.9994           | 0.8996                | 1.5995          |
| `dpo`       | 0.9995           | 0.8997                | 1.5986          |
| `grpo`      | 0.9988           | 0.8991                | 1.593           |
| `rlvr`      | 0.9988           | 0.8990                | 1.5927          |

How to read this table:

- **All four lift quality from `0.49` to about `0.999`.** Four different
  mechanisms -- learned reward model, reward-model-free direct loss, critic-free
  group baseline, verifiable reward -- arrive at essentially the same near-optimal
  policy. That convergence is the lesson: these are not rival tricks but four
  routes through the *same* KL-penalised objective.
- **Win rate against the reference is about `0.90` for every method.** The tuned
  policy beats a uniform draw roughly nine times in ten (ties counting as half a
  win, per `win_rate_vs_reference`). This is the report card real
  preference-tuned models are graded on.
- **KL is controlled, about `1.6` for all four.** The leash held. Nobody blew up
  the divergence to grab the last sliver of quality. On this toy the methods differ
  more in their *training dynamics* -- RLHF's fast smooth start, DPO's gentle
  steady climb, GRPO's and RLVR's noisy sampled ascents -- than in their final
  resting point.

## Honest limits

Worth stating plainly, because it is the easiest thing to oversell:

- **This is a toy, full stop.** A 4x5 quality matrix and a per-prompt softmax over
  five logits. No text, no tokens, no transformer, no GPU. The whole comparison
  runs on a CPU in milliseconds.
- **What transfers is the math and the relationships, not the scale.** You should
  leave able to write down each objective, say where each method's reward comes
  from, and explain why the KL leash exists. You should *not* leave thinking these
  exact quality and KL numbers say anything about a real model -- they are
  properties of this matrix and these hyperparameters.
- **The near-ties are an artifact of an easy problem.** On a clean toy where the
  graded reward and the verifier agree, RLHF, DPO, GRPO, and RLVR land in the same
  place. At real scale they diverge sharply in compute, stability, susceptibility
  to reward hacking, and what domains they even apply to -- which is the whole
  reason all four exist.

## See also

- [Locus of learning](locus-of-learning.md) -- the map of *where* learning can
  live; this page is locus B (the token policy), the orchestration lanes are the
  others.
- [Reward design and hacking](reward-design-and-hacking.md) -- why a learned
  reward model is exploitable and a verifier is not, the tension RLVR is built
  around.
- [Math notes](math-notes.md) -- fuller derivations: Bradley-Terry, the
  KL-penalised objective, and how DPO's loss falls out of it.
- [Start here](00-start-here.md) -- the showcase overview and the rest of the
  learning path.
