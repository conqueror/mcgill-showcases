"""Tune a toy language model's weights from preferences: RLHF and DPO (locus of learning B).

What + why: the other lanes learn the agent's *orchestration policy* (which tool to call). This lane
is the other major place learning can live -- the **LLM weights themselves**. Here a tiny, fully
tabular "language model" generates a response to a prompt, and we improve *the weights that pick the
response* from human/AI **preferences** rather than from a hand-written reward. This is the family
behind instruction-tuned and reasoning models: RLHF, DPO, GRPO, RLVR.

This module implements the two classic members on a toy you can run on a CPU in milliseconds:

* **RLHF** (reinforcement learning from human feedback) in its original two-stage form: first fit a
  **reward model** to the preference pairs (Bradley-Terry), then improve the policy by
  reward-maximising policy gradient with a KL penalty back to the reference policy (so it does not
  drift far from the pretrained behaviour). Reward model + RL.
* **DPO** (direct preference optimization): skip the reward model entirely and optimise the *policy*
  directly on the same preference pairs with a logistic loss. Same data, no separate reward model,
  no sampling -- DPO is "RLHF without the RM".

The "language model" is a per-prompt softmax over a small response vocabulary (the logits are its
"weights"); a fixed quality matrix plays the role of the latent human judgement, and preferences are
derived from it. Everything is deterministic and tabular so the *mechanisms* -- the Bradley-Terry
loss, the KL-penalised policy gradient, the DPO logistic loss -- are visible without a GPU, a neural
net, or any network call.

RL concept:
    Locus of learning B -- learning the LLM's weights from preferences. RLHF (reward model + KL-
    penalised policy gradient) versus DPO (direct, reward-model-free). The reference-policy KL is
    the leash that keeps preference-tuning from collapsing the model.

Math:
    Policy: pi_theta(r | p) = softmax(theta[p])[r]; reference pi_ref likewise.
    Bradley-Terry RM: minimise -log sigmoid(r_phi[c] - r_phi[l]) over preferred pairs (c > l).
    RLHF objective: max_theta E_{r~pi}[ r_phi(r) ] - beta * KL(pi || pi_ref).
    DPO loss: -log sigmoid( beta * [ (log pi_theta(c) - log pi_ref(c))
                                     - (log pi_theta(l) - log pi_ref(l)) ] ).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from learning_agents.policy_gradient import softmax

# Toy "language model" task: NUM_PROMPTS prompts, each with a softmax over NUM_RESPONSES responses.
# QUALITY[p][r] is the latent human judgement of response r for prompt p (in [0, 1]); each prompt
# has a single best response. Preferences and the reward model are derived from this matrix; the
# policy never sees it directly. A uniform policy scores ~0.49 here; the optimum scores 1.0.
QUALITY: tuple[tuple[float, ...], ...] = (
    (1.0, 0.7, 0.4, 0.2, 0.1),  # prompt 0: best response is index 0
    (0.4, 1.0, 0.6, 0.3, 0.2),  # prompt 1: best response is index 1
    (0.2, 0.5, 1.0, 0.5, 0.2),  # prompt 2: best response is index 2
    (0.1, 0.3, 0.5, 1.0, 0.6),  # prompt 3: best response is index 3
)
NUM_PROMPTS: int = len(QUALITY)
NUM_RESPONSES: int = len(QUALITY[0])

__all__ = [
    "NUM_PROMPTS",
    "NUM_RESPONSES",
    "QUALITY",
    "PreferenceOptResult",
    "build_preference_pairs",
    "compare_preference_methods",
    "expected_quality",
    "policy_kl",
    "reference_logits",
    "response_is_correct",
    "train_dpo",
    "train_grpo",
    "train_reward_model",
    "train_rlhf",
    "train_rlvr",
    "win_rate_vs_reference",
]


@dataclass(frozen=True)
class PreferenceOptResult:
    """A trained toy-LM policy (per-prompt logits) plus its per-epoch learning curve.

    Attributes:
        logits: The trained policy weights ``theta[p][r]`` -- one logit per (prompt, response).
        training_curve: One dict per epoch with ``epoch``, ``expected_quality`` (the mean latent
            quality of the policy's responses), and ``kl_to_reference`` (how far it has moved from
            the reference policy) -- the preference-tuning analogue of a training curve.

    RL concept: the preference-tuned policy and the quality/KL trade-off it traces while training.
    """

    logits: list[list[float]]
    training_curve: list[dict[str, int | float]]


def reference_logits() -> list[list[float]]:
    """Return the reference (pretrained) policy: a uniform softmax over responses per prompt.

    What + why: RLHF and DPO both regularise toward a *reference* policy (the pretrained model) so
    preference-tuning improves quality without collapsing the distribution. Here the reference is
    uniform (all-zero logits), the maximally uncertain starting point every method is measured
    against.

    Returns:
        A ``NUM_PROMPTS x NUM_RESPONSES`` grid of zeros (uniform softmax per prompt).

    RL concept: the reference policy pi_ref -- the leash for the KL penalty and the DPO baseline.
    """
    return [[0.0] * NUM_RESPONSES for _ in range(NUM_PROMPTS)]


def response_is_correct(prompt: int, response: int) -> bool:
    """Return whether ``response`` is the single best (verifiably correct) one for ``prompt``.

    What + why: a deterministic verifier -- true exactly for the highest-quality response of the
    prompt. RLHF/DPO here use graded preferences, but this binary correctness signal is the hook the
    verifiable-reward methods (RLVR) build on, and a clean way to score final policies.

    Args:
        prompt: Prompt index in ``range(NUM_PROMPTS)``.
        response: Response index in ``range(NUM_RESPONSES)``.

    Returns:
        True iff ``response`` attains the maximum quality for ``prompt``.

    RL concept: a verifiable (binary) reward signal, as opposed to a learned reward model.
    """
    return QUALITY[prompt][response] == max(QUALITY[prompt])


def expected_quality(logits: list[list[float]]) -> float:
    """Compute the policy's mean latent response quality across prompts.

    What + why: the headline metric for this lane -- under the policy ``softmax(logits[p])``, the
    expected quality of the response it would emit, averaged over prompts. Preference-tuning should
    raise this from the uniform reference (~0.49) toward the optimum (1.0) without ever observing
    the quality matrix directly.

    Args:
        logits: Per-prompt policy logits ``theta[p][r]``.

    Returns:
        The mean over prompts of ``sum_r pi(r | p) * QUALITY[p][r]``, rounded to 4 decimals.

    RL concept: the true objective preference-tuning is implicitly optimising via preferences.
    """
    total = 0.0
    for prompt in range(NUM_PROMPTS):
        probabilities = softmax(logits[prompt])
        total += sum(probabilities[r] * QUALITY[prompt][r] for r in range(NUM_RESPONSES))
    return round(total / NUM_PROMPTS, 4)


def policy_kl(logits: list[list[float]], ref_logits: list[list[float]]) -> float:
    """Compute the mean KL divergence ``KL(pi || pi_ref)`` across prompts.

    What + why: measures how far preference-tuning has moved the policy from the reference. RLHF
    penalises this directly; for DPO it is a diagnostic. A method that races ``expected_quality`` up
    while letting KL explode has over-fit the preferences and drifted from the pretrained model.

    Args:
        logits: The (tuned) policy logits.
        ref_logits: The reference policy logits.

    Returns:
        The mean over prompts of ``sum_r pi(r | p) * log(pi(r | p) / pi_ref(r | p))``, rounded to 4
        decimals (>= 0).

    RL concept: the KL leash to the reference policy -- the regulariser at the heart of RLHF/DPO.
    """
    total = 0.0
    for prompt in range(NUM_PROMPTS):
        policy = softmax(logits[prompt])
        ref = softmax(ref_logits[prompt])
        total += sum(
            policy[r] * math.log(policy[r] / ref[r])
            for r in range(NUM_RESPONSES)
            if policy[r] > 0.0
        )
    return round(total / NUM_PROMPTS, 4)


def win_rate_vs_reference(logits: list[list[float]], ref_logits: list[list[float]]) -> float:
    """Compute the probability a policy response out-qualities a reference response.

    What + why: the standard preference-tuning report card -- the (exact, tabular) win rate of the
    tuned policy against the reference under the latent quality judge, averaged over prompts. Ties
    count as half a win. A successful method scores well above 0.5.

    Args:
        logits: The tuned policy logits.
        ref_logits: The reference policy logits.

    Returns:
        The mean over prompts of ``sum_{a,b} pi(a) pi_ref(b) * [quality(a) <=> quality(b)]`` (win=1,
        tie=0.5), rounded to 4 decimals.

    RL concept: win rate against the reference -- how preference-tuned models are actually compared.
    """
    total = 0.0
    for prompt in range(NUM_PROMPTS):
        policy = softmax(logits[prompt])
        ref = softmax(ref_logits[prompt])
        prompt_winrate = 0.0
        for a in range(NUM_RESPONSES):
            for b in range(NUM_RESPONSES):
                if QUALITY[prompt][a] > QUALITY[prompt][b]:
                    outcome = 1.0
                elif QUALITY[prompt][a] == QUALITY[prompt][b]:
                    outcome = 0.5
                else:
                    outcome = 0.0
                prompt_winrate += policy[a] * ref[b] * outcome
        total += prompt_winrate
    return round(total / NUM_PROMPTS, 4)


def build_preference_pairs() -> list[tuple[int, int, int]]:
    """Derive ``(prompt, chosen, rejected)`` preference pairs from the latent quality matrix.

    What + why: stands in for a labelled human-preference dataset. For every prompt and every
    ordered pair of responses whose qualities differ, the higher-quality response is the *chosen*
    one and the lower is *rejected*. This is exactly the supervision RLHF's reward model and DPO
    both consume -- relative judgements, never absolute scores.

    Returns:
        A deterministic list of ``(prompt, chosen, rejected)`` triples with
        ``QUALITY[prompt][chosen] > QUALITY[prompt][rejected]``.

    RL concept: a preference dataset -- the supervision signal for reward modelling and DPO.
    """
    pairs: list[tuple[int, int, int]] = []
    for prompt in range(NUM_PROMPTS):
        for chosen in range(NUM_RESPONSES):
            for rejected in range(NUM_RESPONSES):
                if QUALITY[prompt][chosen] > QUALITY[prompt][rejected]:
                    pairs.append((prompt, chosen, rejected))
    return pairs


def _sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid 1 / (1 + e^-x)."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def train_reward_model(
    *,
    pairs: list[tuple[int, int, int]] | None = None,
    learning_rate: float = 0.5,
    epochs: int = 300,
) -> list[list[float]]:
    """Fit a Bradley-Terry reward model ``r_phi[p][r]`` from preference pairs.

    What + why: stage one of classic RLHF. The reward model learns a scalar score per response such
    that the chosen response scores above the rejected one, by minimising the logistic (Bradley-
    Terry) loss ``-log sigmoid(r_phi[chosen] - r_phi[rejected])`` over the preference pairs. It
    turns *relative* preferences into an *absolute* reward the policy can then be optimised against.
    This is a per-(prompt, response) table here; in practice a neural net sharing the base model.

    Args:
        pairs: Preference triples; defaults to :func:`build_preference_pairs`.
        learning_rate: Gradient-ascent step size on the reward table.
        epochs: Number of full-batch passes over the preference pairs.

    Returns:
        The learned reward table ``r_phi[p][r]`` (mean-centred per prompt for identifiability).

    RL concept: reward modelling from preferences (the Bradley-Terry model) -- RLHF stage one.
    """
    preference_pairs = pairs if pairs is not None else build_preference_pairs()
    reward = [[0.0] * NUM_RESPONSES for _ in range(NUM_PROMPTS)]
    for _epoch in range(epochs):
        gradient = [[0.0] * NUM_RESPONSES for _ in range(NUM_PROMPTS)]
        for prompt, chosen, rejected in preference_pairs:
            margin = reward[prompt][chosen] - reward[prompt][rejected]
            # d/dmargin of -log sigmoid(margin) is -(1 - sigmoid(margin)); ascend log-likelihood.
            grad = 1.0 - _sigmoid(margin)
            gradient[prompt][chosen] += grad
            gradient[prompt][rejected] -= grad
        for prompt in range(NUM_PROMPTS):
            for response in range(NUM_RESPONSES):
                reward[prompt][response] += learning_rate * gradient[prompt][response]
    # Centre each prompt's rewards (a constant shift does not change Bradley-Terry or the policy).
    for prompt in range(NUM_PROMPTS):
        mean = sum(reward[prompt]) / NUM_RESPONSES
        reward[prompt] = [value - mean for value in reward[prompt]]
    return reward


def train_rlhf(
    *,
    reward_model: list[list[float]] | None = None,
    ref_logits: list[list[float]] | None = None,
    beta: float = 0.05,
    learning_rate: float = 0.5,
    epochs: int = 300,
) -> PreferenceOptResult:
    """Improve the policy against a reward model with a KL penalty (classic RLHF stage two).

    What + why: stage two of RLHF. Starting from the reference policy, it ascends the KL-penalised
    objective ``E_{r~pi}[r_phi(r)] - beta * KL(pi || pi_ref)`` by tabular policy gradient. The KL
    penalty is applied the way real RLHF/PPO apply it -- folded into a per-response reward
    ``r_phi(r) - beta * (log pi(r) - log pi_ref(r))`` -- and the gradient uses the policy's own mean
    as the baseline (the score-function estimator ``pi(a) * (reward(a) - mean_reward)``). The KL
    term keeps the tuned policy from collapsing onto the reward model's argmax and drifting off the
    pretrained manifold.

    Args:
        reward_model: The reward table from :func:`train_reward_model`; fit afresh if ``None``.
        ref_logits: Reference policy; defaults to the uniform :func:`reference_logits`.
        beta: KL-penalty weight (the leash strength).
        learning_rate: Policy-gradient step size.
        epochs: Number of full-batch policy-gradient updates.

    Returns:
        A :class:`PreferenceOptResult` with the tuned logits and the per-epoch quality/KL curve.

    RL concept: KL-penalised reward maximisation -- RLHF stage two (reward model + RL).
    """
    reward = reward_model if reward_model is not None else train_reward_model()
    reference = ref_logits if ref_logits is not None else reference_logits()
    logits = [row[:] for row in reference]  # start from the reference (pretrained) policy
    curve: list[dict[str, int | float]] = []
    for epoch in range(1, epochs + 1):
        for prompt in range(NUM_PROMPTS):
            policy = softmax(logits[prompt])
            ref = softmax(reference[prompt])
            # KL-as-reward: penalise moving probability mass away from the reference policy.
            effective = [
                reward[prompt][r] - beta * (math.log(policy[r]) - math.log(ref[r]))
                for r in range(NUM_RESPONSES)
            ]
            baseline = sum(policy[r] * effective[r] for r in range(NUM_RESPONSES))
            for r in range(NUM_RESPONSES):
                # Tabular score-function gradient with the mean reward as the baseline.
                logits[prompt][r] += learning_rate * policy[r] * (effective[r] - baseline)
        curve.append(
            {
                "epoch": epoch,
                "expected_quality": expected_quality(logits),
                "kl_to_reference": policy_kl(logits, reference),
            }
        )
    return PreferenceOptResult(logits=logits, training_curve=curve)


def train_dpo(
    *,
    pairs: list[tuple[int, int, int]] | None = None,
    ref_logits: list[list[float]] | None = None,
    beta: float = 0.1,
    learning_rate: float = 0.5,
    epochs: int = 300,
) -> PreferenceOptResult:
    """Optimise the policy directly on preferences with the DPO loss (no reward model).

    What + why: DPO is "RLHF without the reward model". It optimises the *policy* directly on the
    same preference pairs with the logistic loss
    ``-log sigmoid(beta * [(log pi(c) - log pi_ref(c)) - (log pi(l) - log pi_ref(l))])``, which is
    provably equivalent to the RLHF objective under a Bradley-Terry assumption -- but with no
    reward model, no sampling, and no RL loop. For the tabular softmax the per-logit gradient is
    strikingly simple: the policy probabilities cancel, so each step nudges the chosen logit up and
    the rejected logit down by ``beta * (1 - sigmoid(margin))``, scaled by the learning rate.

    Args:
        pairs: Preference triples; defaults to :func:`build_preference_pairs`.
        ref_logits: Reference policy; defaults to the uniform :func:`reference_logits`.
        beta: The DPO temperature (implicit KL strength).
        learning_rate: Gradient-ascent step size.
        epochs: Number of full-batch passes over the preference pairs.

    Returns:
        A :class:`PreferenceOptResult` with the tuned logits and the per-epoch quality/KL curve.

    RL concept: direct preference optimization -- preference-tuning the policy with no reward model.

    Math:
        d/dtheta[p][c] of the DPO loss simplifies to ``-beta * (1 - sigmoid(margin))`` and the
        rejected logit's gradient is its negation; the per-response softmax terms cancel.
    """
    preference_pairs = pairs if pairs is not None else build_preference_pairs()
    reference = ref_logits if ref_logits is not None else reference_logits()
    logits = [row[:] for row in reference]
    curve: list[dict[str, int | float]] = []
    for epoch in range(1, epochs + 1):
        gradient = [[0.0] * NUM_RESPONSES for _ in range(NUM_PROMPTS)]
        for prompt, chosen, rejected in preference_pairs:
            policy = softmax(logits[prompt])
            ref = softmax(reference[prompt])
            chosen_logratio = math.log(policy[chosen]) - math.log(ref[chosen])
            rejected_logratio = math.log(policy[rejected]) - math.log(ref[rejected])
            margin = beta * (chosen_logratio - rejected_logratio)
            step = beta * (1.0 - _sigmoid(margin))  # ascend the log-likelihood of the preference
            gradient[prompt][chosen] += step
            gradient[prompt][rejected] -= step
        for prompt in range(NUM_PROMPTS):
            for response in range(NUM_RESPONSES):
                logits[prompt][response] += learning_rate * gradient[prompt][response]
        curve.append(
            {
                "epoch": epoch,
                "expected_quality": expected_quality(logits),
                "kl_to_reference": policy_kl(logits, reference),
            }
        )
    return PreferenceOptResult(logits=logits, training_curve=curve)


def train_grpo(
    *,
    ref_logits: list[list[float]] | None = None,
    use_verifier: bool = False,
    group_size: int = 16,
    learning_rate: float = 0.5,
    epochs: int = 300,
    seed: int = 7,
) -> PreferenceOptResult:
    """Improve the policy with Group-Relative Policy Optimization (no value model, no reward model).

    What + why: GRPO is the method behind recent reasoning models. Instead of a learned value
    function (a critic) for the baseline, it samples a *group* of responses per prompt, scores each,
    and uses the group's own mean as the baseline and its standard deviation to normalise -- so the
    advantage of a sampled response is ``(reward - group_mean) / (group_std + eps)``. Responses that
    beat their group's average are made more likely; below-average ones less likely. There is no
    critic and no separate reward model, which is what makes it cheap and stable for large policies.
    Here the score is the latent quality (or, with ``use_verifier``, the binary correctness reward,
    the variant exposed as :func:`train_rlvr`). A small constant is added to the std for safety,
    and the gradient is the standard advantage-weighted score function over the sampled group.

    Args:
        ref_logits: Reference (starting) policy; defaults to the uniform :func:`reference_logits`.
        use_verifier: If True, reward each response by the binary verifier
            (:func:`response_is_correct`) instead of the graded quality -- this is the RLVR variant.
        group_size: Number of responses sampled per prompt per epoch (the group the baseline is
            computed from).
        learning_rate: Policy-gradient step size.
        epochs: Number of sampled-group updates per prompt.
        seed: RNG seed for group sampling, so the run is reproducible.

    Returns:
        A :class:`PreferenceOptResult` with the tuned logits and the per-epoch quality/KL curve.

    RL concept: GRPO -- a critic-free policy gradient whose baseline is the sampled group's mean
    (advantage normalised by the group's spread).

    Math:
        For a sampled group {r_i}: A_i = (reward(r_i) - mean_i reward) / (std_i reward + eps);
        theta[p][a] += lr * mean_i [ A_i * (1[a = r_i] - pi(a | p)) ].
    """
    reference = ref_logits if ref_logits is not None else reference_logits()
    logits = [row[:] for row in reference]
    rng = random.Random(seed)
    curve: list[dict[str, int | float]] = []
    for epoch in range(1, epochs + 1):
        for prompt in range(NUM_PROMPTS):
            policy = softmax(logits[prompt])
            group = rng.choices(range(NUM_RESPONSES), weights=policy, k=group_size)
            rewards = [
                (1.0 if response_is_correct(prompt, r) else 0.0)
                if use_verifier
                else QUALITY[prompt][r]
                for r in group
            ]
            mean = sum(rewards) / group_size
            variance = sum((value - mean) ** 2 for value in rewards) / group_size
            std = math.sqrt(variance)
            gradient = [0.0] * NUM_RESPONSES
            for sampled, reward in zip(group, rewards, strict=True):
                advantage = (reward - mean) / (std + 1e-8)  # group-relative, critic-free baseline
                for a in range(NUM_RESPONSES):
                    gradient[a] += advantage * ((1.0 if a == sampled else 0.0) - policy[a])
            for a in range(NUM_RESPONSES):
                logits[prompt][a] += learning_rate * gradient[a] / group_size
        curve.append(
            {
                "epoch": epoch,
                "expected_quality": expected_quality(logits),
                "kl_to_reference": policy_kl(logits, reference),
            }
        )
    return PreferenceOptResult(logits=logits, training_curve=curve)


def train_rlvr(
    *,
    ref_logits: list[list[float]] | None = None,
    group_size: int = 16,
    learning_rate: float = 0.5,
    epochs: int = 300,
    seed: int = 7,
) -> PreferenceOptResult:
    """Improve the policy with RL from Verifiable Rewards (GRPO on a binary correctness check).

    What + why: RLVR is GRPO whose reward is a *verifier* -- a deterministic, programmatic check of
    correctness -- rather than a learned reward model or graded human preference. This is how
    reasoning models are trained on math/code: a response earns reward 1 only if it is verifiably
    correct, 0 otherwise, and GRPO's group-relative advantage turns that sparse binary signal into a
    learning gradient. Because the reward cannot be gamed the way a learned reward model can, RLVR
    sidesteps reward hacking -- at the cost of needing a verifier at all. This is exactly
    :func:`train_grpo` with ``use_verifier=True``, surfaced as its own entry point for clarity.

    Args:
        ref_logits: Reference (starting) policy; defaults to the uniform :func:`reference_logits`.
        group_size: Responses sampled per prompt per epoch.
        learning_rate: Policy-gradient step size.
        epochs: Number of sampled-group updates per prompt.
        seed: RNG seed for group sampling.

    Returns:
        A :class:`PreferenceOptResult` with the tuned logits and the per-epoch quality/KL curve.

    RL concept: RL from verifiable rewards -- GRPO driven by a programmatic correctness check, the
    reward-hacking-resistant signal behind reasoning models.
    """
    return train_grpo(
        ref_logits=ref_logits,
        use_verifier=True,
        group_size=group_size,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )


def compare_preference_methods(
    *,
    epochs: int = 200,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Train all four preference-tuning methods once and return their comparison + learning curves.

    What + why: the single entry point the Lane B artifacts are built from. It runs RLHF, DPO, GRPO,
    and RLVR from the same uniform reference and reports, per method, the final expected quality,
    win rate against the reference, and KL moved -- plus each method's per-epoch curve. The headline
    lesson: every member of the family lifts the toy policy from the ~0.49 reference to near-optimal
    by different mechanisms (reward model vs reward-model-free vs critic-free vs verifiable reward).

    Args:
        epochs: Training length applied to every method (kept equal for a fair comparison).

    Returns:
        A pair ``(comparison_rows, curve_rows)``. ``comparison_rows`` has one row per method (plus
        the reference) with ``method``, ``expected_quality``, ``win_rate_vs_reference``, and
        ``kl_to_reference``. ``curve_rows`` has one row per (method, epoch) with ``method``,
        ``epoch``, ``expected_quality``, and ``kl_to_reference``.

    RL concept: a like-for-like comparison of the preference-optimisation family (RLHF, DPO, GRPO,
    RLVR) on one toy task.
    """
    reference = reference_logits()
    results = {
        "rlhf": train_rlhf(epochs=epochs),
        "dpo": train_dpo(epochs=epochs),
        "grpo": train_grpo(epochs=epochs),
        "rlvr": train_rlvr(epochs=epochs),
    }
    comparison_rows: list[dict[str, int | float | str]] = [
        {
            "method": "reference",
            "expected_quality": expected_quality(reference),
            "win_rate_vs_reference": win_rate_vs_reference(reference, reference),
            "kl_to_reference": policy_kl(reference, reference),
        }
    ]
    curve_rows: list[dict[str, int | float | str]] = []
    for name, result in results.items():
        comparison_rows.append(
            {
                "method": name,
                "expected_quality": expected_quality(result.logits),
                "win_rate_vs_reference": win_rate_vs_reference(result.logits, reference),
                "kl_to_reference": policy_kl(result.logits, reference),
            }
        )
        for row in result.training_curve:
            curve_rows.append(
                {
                    "method": name,
                    "epoch": row["epoch"],
                    "expected_quality": row["expected_quality"],
                    "kl_to_reference": row["kl_to_reference"],
                }
            )
    return comparison_rows, curve_rows
