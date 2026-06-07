"""Pin preference-tuning of a toy LM's weights: reward modelling, RLHF, and DPO (locus B).

These tests anchor the "learning the LLM weights" lane. They pin that preferences are derived
consistently from the latent quality matrix; that a Bradley-Terry reward model recovers the right
ranking; and that both RLHF (reward model + KL-penalised policy gradient) and DPO (direct, reward-
model-free) lift the toy policy from the uniform reference (~0.49 expected quality) to near-optimal
with a high win rate, while moving a measurable but finite KL from the reference. Everything is
deterministic and tabular, so the mechanisms are checked exactly.

RL concept:
    Locus of learning B -- preference optimisation of the policy weights (RLHF vs DPO).
"""

from __future__ import annotations

from learning_agents.preference_optimization import (
    NUM_PROMPTS,
    NUM_RESPONSES,
    QUALITY,
    build_preference_pairs,
    compare_preference_methods,
    expected_quality,
    policy_kl,
    reference_logits,
    response_is_correct,
    train_dpo,
    train_grpo,
    train_reward_model,
    train_rlhf,
    train_rlvr,
    win_rate_vs_reference,
)

_REFERENCE_QUALITY = 0.49  # uniform policy's mean latent quality on this toy task


def test_reference_policy_and_verifier_baseline() -> None:
    """The uniform reference scores the known baseline and the verifier marks each prompt's best.

    Pins the starting point every method is measured against: the uniform reference policy's mean
    quality is ~0.49, and ``response_is_correct`` is true for exactly the highest-quality response
    of each prompt (the verifiable signal RLVR will later use).
    """
    assert expected_quality(reference_logits()) == _REFERENCE_QUALITY
    for prompt in range(NUM_PROMPTS):
        best = max(range(NUM_RESPONSES), key=lambda r: QUALITY[prompt][r])
        correct = [r for r in range(NUM_RESPONSES) if response_is_correct(prompt, r)]
        assert correct == [best]


def test_preference_pairs_respect_quality_order() -> None:
    """Every derived preference pair prefers the higher-quality response.

    Pins the supervision signal: the dataset is non-empty and each ``(prompt, chosen, rejected)``
    triple satisfies ``QUALITY[chosen] > QUALITY[rejected]`` -- relative judgements, never absolute.
    """
    pairs = build_preference_pairs()
    assert pairs
    for prompt, chosen, rejected in pairs:
        assert QUALITY[prompt][chosen] > QUALITY[prompt][rejected]


def test_reward_model_recovers_the_quality_ranking() -> None:
    """The Bradley-Terry reward model scores each prompt's best response highest.

    Pins RLHF stage one: fit on preferences alone (never the quality matrix), the learned reward
    table's argmax per prompt matches the true best response -- it has turned relative preferences
    into a usable absolute reward.
    """
    reward = train_reward_model()
    for prompt in range(NUM_PROMPTS):
        best = max(range(NUM_RESPONSES), key=lambda r: QUALITY[prompt][r])
        assert reward[prompt].index(max(reward[prompt])) == best


def test_dpo_lifts_quality_and_wins_against_reference() -> None:
    """DPO drives the policy from the uniform reference to near-optimal with a high win rate.

    Pins direct preference optimization: starting from the reference, DPO raises expected quality
    far above the 0.49 baseline, wins well over half its match-ups against the reference, and moves
    a positive (finite) KL -- preference-tuning with no reward model. The quality curve is monotone
    non-decreasing.
    """
    reference = reference_logits()
    result = train_dpo(epochs=300)
    assert expected_quality(result.logits) > 0.9
    assert win_rate_vs_reference(result.logits, reference) > 0.85
    assert policy_kl(result.logits, reference) > 0.0
    quality_curve = [float(row["expected_quality"]) for row in result.training_curve]
    assert quality_curve[-1] > quality_curve[0]
    assert all(
        later >= earlier - 1e-9
        for earlier, later in zip(quality_curve, quality_curve[1:], strict=False)
    )


def test_rlhf_lifts_quality_and_wins_against_reference() -> None:
    """RLHF (reward model + KL-penalised policy gradient) also reaches near-optimal quality.

    Pins the classic two-stage path: fitting a reward model and then ascending the KL-penalised
    objective lifts expected quality far above baseline with a high win rate and a positive KL --
    the same outcome as DPO via a different mechanism.
    """
    reference = reference_logits()
    result = train_rlhf(epochs=300)
    assert expected_quality(result.logits) > 0.9
    assert win_rate_vs_reference(result.logits, reference) > 0.85
    assert policy_kl(result.logits, reference) > 0.0
    quality_curve = [float(row["expected_quality"]) for row in result.training_curve]
    assert quality_curve[-1] > quality_curve[0]


def test_kl_is_zero_against_self_and_positive_after_tuning() -> None:
    """KL to the reference is zero for the reference itself and positive once tuned.

    Pins the KL diagnostic that anchors the regulariser: the reference has zero KL to itself, and a
    DPO-tuned policy has moved a strictly positive distance from it.
    """
    reference = reference_logits()
    assert policy_kl(reference, reference) == 0.0
    tuned = train_dpo(epochs=100)
    assert policy_kl(tuned.logits, reference) > 0.0


def test_preference_tuning_is_deterministic() -> None:
    """Full-batch DPO and RLHF have no RNG: identical inputs reproduce identical policies."""
    assert train_dpo(epochs=50).logits == train_dpo(epochs=50).logits
    assert train_rlhf(epochs=50).logits == train_rlhf(epochs=50).logits


def test_grpo_lifts_quality_with_a_critic_free_baseline() -> None:
    """GRPO improves the policy using the sampled group's mean as the baseline (no critic, no RM).

    Pins the group-relative method: sampling groups and using their mean/std for the advantage lifts
    expected quality far above the 0.49 baseline with a high win rate, and the seeded sampling makes
    the run reproducible.
    """
    reference = reference_logits()
    result = train_grpo(epochs=300, group_size=16, seed=7)
    assert expected_quality(result.logits) > 0.9
    assert win_rate_vs_reference(result.logits, reference) > 0.85
    assert policy_kl(result.logits, reference) > 0.0
    assert train_grpo(epochs=40, seed=7).logits == train_grpo(epochs=40, seed=7).logits


def test_rlvr_lifts_quality_with_a_verifiable_reward() -> None:
    """RLVR reaches near-optimal using only a binary correctness verifier (GRPO + verifier).

    Pins RL-from-verifiable-rewards: a sparse 0/1 correctness signal, turned into a gradient by
    GRPO's group-relative advantage, still drives the policy from the reference to a high win rate,
    the reward-hacking-resistant signal behind reasoning models.
    """
    reference = reference_logits()
    result = train_rlvr(epochs=300, group_size=16, seed=7)
    assert expected_quality(result.logits) > 0.9
    assert win_rate_vs_reference(result.logits, reference) > 0.85


def test_compare_preference_methods_all_beat_reference() -> None:
    """Every method (RLHF, DPO, GRPO, RLVR) beats the reference in the side-by-side comparison.

    Pins the headline Lane B artifact: the comparison reports the reference plus all four methods,
    each with the expected columns, every method clears the reference on expected quality and wins
    more than half its match-ups, and the curves cover all four methods.
    """
    comparison_rows, curve_rows = compare_preference_methods(epochs=150)
    by_method = {str(row["method"]): row for row in comparison_rows}
    assert set(by_method) == {"reference", "rlhf", "dpo", "grpo", "rlvr"}
    assert set(comparison_rows[0]) == {
        "method",
        "expected_quality",
        "win_rate_vs_reference",
        "kl_to_reference",
    }
    reference_quality = float(by_method["reference"]["expected_quality"])
    for method in ("rlhf", "dpo", "grpo", "rlvr"):
        assert float(by_method[method]["expected_quality"]) > reference_quality
        assert float(by_method[method]["win_rate_vs_reference"]) > 0.5
    # Curves cover all four trained methods with the documented schema.
    assert {str(row["method"]) for row in curve_rows} == {"rlhf", "dpo", "grpo", "rlvr"}
    assert set(curve_rows[0]) == {"method", "epoch", "expected_quality", "kl_to_reference"}
