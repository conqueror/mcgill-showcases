from __future__ import annotations

from .models import DecisionScenario


def recommend_status(
    baseline_bpb: float,
    candidate_bpb: float,
    complexity_delta: int,
    crashed: bool = False,
) -> tuple[str, str]:
    """Recommend whether to keep, discard, or log a crash.

    `complexity_delta` is relative to the starting point:

    - positive: more complex
    - zero: similar complexity
    - negative: simpler
    """

    if crashed:
        return "crash", "The run crashed, so it should be logged but must not advance the branch."

    delta_bpb = round(candidate_bpb - baseline_bpb, 6)
    if delta_bpb < -0.001 and complexity_delta <= 3:
        return "keep", "Meaningful val_bpb improvement justifies the added complexity."
    if delta_bpb < 0 and complexity_delta <= 1:
        return "keep", "The model improved and complexity stayed controlled."
    if abs(delta_bpb) <= 0.0002 and complexity_delta < 0:
        return "keep", "Performance is effectively flat while the code became simpler."
    return "discard", "The improvement is too small, negative, or too costly in complexity."


def default_scenarios() -> tuple[DecisionScenario, ...]:
    """Return a small teaching set of simulated experiment outcomes."""

    return (
        DecisionScenario(
            name="lower_learning_rate",
            baseline_bpb=0.9989,
            candidate_bpb=0.9974,
            complexity_delta=0,
            memory_gb=44.1,
            description="Lower the learning rate slightly without changing the model shape.",
        ),
        DecisionScenario(
            name="deeper_model",
            baseline_bpb=0.9974,
            candidate_bpb=0.9962,
            complexity_delta=3,
            memory_gb=46.5,
            description=(
                "Increase depth and keep the rest of the training recipe close "
                "to baseline."
            ),
        ),
        DecisionScenario(
            name="activation_swap",
            baseline_bpb=0.9962,
            candidate_bpb=1.0011,
            complexity_delta=1,
            memory_gb=46.6,
            description="Swap activations in a way that hurts validation loss.",
        ),
        DecisionScenario(
            name="simplify_attention_pattern",
            baseline_bpb=0.9962,
            candidate_bpb=0.9962,
            complexity_delta=-2,
            memory_gb=43.8,
            description=(
                "Remove a more complicated pattern while preserving roughly the "
                "same score."
            ),
        ),
        DecisionScenario(
            name="double_width_oom",
            baseline_bpb=0.9962,
            candidate_bpb=0.0,
            complexity_delta=4,
            memory_gb=0.0,
            description="Double model width and run out of memory.",
            crashed=True,
        ),
    )
