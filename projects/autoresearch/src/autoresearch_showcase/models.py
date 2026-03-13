from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpstreamProfile:
    """Grounded summary of one upstream autoresearch variant."""

    key: str
    display_name: str
    repo: str
    repo_url: str
    repo_commit: str
    hardware: str
    prepare_runtime: str
    attention_backend: str
    compile_policy: str
    training_metric: str
    mutable_surface: str
    fixed_surface: str
    notes: tuple[str, ...]
    preflight_commands: tuple[str, ...]


@dataclass(frozen=True)
class DecisionScenario:
    """One teaching scenario for keep, discard, or crash decisions."""

    name: str
    baseline_bpb: float
    candidate_bpb: float
    complexity_delta: int
    memory_gb: float
    description: str
    crashed: bool = False

    @property
    def delta_bpb(self) -> float:
        """Return candidate minus baseline. Lower is better."""

        return round(self.candidate_bpb - self.baseline_bpb, 6)
