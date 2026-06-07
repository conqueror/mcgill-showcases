from _pytest.monkeypatch import MonkeyPatch

import adaptive_course_assistant_rl.sarsa as sarsa_module
from adaptive_course_assistant_rl.sarsa import train_sarsa


def test_sarsa_produces_training_rows() -> None:
    result = train_sarsa(episodes=10, seed=9)

    assert len(result.training_curve) == 10
    assert all("total_reward" in row for row in result.training_curve)


def test_sarsa_does_not_sample_next_action_after_terminal_step(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    def fake_epsilon_greedy_action(
        action_values: list[float],
        epsilon: float,
        rng: object,
    ) -> int:
        nonlocal calls
        del action_values, epsilon, rng
        calls += 1
        return 0

    monkeypatch.setattr(sarsa_module, "_epsilon_greedy_action", fake_epsilon_greedy_action)

    sarsa_module.train_sarsa(episodes=3, horizon=1, seed=9)

    assert calls == 3
