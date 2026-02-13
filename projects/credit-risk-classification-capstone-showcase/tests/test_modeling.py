import pandas as pd
from ml_core.imbalance import resample_binary
from ml_core.splits import build_supervised_split

from credit_risk_capstone.data import (
    build_target_from_status,
    clean_and_encode_features,
    make_credit_risk_dataset,
)
from credit_risk_capstone.modeling import evaluate_imbalance_strategies


def test_imbalance_strategy_eval_runs() -> None:
    bundle = make_credit_risk_dataset(n_samples=500, random_state=11)
    target = build_target_from_status(bundle.frame)
    _, model_frame = clean_and_encode_features(bundle.frame)
    split = build_supervised_split(model_frame, target, strategy="stratified", random_state=11)
    result, _scores, strategy = evaluate_imbalance_strategies(split, random_state=11)

    assert not result.empty
    assert strategy in set(result["strategy"])
    assert result["val_f1"].between(0.0, 1.0).all()


def test_resample_binary_handles_inverted_prevalence() -> None:
    x_train = pd.DataFrame({"feature": list(range(12))})
    y_train = pd.Series([1] * 9 + [0] * 3, name="target")

    x_up, y_up = resample_binary(x_train, y_train, method="upsample_minority", random_state=11)
    up_counts = y_up.value_counts()
    assert int(up_counts[0]) == int(up_counts[1])

    x_down, y_down = resample_binary(
        x_train, y_train, method="downsample_majority", random_state=11
    )
    down_counts = y_down.value_counts()
    assert int(down_counts[0]) == int(down_counts[1])
    assert len(x_up) == len(y_up)
    assert len(x_down) == len(y_down)
