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
