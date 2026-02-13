from credit_risk_capstone.data import (
    build_target_from_status,
    clean_and_encode_features,
    make_credit_risk_dataset,
)


def test_dataset_has_expected_columns_and_target() -> None:
    bundle = make_credit_risk_dataset(n_samples=240, random_state=7)
    assert "loan_status" in bundle.frame.columns
    target = build_target_from_status(bundle.frame)
    assert set(target.unique()).issubset({0, 1})
    diagnostics, model_frame = clean_and_encode_features(bundle.frame)
    assert "loan_status" not in diagnostics.columns
    assert model_frame.shape[0] == diagnostics.shape[0]
    assert model_frame.shape[1] > diagnostics.shape[1]
