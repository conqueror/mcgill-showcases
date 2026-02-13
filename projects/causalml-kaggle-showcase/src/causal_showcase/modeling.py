from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from causalml.inference.meta import (
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
)
from causalml.inference.tree import UpliftTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from .data import PreparedData


class _MetaLearnerProtocol(Protocol):
    def fit(
        self,
        *,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        p: np.ndarray | None = ...,
    ) -> object: ...

    def predict(self, *, X: np.ndarray, p: np.ndarray | None = ...) -> np.ndarray: ...

    def estimate_ate(
        self,
        *,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        p: np.ndarray | None = ...,
    ) -> object: ...


@dataclass(frozen=True)
class LearnerResult:
    learner_name: str
    ate: float
    ate_ci_low: float
    ate_ci_high: float
    uplift_scores: np.ndarray


@dataclass(frozen=True)
class UpliftTreeResult:
    uplift_scores: np.ndarray
    tree_summary: str


def _flatten_predictions(pred: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0]
    return arr.reshape(arr.shape[0], -1)[:, 0]


def _estimate_ate_with_ci(
    model: _MetaLearnerProtocol,
    X: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    p: np.ndarray | None = None,
) -> tuple[float, float, float]:
    raw_result: object
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        if p is None:
            raw_result = model.estimate_ate(X=X, treatment=treatment, y=y)
        else:
            raw_result = model.estimate_ate(X=X, treatment=treatment, y=y, p=p)

    if isinstance(raw_result, tuple):
        if len(raw_result) == 3:
            ate_raw, lb_raw, ub_raw = raw_result
        else:
            ate_raw = raw_result[0]
            lb_raw = np.array([np.nan])
            ub_raw = np.array([np.nan])
    else:
        ate_raw = raw_result
        lb_raw = np.array([np.nan])
        ub_raw = np.array([np.nan])

    ate_val = float(np.asarray(ate_raw).reshape(-1)[0])
    lb_val = float(np.asarray(lb_raw).reshape(-1)[0])
    ub_val = float(np.asarray(ub_raw).reshape(-1)[0])
    return ate_val, lb_val, ub_val


def fit_meta_learners(
    train_data: PreparedData,
    test_data: PreparedData,
) -> dict[str, LearnerResult]:
    """
    Train S/T/X/R learners and return ATE + individual uplift estimates.

    We use the same random forest base learner so differences across methods
    reflect the meta-learner strategy, not base model choice.
    """
    x_train = train_data.X.to_numpy()
    x_test = test_data.X.to_numpy()
    w_train = train_data.treatment
    y_train = train_data.outcome

    propensity_model = LogisticRegression(max_iter=2000)
    propensity_model.fit(x_train, w_train)
    p_train = propensity_model.predict_proba(x_train)[:, 1]
    p_test = propensity_model.predict_proba(x_test)[:, 1]

    def _rf() -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=300, random_state=42)

    learners: dict[str, _MetaLearnerProtocol] = {
        "S-Learner": cast(_MetaLearnerProtocol, BaseSRegressor(learner=_rf(), control_name=0)),
        "T-Learner": cast(_MetaLearnerProtocol, BaseTRegressor(learner=_rf(), control_name=0)),
        "X-Learner": cast(_MetaLearnerProtocol, BaseXRegressor(learner=_rf(), control_name=0)),
        "R-Learner": cast(_MetaLearnerProtocol, BaseRRegressor(learner=_rf(), control_name=0)),
    }

    results: dict[str, LearnerResult] = {}
    for name, typed_learner in learners.items():
        if name in {"X-Learner", "R-Learner"}:
            typed_learner.fit(X=x_train, treatment=w_train, y=y_train, p=p_train)
            uplift_scores = _flatten_predictions(typed_learner.predict(X=x_test, p=p_test))
            ate, lb, ub = _estimate_ate_with_ci(
                typed_learner,
                X=x_train,
                treatment=w_train,
                y=y_train,
                p=p_train,
            )
        else:
            typed_learner.fit(X=x_train, treatment=w_train, y=y_train)
            uplift_scores = _flatten_predictions(typed_learner.predict(X=x_test))
            ate, lb, ub = _estimate_ate_with_ci(
                typed_learner,
                X=x_train,
                treatment=w_train,
                y=y_train,
            )

        results[name] = LearnerResult(
            learner_name=name,
            ate=ate,
            ate_ci_low=lb,
            ate_ci_high=ub,
            uplift_scores=uplift_scores,
        )

    return results


def fit_uplift_tree(train_data: PreparedData, test_data: PreparedData) -> UpliftTreeResult:
    """Train KL-based uplift tree and return uplift scores for the test set."""
    train_treatment = np.where(train_data.treatment == 1, "ad", "control")

    model = UpliftTreeClassifier(
        max_depth=4,
        min_samples_leaf=500,
        min_samples_treatment=250,
        n_reg=100,
        evaluationFunction="KL",
        control_name="control",
    )
    model.fit(
        train_data.X.to_numpy(),
        treatment=train_treatment,
        y=train_data.outcome,
    )

    predictions = model.predict(test_data.X.to_numpy())
    uplift_scores = _flatten_predictions(predictions)

    tree_summary = str(model.fitted_uplift_tree)
    return UpliftTreeResult(uplift_scores=uplift_scores, tree_summary=tree_summary)
