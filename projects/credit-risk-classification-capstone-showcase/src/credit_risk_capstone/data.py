from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_STATUSES = {
    "Default",
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
}
DEFAULT_STATUS_POLICY = "Does not meet the credit policy. Status:Charged Off"


@dataclass(frozen=True)
class CreditRiskBundle:
    frame: pd.DataFrame


def make_credit_risk_dataset(*, n_samples: int = 3600, random_state: int = 42) -> CreditRiskBundle:
    rng = np.random.default_rng(random_state)

    annual_income = rng.lognormal(mean=11.2, sigma=0.45, size=n_samples)
    loan_amount = rng.normal(loc=17000.0, scale=7000.0, size=n_samples).clip(1200, 60000)
    interest_rate = rng.normal(loc=13.2, scale=4.3, size=n_samples).clip(4.5, 31.0)
    dti = rng.normal(loc=18.0, scale=8.0, size=n_samples).clip(0.1, 55.0)
    fico_score = rng.normal(loc=690.0, scale=55.0, size=n_samples).clip(520.0, 850.0)
    revolving_utilization = rng.normal(loc=48.0, scale=24.0, size=n_samples).clip(0.0, 120.0)

    employment_years = rng.choice(
        np.array(
            ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years", None],
            dtype=object,
        ),
        p=[0.08, 0.11, 0.12, 0.10, 0.24, 0.32, 0.03],
        size=n_samples,
    )
    home_ownership = rng.choice(
        np.array(["MORTGAGE", "RENT", "OWN", "OTHER", None], dtype=object),
        p=[0.47, 0.39, 0.12, 0.01, 0.01],
        size=n_samples,
    )
    loan_purpose = rng.choice(
        np.array(
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "small_business",
                "major_purchase",
                "car",
                None,
            ],
            dtype=object,
        ),
        p=[0.36, 0.24, 0.11, 0.07, 0.10, 0.10, 0.02],
        size=n_samples,
    )
    verification_status = rng.choice(
        np.array(["Verified", "Source Verified", "Not Verified", None], dtype=object),
        p=[0.36, 0.31, 0.31, 0.02],
        size=n_samples,
    )
    application_type = rng.choice(["Individual", "Joint App"], p=[0.92, 0.08], size=n_samples)

    missing_mask = rng.random(n_samples) < 0.09
    annual_income[missing_mask] = np.nan
    dti[rng.random(n_samples) < 0.07] = np.nan
    revolving_utilization[rng.random(n_samples) < 0.06] = np.nan

    risk_score = (
        0.000055 * np.nan_to_num(loan_amount, nan=17000)
        + 0.08 * np.nan_to_num(interest_rate, nan=13.2)
        + 0.045 * np.nan_to_num(dti, nan=18.0)
        + 0.03 * np.nan_to_num(revolving_utilization, nan=48.0)
        - 0.014 * np.nan_to_num(fico_score, nan=690.0)
    )
    risk_score += np.where(np.asarray(home_ownership) == "RENT", 1.25, 0.0)
    risk_score += np.where(np.asarray(loan_purpose) == "small_business", 1.55, 0.0)
    risk_score += np.where(np.asarray(application_type) == "Joint App", 0.55, 0.0)
    risk_score += rng.normal(loc=0.0, scale=2.0, size=n_samples)

    default_probability = 1.0 / (1.0 + np.exp(-(risk_score - 5.3) / 2.4))
    random_draws = rng.random(n_samples)
    is_default = random_draws < default_probability

    loan_status = np.where(is_default, "Charged Off", "Fully Paid").astype(object)
    policy_fail = rng.random(n_samples) < 0.03
    loan_status = np.where(policy_fail & is_default, DEFAULT_STATUS_POLICY, loan_status)

    frame = pd.DataFrame(
        {
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "dti": dti,
            "fico_score": fico_score,
            "revolving_utilization": revolving_utilization,
            "employment_length": employment_years,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "verification_status": verification_status,
            "application_type": application_type,
            "loan_status": loan_status,
        }
    )
    return CreditRiskBundle(frame=frame)

def build_target_from_status(frame: pd.DataFrame) -> pd.Series:
    status = frame["loan_status"].astype(str)
    return status.isin(DEFAULT_STATUSES).astype(int).rename("target")


def _employment_years_numeric(series: pd.Series) -> pd.Series:
    normalized = (
        series.fillna("0 years")
        .astype(str)
        .str.replace(r"\+", "", regex=True)
        .str.replace("< 1 year", "0 years", regex=False)
        .str.replace(" year", " years", regex=False)
    )
    extracted = normalized.str.extract(r"(?P<years>\d+)")["years"]
    return pd.to_numeric(extracted, errors="coerce").fillna(0.0)


def clean_and_encode_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = frame.drop(columns=["loan_status"]).copy()

    diagnostics["employment_years_numeric"] = _employment_years_numeric(
        diagnostics["employment_length"]
    )
    diagnostics["income_missing_indicator"] = diagnostics["annual_income"].isna().astype(int)
    diagnostics["dti_missing_indicator"] = diagnostics["dti"].isna().astype(int)

    for col in diagnostics.select_dtypes(include=[np.number]).columns:
        diagnostics[col] = diagnostics[col].fillna(diagnostics[col].median(skipna=True))

    cat_cols = diagnostics.select_dtypes(exclude=[np.number]).columns
    diagnostics.loc[:, cat_cols] = diagnostics.loc[:, cat_cols].fillna("UNKNOWN")

    model_frame = pd.get_dummies(diagnostics, columns=list(cat_cols), dummy_na=False)
    model_frame = model_frame.astype(float)

    return diagnostics, model_frame
