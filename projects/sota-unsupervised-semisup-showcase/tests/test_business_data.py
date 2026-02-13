from __future__ import annotations

from pathlib import Path

import pandas as pd

from sota_showcase.data import (
    LOAN_FEATURE_COLUMNS,
    load_business_loan_dataset,
    parse_emp_length_years,
    parse_percentage,
    parse_term_months,
)


def test_parsers_cover_common_loan_formats() -> None:
    terms = pd.Series(["36 months", "60 months", None])
    rates = pd.Series(["13.56%", "7.9%", None])
    emp = pd.Series(["10+ years", "< 1 year", None])

    parsed_terms = parse_term_months(terms).tolist()
    parsed_rates = parse_percentage(rates).tolist()
    parsed_emp = parse_emp_length_years(emp).tolist()

    assert parsed_terms[:2] == [36.0, 60.0]
    assert parsed_rates[:2] == [13.56, 7.9]
    assert parsed_emp[0] == 10.0
    assert parsed_emp[1] == 0.5


def test_business_loader_builds_feature_matrix(tmp_path: Path) -> None:
    rows = [
        {
            "loan_status": "Fully Paid",
            "loan_amnt": 5000,
            "term": "36 months",
            "int_rate": "10.65%",
            "installment": 162.87,
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": "10+ years",
            "home_ownership": "RENT",
            "annual_inc": 24000,
            "verification_status": "Verified",
            "purpose": "credit_card",
            "addr_state": "AZ",
            "dti": 27.65,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 3,
            "pub_rec": 0,
            "revol_bal": 13648,
            "revol_util": "83.7%",
            "total_acc": 9,
            "application_type": "INDIVIDUAL",
        },
        {
            "loan_status": "Charged Off",
            "loan_amnt": 2500,
            "term": "60 months",
            "int_rate": "15.27%",
            "installment": 59.83,
            "grade": "C",
            "sub_grade": "C4",
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "annual_inc": 30000,
            "verification_status": "Source Verified",
            "purpose": "car",
            "addr_state": "GA",
            "dti": 1.0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 5,
            "open_acc": 3,
            "pub_rec": 0,
            "revol_bal": 1687,
            "revol_util": "9.4%",
            "total_acc": 4,
            "application_type": "INDIVIDUAL",
        },
    ]
    all_rows = rows * 4
    frame = pd.DataFrame(all_rows)

    # Guard against missing any required features in fixture construction.
    assert set(LOAN_FEATURE_COLUMNS).issubset(set(frame.columns))

    csv_path = tmp_path / "loan_fixture.csv"
    frame.to_csv(csv_path, index=False)

    dataset = load_business_loan_dataset(
        csv_path=csv_path,
        read_rows=100,
        sample_size=6,
        random_state=42,
        scale=True,
    )

    assert dataset.X.shape[0] == 6
    assert dataset.X.shape[1] > 10
    assert set(dataset.y.tolist()) == {0, 1}
    assert dataset.dataset_name == "lendingclub_credit_risk"
