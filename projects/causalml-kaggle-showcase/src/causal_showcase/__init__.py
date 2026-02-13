"""CausalML learning showcase package."""

from .data import PreparedData, load_marketing_ab_data, train_test_split_prepared
from .diagnostics import PropensityDiagnostics, covariate_balance_table, propensity_diagnostics
from .evaluation import estimate_empirical_ate, qini_auc, qini_curve, uplift_at_k
from .modeling import fit_meta_learners, fit_uplift_tree
from .policy import select_best_model_per_budget, simulate_policy_table
from .verification import VerificationResult, verify_learning_artifacts

__all__ = [
    "PreparedData",
    "PropensityDiagnostics",
    "VerificationResult",
    "covariate_balance_table",
    "estimate_empirical_ate",
    "fit_meta_learners",
    "fit_uplift_tree",
    "load_marketing_ab_data",
    "propensity_diagnostics",
    "qini_auc",
    "qini_curve",
    "select_best_model_per_budget",
    "simulate_policy_table",
    "train_test_split_prepared",
    "uplift_at_k",
    "verify_learning_artifacts",
]
