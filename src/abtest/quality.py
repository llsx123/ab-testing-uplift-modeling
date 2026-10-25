"""Experiment diagnostics: sample-ratio mismatch (SRM), covariate balance, data checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SMD_THRESHOLD = 0.10


def data_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Basic integrity checks that must be clean before any statistical test."""
    checks = [
        ("duplicate_user_id", int(frame["user_id"].duplicated().sum()), "user_id must be unique"),
        ("missing_values", int(frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)].isna().sum().sum()), "no missing covariates"),
        ("invalid_treatment", int((~frame["treatment"].isin([0, 1])).sum()), "treatment must be 0/1"),
        ("negative_prior_orders", int((frame["prior_orders"] < 0).sum()), "prior_orders >= 0"),
        ("recency_out_of_range", int(((frame["recency_days"] < 0) | (frame["recency_days"] > 3650)).sum()), "0 <= recency_days <= 3650"),
        ("negative_revenue", int((frame["revenue"] < 0).sum()), "revenue >= 0"),
    ]
    return pd.DataFrame(checks, columns=["check", "rows", "description"])


def srm_test(frame: pd.DataFrame, expected_share: float = 0.5) -> dict[str, float]:
    """Chi-square goodness-of-fit test for the assignment ratio.

    SRM is the first thing to check: a broken randomiser invalidates every downstream
    comparison, so the pipeline refuses to continue when it is detected.
    """
    n = len(frame)
    n_t = int((frame["treatment"] == 1).sum())
    n_c = n - n_t
    observed = np.array([n_t, n_c], dtype=float)
    expected = np.array([expected_share, 1 - expected_share], dtype=float) * n
    chi2, p_value = stats.chisquare(observed, expected)
    return {
        "n": n,
        "n_treatment": n_t,
        "n_control": n_c,
        "observed_share": float(n_t / n),
        "expected_share": float(expected_share),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < 0.001),
    }


def _smd_numeric(treated: pd.Series, control: pd.Series) -> float:
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((treated.mean() - control.mean()) / pooled)


def balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardised mean differences per covariate / level (|SMD| < 0.1 is balanced)."""
    treated = frame[frame["treatment"] == 1]
    control = frame[frame["treatment"] == 0]
    rows: list[dict] = []

    for feature in NUMERIC_FEATURES:
        rows.append(
            {
                "feature": feature,
                "level": "(numeric)",
                "mean_treated": float(treated[feature].mean()),
                "mean_control": float(control[feature].mean()),
                "smd": _smd_numeric(treated[feature], control[feature]),
            }
        )

    for feature in CATEGORICAL_FEATURES:
        levels = sorted(frame[feature].unique())
        for level in levels:
            p_t = float((treated[feature] == level).mean())
            p_c = float((control[feature] == level).mean())
            pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2.0)
            smd = 0.0 if pooled == 0 else (p_t - p_c) / pooled
            rows.append(
                {
                    "feature": feature,
                    "level": str(level),
                    "mean_treated": p_t,
                    "mean_control": p_c,
                    "smd": float(smd),
                }
            )

    table = pd.DataFrame(rows)
    table["abs_smd"] = table["smd"].abs()
    table["balanced"] = table["abs_smd"] < SMD_THRESHOLD
    return table
