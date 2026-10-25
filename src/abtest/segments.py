"""Observed treatment effect by pre-treatment segment (no model involved)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import two_proportion_ztest

SEGMENT_ORDER = ("0 orders", "1 order", "2-3 orders", "4+ orders")


def add_segments(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["segment"] = pd.cut(
        result["prior_orders"],
        bins=[-1, 0, 1, 3, 99],
        labels=list(SEGMENT_ORDER),
    ).astype(str)
    return result


def segment_uplift(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per-segment conversion uplift with a two-proportion confidence interval."""
    rows = []
    for segment in SEGMENT_ORDER:
        subset = frame[frame["segment"] == segment]
        treated = subset[subset["treatment"] == 1]
        control = subset[subset["treatment"] == 0]
        if len(treated) == 0 or len(control) == 0:
            continue
        test = two_proportion_ztest(
            int(treated["converted"].sum()), len(treated),
            int(control["converted"].sum()), len(control), alpha,
        )
        rows.append(
            {
                "segment": segment,
                "n": int(len(subset)),
                "n_treated": int(len(treated)),
                "n_control": int(len(control)),
                "conversion_treated": test["rate_treatment"],
                "conversion_control": test["rate_control"],
                "uplift": test["diff"],
                "ci_low": test["ci_low"],
                "ci_high": test["ci_high"],
                "p_value": test["p_value"],
            }
        )
    return pd.DataFrame(rows)
