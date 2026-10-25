"""Statistical tests and uncertainty quantification.

Method choice follows the *metric type*, not habit:

* binary metric (conversion)  -> two-proportion z-test (normal approximation) + bootstrap
* continuous, skewed metric (revenue per user, many zeros) -> Welch t-test + bootstrap
* several metrics at once    -> Holm-Bonferroni adjustment

The bootstrap is **stratified by arm** (resample within treatment and control), which
preserves the randomisation ratio and is the correct null for a difference in means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_ztest(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict[str, float]:
    """Difference in conversion rates with a normal-approximation confidence interval."""
    p_t = x_t / n_t
    p_c = x_c / n_c
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = diff / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return {
        "rate_treatment": float(p_t),
        "rate_control": float(p_c),
        "diff": float(diff),
        "se": float(se),
        "z": float(z),
        "p_value": p_value,
        "ci_low": float(diff - z_crit * se),
        "ci_high": float(diff + z_crit * se),
        "relative_lift": float(diff / p_c) if p_c > 0 else float("nan"),
    }


def welch_ttest(treated: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Welch's t-test (unequal variances) for a continuous metric such as revenue."""
    result = stats.ttest_ind(treated, control, equal_var=False)
    diff = float(np.mean(treated) - np.mean(control))
    se = float(np.sqrt(np.var(treated, ddof=1) / treated.size + np.var(control, ddof=1) / control.size))
    df = (np.var(treated, ddof=1) / treated.size + np.var(control, ddof=1) / control.size) ** 2 / (
        (np.var(treated, ddof=1) / treated.size) ** 2 / (treated.size - 1)
        + (np.var(control, ddof=1) / control.size) ** 2 / (control.size - 1)
    )
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    return {
        "mean_treatment": float(np.mean(treated)),
        "mean_control": float(np.mean(control)),
        "diff": diff,
        "se": se,
        "t": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": float(diff - t_crit * se),
        "ci_high": float(diff + t_crit * se),
    }


def bootstrap_difference(
    frame: pd.DataFrame,
    value_col: str,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for ``mean(treatment) - mean(control)``.

    Resampling is done within each arm (stratified), so each replicate keeps the
    original allocation ratio. The bootstrap p-value is the two-sided share of
    replicates on the other side of zero (never exactly 0 with finite replicates).
    """
    rng = np.random.default_rng(seed)
    treated = frame.loc[frame["treatment"] == 1, value_col].to_numpy(dtype=float)
    control = frame.loc[frame["treatment"] == 0, value_col].to_numpy(dtype=float)
    if treated.size == 0 or control.size == 0:
        raise ValueError("both arms must be present")

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_t = rng.choice(treated, size=treated.size, replace=True)
        sample_c = rng.choice(control, size=control.size, replace=True)
        diffs[i] = sample_t.mean() - sample_c.mean()

    point = float(treated.mean() - control.mean())
    lower = float(np.quantile(diffs, alpha / 2))
    upper = float(np.quantile(diffs, 1 - alpha / 2))
    p_boot = float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    # No finite number of replicates can prove p = 0; report the resolution limit instead.
    if p_boot == 0.0:
        p_boot = 1.0 / (n_boot + 1)
    return {
        "estimate": point,
        "ci_low": lower,
        "ci_high": upper,
        "boot_se": float(diffs.std(ddof=1)),
        "p_value": min(1.0, p_boot),
        "n_boot": int(n_boot),
    }


def minimum_detectable_effect(p: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute lift for a two-proportion test (normal approximation)."""
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * np.sqrt(2 * p * (1 - p) / n_per_arm))


def holm_adjust(p_values: dict[str, float]) -> pd.DataFrame:
    """Holm-Bonferroni step-down adjustment for testing several metrics at once."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rows = []
    previous = 0.0
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        adjusted = max(adjusted, previous)
        previous = adjusted
        rows.append({"metric": name, "p_value": p, "adjusted_p": adjusted, "rank": rank + 1})
    return pd.DataFrame(rows)
