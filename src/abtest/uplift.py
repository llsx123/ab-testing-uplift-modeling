"""T-learner uplift model and Qini / AUUC evaluation.

Definitions used here (written down because "Qini" is often implemented differently):

* Sort the evaluation population by predicted uplift, descending.
* At each prefix of size ``k``: ``Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)``
  where ``Y_t`` / ``Y_c`` are cumulative conversions in the treated / control arm and
  ``N_t`` / ``N_c`` the cumulative arm sizes. This rescales the control arm to the
  treated arm's size, which is the standard Qini construction for a randomised
  experiment with unequal arm sizes.
* ``AUUC`` (area under the uplift curve) is the trapezoidal integral of ``Q(k)`` over
  the cumulative fraction of the population (units: conversions).
* A random ordering does **not** produce a zero area: even random targeting ends at the
  full population and therefore captures the average effect at ``k = n``. The sensible
  baseline is therefore the area of a random ranking, and the **normalised Qini
  coefficient** is ``(area_model − area_random) / (area_oracle − area_random)``:
  ``1`` = oracle ranking, ``0`` = random targeting, negative = worse than random.
  A random ordering is linear in expectation, so ``area_random = 0.5 · Q(n)``.

Predicted uplift is a score for *ranking*, not a causal claim about an individual:
the model is trained on randomised data and its output is a difference of two
response probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, SimConfig


def split_train_test(frame: pd.DataFrame, cfg: SimConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *by user*, stratified on the arm, so no user appears in both parts."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed + 1)
    train_mask = np.zeros(len(frame), dtype=bool)
    for arm in (0, 1):
        idx = np.flatnonzero(frame["treatment"].to_numpy() == arm)
        take = rng.random(idx.size) < cfg.train_share
        train_mask[idx[take]] = True
    return frame[train_mask].reset_index(drop=True), frame[~train_mask].reset_index(drop=True)


def _build_pipeline(seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=3,
        min_samples_leaf=50,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_t_learner(train: pd.DataFrame, seed: int = 42) -> dict[str, Pipeline]:
    """Two response models: P(conversion | x, treated) and P(conversion | x, control)."""
    features = train[list(FEATURE_COLUMNS)]
    treated = train["treatment"].to_numpy() == 1
    model_t = _build_pipeline(seed)
    model_c = _build_pipeline(seed)
    model_t.fit(features[treated], train.loc[treated, "converted"])
    model_c.fit(features[~treated], train.loc[~treated, "converted"])
    return {"treated": model_t, "control": model_c}


def predict_uplift(models: dict[str, Pipeline], frame: pd.DataFrame) -> np.ndarray:
    features = frame[list(FEATURE_COLUMNS)]
    p_t = models["treated"].predict_proba(features)[:, 1]
    p_c = models["control"].predict_proba(features)[:, 1]
    return p_t - p_c


def qini_curve(treatment: np.ndarray, outcome: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    treatment = np.asarray(treatment).astype(int)
    outcome = np.asarray(outcome).astype(float)
    score = np.asarray(score, dtype=float)
    if not (len(treatment) == len(outcome) == len(score)):
        raise ValueError("treatment / outcome / score must have the same length")

    order = np.argsort(-score, kind="mergesort")
    t = treatment[order]
    y = outcome[order]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - y_c * (n_t / np.maximum(n_c, 1))
    qini = np.nan_to_num(qini, nan=0.0)

    fraction = np.arange(1, len(t) + 1) / len(t)
    # Expected Qini of a random ordering is linear from 0 to Q(n): even random
    # targeting captures the average effect once the whole population is reached.
    random_reference = fraction * qini[-1]
    return pd.DataFrame(
        {
            "fraction": fraction,
            "qini": qini,
            "random_reference": random_reference,
            "n_treated": n_t,
            "n_control": n_c,
            "conversions_treated": y_t,
            "conversions_control": y_c,
        }
    )


def qini_area(curve: pd.DataFrame) -> float:
    """Area under the Qini curve, in conversions (trapezoidal rule)."""
    x = np.concatenate([[0.0], curve["fraction"].to_numpy()])
    y = np.concatenate([[0.0], curve["qini"].to_numpy()])
    return float(np.trapezoid(y, x))


def random_baseline_area(curve: pd.DataFrame) -> float:
    """Area under the random-ordering reference: ``0.5 * Q(n)`` (linear curve)."""
    return 0.5 * float(curve["qini"].iloc[-1])


def normalized_auuc(model_curve: pd.DataFrame, oracle_curve: pd.DataFrame) -> float:
    """Normalised Qini coefficient: (model − random) / (oracle − random).

    1.0 = ranking as good as the true individual effect, 0.0 = random targeting,
    negative = worse than random. Both curves must be evaluated on the same holdout,
    so they share the same ``Q(n)`` endpoint and therefore the same random baseline.
    """
    baseline = random_baseline_area(oracle_curve)
    denominator = qini_area(oracle_curve) - baseline
    if denominator <= 0:
        raise ValueError("oracle must beat the random baseline for a meaningful normalisation")
    return float((qini_area(model_curve) - baseline) / denominator)


def decile_uplift(
    treatment: np.ndarray,
    outcome: np.ndarray,
    score: np.ndarray,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Observed uplift per score decile with a normal-approximation confidence interval."""
    treatment = np.asarray(treatment).astype(int)
    outcome = np.asarray(outcome).astype(float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    groups = np.array_split(order, n_bins)
    z = float(stats.norm.ppf(1 - alpha / 2))

    rows = []
    for decile, idx in enumerate(groups, start=1):
        t = treatment[idx]
        y = outcome[idx]
        n_t = int(t.sum())
        n_c = int((1 - t).sum())
        rate_t = float(y[t == 1].mean()) if n_t else float("nan")
        rate_c = float(y[t == 0].mean()) if n_c else float("nan")
        uplift = rate_t - rate_c
        se = np.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c) if n_t and n_c else np.nan
        rows.append(
            {
                "decile": decile,
                "n": int(len(idx)),
                "n_treated": n_t,
                "n_control": n_c,
                "conversion_treated": rate_t,
                "conversion_control": rate_c,
                "uplift": float(uplift),
                "ci_low": float(uplift - z * se),
                "ci_high": float(uplift + z * se),
            }
        )
    return pd.DataFrame(rows)
