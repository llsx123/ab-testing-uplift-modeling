#!/usr/bin/env python
"""Validate the experiment analysis on the real Hillstrom (2008) e-mail dataset.

Source: Kevin Hillstrom, MineThatData E-Mail Analytics And Data Mining Challenge
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

The script downloads the CSV to ``data/hillstrom.csv`` if it is missing, reproduces
the three-arm analysis (Mens / Womens / No E-Mail) and a T-learner Qini evaluation,
then writes ``reports/hillstrom_summary.json``.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "hillstrom.csv"
OUT = PROJECT_ROOT / "reports" / "hillstrom_summary.json"
NUM = ["recency", "history"]
CAT = ["channel", "zip_code", "history_segment"]


def ensure_data() -> None:
    if DATA.exists():
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, DATA)


def qini_curve(treatment: np.ndarray, outcome: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-score, kind="mergesort")
    t, y = treatment[order], outcome[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = np.nan_to_num(y_t - y_c * (n_t / np.maximum(n_c, 1)))
    return pd.DataFrame({"fraction": np.arange(1, len(t) + 1) / len(t), "qini": q})


def area(curve: pd.DataFrame) -> float:
    x = np.concatenate([[0.0], curve["fraction"].to_numpy()])
    y = np.concatenate([[0.0], curve["qini"].to_numpy()])
    return float(np.trapezoid(y, x))


def uplift_eval(h: pd.DataFrame, arm: str) -> dict:
    subset = h[h["segment"].isin([arm, "No E-Mail"])].copy()
    subset["treatment"] = (subset["segment"] == arm).astype(int)
    rng = np.random.default_rng(20240101)
    train_mask = rng.random(len(subset)) < 0.7
    train, test = subset[train_mask], subset[~train_mask]

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    models = {}
    for arm_value in (0, 1):
        model = Pipeline([("prep", prep), ("m", HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06, min_samples_leaf=50, random_state=1))])
        part = train[train["treatment"] == arm_value]
        model.fit(part[NUM + CAT], part["conversion"])
        models[arm_value] = model

    score = (
        models[1].predict_proba(test[NUM + CAT])[:, 1]
        - models[0].predict_proba(test[NUM + CAT])[:, 1]
    )
    curve = qini_curve(test["treatment"].to_numpy(), test["conversion"].to_numpy(), score)
    model_area = area(curve)
    random_area = 0.5 * float(curve["qini"].iloc[-1])

    order = np.argsort(-score, kind="mergesort")
    top = order[: len(order) // 10]
    t, y = test["treatment"].to_numpy()[top], test["conversion"].to_numpy()[top]
    top_uplift = float(y[t == 1].mean() - y[t == 0].mean()) * 100

    return {
        "arm": arm,
        "n": int(len(subset)),
        "holdout": int(len(test)),
        "qini_area": round(model_area, 4),
        "random_baseline_area": round(random_area, 4),
        "qini_lift_over_random": round((model_area - random_area) / random_area, 4) if random_area else None,
        "top_decile_uplift_pp": round(top_uplift, 4),
        "overall_uplift_pp": round(float(
            test[test["treatment"] == 1]["conversion"].mean()
            - test[test["treatment"] == 0]["conversion"].mean()) * 100, 4),
    }


def arm_stats(h: pd.DataFrame, arm: str) -> dict:
    treated = h[h["segment"] == arm]
    control = h[h["segment"] == "No E-Mail"]
    x_t, n_t = int(treated["conversion"].sum()), len(treated)
    x_c, n_c = int(control["conversion"].sum()), len(control)
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    return {
        "arm": arm,
        "n": int(len(treated)),
        "conversion_treated": round(float(p_t), 5),
        "conversion_control": round(float(p_c), 5),
        "conversion_lift_pp": round(float(diff * 100), 3),
        "conversion_ci_pp": [round(float((diff - 1.96 * se) * 100), 3),
                             round(float((diff + 1.96 * se) * 100), 3)],
        "visit_lift_pp": round(float(treated["visit"].mean() - control["visit"].mean()) * 100, 3),
        "spend_lift": round(float(treated["spend"].mean() - control["spend"].mean()), 4),
        "spend_p": float(stats.ttest_ind(treated["spend"], control["spend"], equal_var=False).pvalue),
    }


def main() -> int:
    ensure_data()
    h = pd.read_csv(DATA)
    summary = {
        "source": URL,
        "customers": int(len(h)),
        "segments": {k: int(v) for k, v in h["segment"].value_counts().items()},
        "arms": {arm: arm_stats(h, arm) for arm in ("Mens E-Mail", "Womens E-Mail")},
        "uplift": {arm: uplift_eval(h, arm) for arm in ("Mens E-Mail", "Womens E-Mail")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"customers": summary["customers"], "arms": summary["arms"],
                      "uplift": summary["uplift"]}, ensure_ascii=False, indent=2))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
