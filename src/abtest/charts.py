"""Figures for the A/B testing report (headless matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2E5EAA"
LIGHT_BLUE = "#8FB3F5"
RED = "#D64545"
GRAY = "#9AA5B1"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def balance_plot(balance: pd.DataFrame, path: Path) -> Path:
    table = balance.copy()
    table["label"] = np.where(table["level"] == "(numeric)", table["feature"], table["feature"] + "=" + table["level"])
    table = table.sort_values("abs_smd")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(table) + 2))
    colors = [BLUE if flag else RED for flag in table["balanced"]]
    ax.barh(table["label"], table["abs_smd"], color=colors)
    ax.axvline(0.1, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("|standardised mean difference|")
    ax.set_title("Covariate balance (dashed line = 0.1 threshold)")
    return _save(fig, path)


def qini_plot(model_curve: pd.DataFrame, oracle_curve: pd.DataFrame, path: Path, normalized_auuc: float) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_curve["fraction"], model_curve["qini"], color=BLUE, linewidth=2, label=f"T-learner (normalised Qini={normalized_auuc:.3f})")
    ax.plot(oracle_curve["fraction"], oracle_curve["qini"], color=GRAY, linestyle="--", label="Oracle (true uplift)")
    ax.plot(model_curve["fraction"], model_curve["random_reference"], color=RED, linestyle=":", label="Random targeting")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini / uplift curve on the holdout set")
    ax.legend(frameon=False)
    return _save(fig, path)


def decile_plot(deciles: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = deciles["decile"]
    ax.bar(x, deciles["uplift"], color=LIGHT_BLUE, edgecolor=BLUE, label="Observed uplift")
    ax.errorbar(
        x,
        deciles["uplift"],
        yerr=[deciles["uplift"] - deciles["ci_low"], deciles["ci_high"] - deciles["uplift"]],
        fmt="none",
        ecolor=BLUE,
        capsize=3,
    )
    ax.axhline(0, color=RED, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xlabel("Uplift decile (1 = highest predicted uplift)")
    ax.set_ylabel("Conversion uplift (treated - control)")
    ax.set_title("Observed uplift by predicted-uplift decile")
    return _save(fig, path)


def segment_plot(segment_table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(segment_table["segment"], segment_table["uplift"], color=LIGHT_BLUE, edgecolor=BLUE)
    ax.set_ylabel("Conversion uplift")
    ax.set_title("Observed uplift by pre-treatment segment")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)
