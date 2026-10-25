"""Simulated randomised experiment with a known, heterogeneous treatment effect.

The simulation is the *data-generating mechanism*: baseline conversion depends on
pre-treatment covariates, the treatment effect ``tau(x)`` depends on prior orders,
channel and recency, and revenue is drawn only for converters. Because ``tau`` is
known by construction we can (a) check that the estimated ATE is close to the truth
and (b) grade the uplift model against an oracle ranking.

This is **simulation**, not a real business experiment: every number in the report
describes the simulator, not a real campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig

CHANNELS = ("organic", "paid_search", "social", "email")
CHANNEL_PROBS = (0.40, 0.30, 0.20, 0.10)
CHANNEL_EFFECT = {"organic": 0.10, "paid_search": -0.10, "social": 0.00, "email": 0.05}

DEVICES = ("mobile", "desktop")
DEVICE_PROBS = (0.60, 0.40)
DEVICE_EFFECT = {"mobile": -0.05, "desktop": 0.05}


@dataclass
class SimulatedExperiment:
    users: pd.DataFrame          # user_id, treatment, covariates, p0, p1, true_tau
    outcomes: pd.DataFrame       # user_id, converted, revenue
    truth: dict[str, float]      # population-level ground truth


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _baseline_probability(prior_orders: np.ndarray, recency_days: np.ndarray,
                          channel: np.ndarray, device: np.ndarray) -> np.ndarray:
    logit = (
        -2.10
        + 0.25 * prior_orders
        - 0.005 * recency_days
        + np.array([CHANNEL_EFFECT[c] for c in channel])
        + np.array([DEVICE_EFFECT[d] for d in device])
    )
    return _sigmoid(logit)


def _treatment_effect(prior_orders: np.ndarray, recency_days: np.ndarray,
                      channel: np.ndarray, device: np.ndarray) -> np.ndarray:
    """True individual uplift. Deliberately explainable: additive business rules.

    Heterogeneity is strong on purpose (0.2pp for cold users up to ~12pp for loyal
    email subscribers): a realistic marketing campaign rarely lifts every segment
    equally, and a weak signal would make the uplift model impossible to grade.
    """
    order_effect = np.select(
        [prior_orders == 0, prior_orders == 1, prior_orders <= 3, prior_orders > 3],
        [0.000, 0.025, 0.060, 0.090],
        default=0.0,
    )
    tau = (
        0.005
        + order_effect
        + 0.030 * (channel == "email")
        + 0.010 * (device == "mobile")
        - 0.010 * (recency_days > 180)
    )
    return np.clip(tau, 0.002, 0.14)


def revenue_per_converter(prior_orders: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Revenue is log-normal and grows mildly with purchase history."""
    mu = np.log(80.0) + 0.18 * prior_orders
    return np.round(rng.lognormal(mean=mu, sigma=0.65, size=prior_orders.size), 2)


def simulate_experiment(cfg: SimConfig | None = None) -> SimulatedExperiment:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    prior_orders = np.clip(rng.poisson(1.2, size=n), 0, 12)
    recency_days = rng.integers(0, 366, size=n)
    tenure_days = recency_days + rng.integers(30, 900, size=n)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_PROBS)
    device = rng.choice(DEVICES, size=n, p=DEVICE_PROBS)

    p0 = _baseline_probability(prior_orders, recency_days, channel, device)
    tau = _treatment_effect(prior_orders, recency_days, channel, device)
    p1 = np.clip(p0 + tau, 0.0, 1.0)

    # Randomised assignment; a non-zero srm_offset deliberately breaks the ratio.
    treat_share = float(np.clip(cfg.treat_share + cfg.srm_offset, 0.05, 0.95))
    treatment = (rng.random(n) < treat_share).astype(int)

    converted = (rng.random(n) < np.where(treatment == 1, p1, p0)).astype(int)
    revenue = np.zeros(n)
    converters = np.flatnonzero(converted == 1)
    revenue[converters] = revenue_per_converter(prior_orders[converters], rng)

    users = pd.DataFrame(
        {
            "user_id": [f"u{i:07d}" for i in range(n)],
            "treatment": treatment,
            "prior_orders": prior_orders,
            "recency_days": recency_days,
            "tenure_days": tenure_days,
            "channel": channel,
            "device": device,
            "p0": p0,
            "p1": p1,
            "true_tau": tau,
        }
    )
    outcomes = pd.DataFrame({"user_id": users["user_id"], "converted": converted, "revenue": revenue})

    expected_revenue_per_conversion = float(
        np.mean(np.exp(np.log(80.0) + 0.18 * prior_orders) * np.exp(0.65 ** 2 / 2))
    )
    truth = {
        "ate_conversion": float(np.mean(p1 - p0)),
        "ate_revenue": float(np.mean((p1 - p0) * expected_revenue_per_conversion)),
        "baseline_conversion": float(np.mean(p0)),
        "treated_conversion": float(np.mean(p1)),
        "mean_tau_top_quartile": float(np.mean(tau[tau >= np.quantile(tau, 0.75)])),
        "expected_revenue_per_conversion": expected_revenue_per_conversion,
    }
    return SimulatedExperiment(users=users, outcomes=outcomes, truth=truth)


def build_analysis_frame(sim: SimulatedExperiment) -> pd.DataFrame:
    """One row per user with covariates, arm and observed outcomes (for evaluation)."""
    return sim.users.merge(sim.outcomes, on="user_id", how="inner")
