"""The simulator must be reproducible, and its ground truth must match its design."""

from __future__ import annotations

import numpy as np
import pytest

from abtest.config import FEATURE_COLUMNS, POST_TREATMENT_COLUMNS, SimConfig
from abtest.simulate import build_analysis_frame, simulate_experiment


def test_reproducible_with_same_seed():
    first = simulate_experiment(SimConfig(seed=5, n_users=2000))
    second = simulate_experiment(SimConfig(seed=5, n_users=2000))
    assert first.truth == second.truth
    assert first.users.equals(second.users)
    assert first.outcomes.equals(second.outcomes)


def test_different_seed_gives_different_sample():
    first = simulate_experiment(SimConfig(seed=5, n_users=2000))
    second = simulate_experiment(SimConfig(seed=6, n_users=2000))
    assert not first.outcomes.equals(second.outcomes)


def test_truth_matches_design():
    sim = simulate_experiment(SimConfig(seed=21, n_users=20000))
    assert 0.005 < sim.truth["ate_conversion"] < 0.08
    assert sim.truth["treated_conversion"] > sim.truth["baseline_conversion"]
    # ATE must equal the mean of the individual effects by construction.
    assert sim.truth["ate_conversion"] == pytest.approx(float(sim.users["true_tau"].mean()), rel=1e-9)


def test_effect_is_heterogeneous():
    sim = simulate_experiment(SimConfig(seed=22, n_users=20000))
    tau = sim.users["true_tau"].to_numpy()
    top_quartile = tau >= np.quantile(tau, 0.75)
    assert tau[top_quartile].mean() > tau.mean() + 0.005


def test_revenue_only_for_converters():
    sim = simulate_experiment(SimConfig(seed=23, n_users=5000))
    merged = build_analysis_frame(sim)
    assert (merged.loc[merged["converted"] == 0, "revenue"] == 0).all()
    assert (merged.loc[merged["converted"] == 1, "revenue"] > 0).all()


def test_features_are_pre_treatment_only():
    assert set(FEATURE_COLUMNS).isdisjoint(POST_TREATMENT_COLUMNS)
    sim = simulate_experiment(SimConfig(seed=24, n_users=2000))
    assert set(POST_TREATMENT_COLUMNS) <= set(sim.users.columns) | set(sim.outcomes.columns)


def test_probabilities_are_in_range():
    sim = simulate_experiment(SimConfig(seed=25, n_users=10000))
    for column in ("p0", "p1"):
        values = sim.users[column].to_numpy()
        assert values.min() > 0 and values.max() < 1
    assert (sim.users["p1"] >= sim.users["p0"]).all()
