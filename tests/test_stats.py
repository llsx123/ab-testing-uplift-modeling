"""Statistical routines: properties, known examples and bootstrap behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abtest import stats


def test_ztest_identical_arms_gives_p_one():
    result = stats.two_proportion_ztest(100, 1000, 100, 1000)
    assert result["diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_ztest_known_example():
    # 20% vs 15% on 1000 users per arm -> z ~ 2.95, significant
    result = stats.two_proportion_ztest(200, 1000, 150, 1000)
    assert result["diff"] == pytest.approx(0.05)
    assert result["z"] == pytest.approx(2.947, abs=0.01)
    assert result["p_value"] < 0.01
    assert result["ci_low"] > 0
    assert result["ci_low"] < result["diff"] < result["ci_high"]


def test_ztest_interval_width_matches_normal_approximation():
    result = stats.two_proportion_ztest(200, 1000, 150, 1000)
    width = result["ci_high"] - result["ci_low"]
    assert width == pytest.approx(2 * 1.959963984540054 * result["se"], rel=1e-9)


def test_welch_identical_samples():
    rng = np.random.default_rng(0)
    sample = rng.normal(10, 5, size=2000)
    result = stats.welch_ttest(sample, sample)
    assert result["diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0)


def test_welch_detects_shift_and_ci():
    rng = np.random.default_rng(1)
    treated = rng.normal(12, 5, size=2000)
    control = rng.normal(10, 5, size=2000)
    result = stats.welch_ttest(treated, control)
    assert result["diff"] == pytest.approx(2.0, abs=0.5)
    assert result["p_value"] < 1e-6
    assert result["ci_low"] < result["diff"] < result["ci_high"]


def test_bootstrap_contains_zero_when_arms_are_identical():
    rng = np.random.default_rng(2)
    n = 4000
    frame = pd.DataFrame(
        {
            "treatment": np.repeat([1, 0], n),
            "converted": rng.binomial(1, 0.1, size=2 * n),
        }
    )
    result = stats.bootstrap_difference(frame, "converted", n_boot=500, seed=3)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.05


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(4)
    n = 4000
    treated = rng.binomial(1, 0.15, size=n)
    control = rng.binomial(1, 0.10, size=n)
    frame = pd.DataFrame({"treatment": np.concatenate([np.ones(n), np.zeros(n)]),
                          "converted": np.concatenate([treated, control])})
    result = stats.bootstrap_difference(frame, "converted", n_boot=500, seed=5)
    assert result["estimate"] == pytest.approx(0.05, abs=0.02)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01


def test_bootstrap_is_reproducible():
    frame = pd.DataFrame({"treatment": [1, 1, 0, 0] * 100, "converted": [1, 0, 1, 0] * 100})
    first = stats.bootstrap_difference(frame, "converted", n_boot=200, seed=11)
    second = stats.bootstrap_difference(frame, "converted", n_boot=200, seed=11)
    assert first == second


def test_minimum_detectable_effect_matches_formula():
    p, n = 0.09, 30000
    expected = (1.959963984540054 + 0.8416212335729143) * np.sqrt(2 * p * (1 - p) / n)
    assert stats.minimum_detectable_effect(p, n) == pytest.approx(expected, rel=1e-9)
    assert 0 < stats.minimum_detectable_effect(p, n) < 0.02


def test_holm_adjustment_is_monotone_and_conservative():
    table = stats.holm_adjust({"a": 0.001, "b": 0.04, "c": 0.03})
    assert (table["adjusted_p"] >= table["p_value"] - 1e-12).all()
    assert (table["adjusted_p"] <= 1.0).all()
    ordered = table.sort_values("rank")
    assert ordered["adjusted_p"].is_monotonic_increasing
    assert ordered.iloc[0]["adjusted_p"] == pytest.approx(0.003)
