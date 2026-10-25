"""Qini/AUUC implementation, decile table and the T-learner pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abtest import uplift
from abtest.config import SimConfig
from abtest.simulate import build_analysis_frame, simulate_experiment


def test_qini_curve_shape_and_monotone_fraction():
    treatment = np.array([1, 1, 0, 0, 1, 0])
    outcome = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    curve = uplift.qini_curve(treatment, outcome, score)
    assert len(curve) == 6
    assert curve["fraction"].is_monotonic_increasing
    assert curve["fraction"].iloc[-1] == pytest.approx(1.0)
    assert (curve["n_treated"] + curve["n_control"]).to_numpy()[-1] == 6


def test_qini_area_matches_hand_computation():
    # treatment = [1,0,1,0], outcome = [1,1,0,1] with descending score
    # Q = [1, 0, -1, -1] at fractions [.25, .5, .75, 1] -> trapezoid = -0.125
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([1, 1, 0, 1])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    curve = uplift.qini_curve(treatment, outcome, score)
    assert curve["qini"].to_numpy() == pytest.approx([1.0, 0.0, -1.0, -1.0])
    assert uplift.qini_area(curve) == pytest.approx(-0.125)


def test_oracle_ranking_is_perfect_by_construction(small_frame, small_cfg):
    _, holdout = uplift.split_train_test(small_frame, small_cfg)
    oracle = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["true_tau"])
    same = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["true_tau"].copy())
    assert uplift.normalized_auuc(oracle, same) == pytest.approx(1.0)
    assert uplift.qini_area(oracle) > 0


def test_random_reference_is_linear_and_starts_at_zero():
    treatment = np.array([1, 1, 0, 0, 1, 0])
    outcome = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    curve = uplift.qini_curve(treatment, outcome, score)
    expected = curve["fraction"] * curve["qini"].iloc[-1]
    assert curve["random_reference"].to_numpy() == pytest.approx(expected.to_numpy())
    assert curve["random_reference"].iloc[-1] == pytest.approx(curve["qini"].iloc[-1])
    assert uplift.random_baseline_area(curve) == pytest.approx(0.5 * curve["qini"].iloc[-1])


def test_reversed_ranking_is_worse_than_random(small_frame, small_cfg):
    _, holdout = uplift.split_train_test(small_frame, small_cfg)
    oracle = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["true_tau"])
    reversed_curve = uplift.qini_curve(
        holdout["treatment"], holdout["converted"], -holdout["true_tau"]
    )
    assert uplift.normalized_auuc(reversed_curve, oracle) < 0
    assert uplift.qini_area(oracle) > uplift.qini_area(reversed_curve)


def test_decile_table_partitions_the_population(small_frame, small_cfg):
    _, holdout = uplift.split_train_test(small_frame, small_cfg)
    deciles = uplift.decile_uplift(holdout["treatment"], holdout["converted"], holdout["true_tau"])
    assert len(deciles) == 10
    assert deciles["n"].sum() == len(holdout)
    assert list(deciles["decile"]) == list(range(1, 11))
    assert (deciles["uplift"] - deciles["ci_low"] >= 0).all()
    assert (deciles["ci_high"] - deciles["uplift"] >= 0).all()


def test_decile_one_contains_the_highest_scores(small_frame, small_cfg):
    _, holdout = uplift.split_train_test(small_frame, small_cfg)
    holdout = holdout.sort_values("true_tau", ascending=False).reset_index(drop=True)
    scores = holdout["true_tau"].to_numpy()
    deciles = uplift.decile_uplift(holdout["treatment"], holdout["converted"], scores)
    assert deciles.iloc[0]["conversion_treated"] >= deciles.iloc[-1]["conversion_treated"]


def test_t_learner_ranks_better_than_random(trained):
    holdout = trained["holdout"]
    model_curve = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["predicted_uplift"])
    oracle_curve = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["true_tau"])
    ratio = uplift.normalized_auuc(model_curve, oracle_curve)
    assert ratio > 0.1, f"normalised Qini coefficient too low: {ratio}"
    assert uplift.qini_area(model_curve) > uplift.random_baseline_area(model_curve)


def test_t_learner_correlates_with_true_effect(trained):
    holdout = trained["holdout"]
    correlation = np.corrcoef(holdout["predicted_uplift"], holdout["true_tau"])[0, 1]
    assert correlation > 0.1, f"prediction/truth correlation too low: {correlation}"


def test_t_learner_top_decile_is_positive(trained):
    holdout = trained["holdout"]
    deciles = uplift.decile_uplift(holdout["treatment"], holdout["converted"], holdout["predicted_uplift"])
    assert deciles.iloc[0]["uplift"] > 0
    assert deciles.iloc[0]["uplift"] > deciles.iloc[-1]["uplift"]


def test_prediction_is_bounded_and_finite(trained):
    predictions = trained["holdout"]["predicted_uplift"].to_numpy()
    assert np.isfinite(predictions).all()
    assert predictions.min() > -1 and predictions.max() < 1


def test_train_and_holdout_are_disjoint(small_frame, small_cfg):
    train, holdout = uplift.split_train_test(small_frame, small_cfg)
    assert set(train["user_id"]).isdisjoint(set(holdout["user_id"]))
    assert abs(len(train) / len(small_frame) - small_cfg.train_share) < 0.02
