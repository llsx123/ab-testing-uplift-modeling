"""Validity checks: SRM detection, covariate balance, data integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abtest import quality
from abtest.config import SimConfig
from abtest.simulate import build_analysis_frame, simulate_experiment


def test_srm_not_triggered_for_balanced_assignment(small_frame):
    result = quality.srm_test(small_frame, expected_share=0.5)
    assert result["srm_detected"] is False
    assert result["p_value"] > 0.001


def test_srm_detected_for_planted_imbalance():
    sim = simulate_experiment(SimConfig(seed=11, n_users=40000, srm_offset=0.05))
    frame = build_analysis_frame(sim)
    result = quality.srm_test(frame, expected_share=0.5)
    assert result["srm_detected"] is True
    assert result["p_value"] < 1e-3
    assert result["observed_share"] > 0.53


def test_balance_passes_for_randomised_assignment(small_frame):
    table = quality.balance_table(small_frame)
    assert table["abs_smd"].max() < quality.SMD_THRESHOLD
    assert table["balanced"].all()


def test_balance_detects_a_planted_covariate_shift(small_frame):
    shifted = small_frame.copy()
    shifted["prior_orders"] = shifted["prior_orders"].astype(float)
    shifted.loc[shifted["treatment"] == 1, "prior_orders"] = (
        shifted.loc[shifted["treatment"] == 1, "prior_orders"] + 0.6
    )
    table = quality.balance_table(shifted)
    row = table[table["feature"] == "prior_orders"].iloc[0]
    assert abs(row["smd"]) > quality.SMD_THRESHOLD
    assert not bool(row["balanced"])


def test_data_checks_are_clean_on_simulated_data(small_frame):
    checks = quality.data_checks(small_frame)
    assert (checks["rows"] == 0).all(), checks.to_dict("records")


def test_balance_table_covers_every_feature_level(small_frame):
    table = quality.balance_table(small_frame)
    assert set(table["feature"]) >= {"prior_orders", "recency_days", "tenure_days", "channel", "device"}
    assert table["level"].nunique() > 5
    assert table["smd"].abs().max() < 1.0


def test_srm_uses_expected_share_parameter():
    sim = simulate_experiment(SimConfig(seed=3, n_users=20000, treat_share=0.3))
    frame = build_analysis_frame(sim)
    correct = quality.srm_test(frame, expected_share=0.3)
    wrong = quality.srm_test(frame, expected_share=0.5)
    assert correct["srm_detected"] is False
    assert wrong["srm_detected"] is True
    assert correct["observed_share"] == pytest.approx(0.3, abs=0.02)
