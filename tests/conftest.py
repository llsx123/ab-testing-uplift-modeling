"""Shared fixtures for the A/B testing test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abtest.config import SimConfig  # noqa: E402
from abtest.simulate import build_analysis_frame, simulate_experiment  # noqa: E402
from abtest import uplift  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg() -> SimConfig:
    return SimConfig(seed=7, n_users=30000, n_bootstrap=300)


@pytest.fixture(scope="session")
def small_frame(small_cfg):
    return build_analysis_frame(simulate_experiment(small_cfg))


@pytest.fixture(scope="session")
def trained(small_cfg, small_frame):
    train, holdout = uplift.split_train_test(small_frame, small_cfg)
    models = uplift.train_t_learner(train, small_cfg.seed)
    holdout = holdout.copy()
    holdout["predicted_uplift"] = uplift.predict_uplift(models, holdout)
    return {"models": models, "train": train, "holdout": holdout, "cfg": small_cfg}
