"""Configuration for the A/B testing & uplift project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

#: Pre-treatment covariates only. ``converted`` / ``revenue`` / ``treatment`` must never
#: appear here: using them as features would leak the outcome into the uplift model.
NUMERIC_FEATURES = ("prior_orders", "recency_days", "tenure_days")
CATEGORICAL_FEATURES = ("channel", "device")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Columns that are only known *after* assignment / after the outcome.
POST_TREATMENT_COLUMNS = ("treatment", "converted", "revenue", "true_tau", "p0", "p1")


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the simulated experiment. Fixed seed => reproducible results."""

    seed: int = 20240101
    n_users: int = 60000
    treat_share: float = 0.5
    #: Add this to the treatment share to simulate a sample-ratio mismatch (0.02 => 52/48).
    srm_offset: float = 0.0
    train_share: float = 0.7
    n_bootstrap: int = 2000
    alpha: float = 0.05
