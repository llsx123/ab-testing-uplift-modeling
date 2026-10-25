#!/usr/bin/env python
"""End-to-end pipeline for the A/B testing & uplift project.

    python scripts/run_pipeline.py

simulate -> validity checks (SRM / balance) -> statistical tests -> T-learner ->
Qini/AUUC -> charts -> Markdown report -> summary JSON.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from abtest import charts, quality, report, segments, simulate, stats as ab_stats, uplift  # noqa: E402
from abtest.config import FEATURE_COLUMNS, OUTPUT_DIR, REPORT_DIR, FIG_DIR, SimConfig  # noqa: E402


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> int:
    started = time.time()
    for directory in (OUTPUT_DIR, REPORT_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig()
    print(f"[1/7] simulate experiment (seed={cfg.seed}, n={cfg.n_users:,}, treat={cfg.treat_share:.2f})")
    sim = simulate.simulate_experiment(cfg)
    frame = simulate.build_analysis_frame(sim)
    print(f"      true ATE (conversion) = {sim.truth['ate_conversion'] * 100:.2f} pp"
          f" | baseline = {sim.truth['baseline_conversion'] * 100:.2f}%")

    print("[2/7] validity checks: data integrity, SRM, covariate balance")
    checks = quality.data_checks(frame)
    srm = quality.srm_test(frame, expected_share=cfg.treat_share)
    balance = quality.balance_table(frame)
    checks.to_csv(OUTPUT_DIR / "data_checks.csv", index=False)
    balance.to_csv(OUTPUT_DIR / "balance.csv", index=False)
    print(f"      SRM p={srm['p_value']:.4f} detected={srm['srm_detected']}"
          f" | max |SMD|={balance['abs_smd'].max():.4f}")
    if srm["srm_detected"]:
        raise SystemExit("SRM detected: the randomiser is broken, stop and fix the assignment before analysis")
    if balance["abs_smd"].max() >= quality.SMD_THRESHOLD:
        raise SystemExit(f"covariate imbalance detected: max |SMD| = {balance['abs_smd'].max():.4f}")

    print("[3/7] hypothesis tests (conversion: z-test, revenue: Welch) + stratified bootstrap")
    treated = frame[frame["treatment"] == 1]
    control = frame[frame["treatment"] == 0]
    conv_test = ab_stats.two_proportion_ztest(
        int(treated["converted"].sum()), len(treated),
        int(control["converted"].sum()), len(control), cfg.alpha,
    )
    rev_test = ab_stats.welch_ttest(
        treated["revenue"].to_numpy(dtype=float), control["revenue"].to_numpy(dtype=float), cfg.alpha
    )
    conv_boot = ab_stats.bootstrap_difference(frame, "converted", cfg.n_bootstrap, cfg.seed, cfg.alpha)
    rev_boot = ab_stats.bootstrap_difference(frame, "revenue", cfg.n_bootstrap, cfg.seed + 7, cfg.alpha)
    mde = ab_stats.minimum_detectable_effect(float(control["converted"].mean()), min(len(treated), len(control)), cfg.alpha)
    holm = ab_stats.holm_adjust({"conversion_rate": conv_test["p_value"], "revenue_per_user": rev_test["p_value"]})
    holm.to_csv(OUTPUT_DIR / "multiple_testing.csv", index=False)
    print(f"      conversion diff = {conv_test['diff'] * 100:.2f} pp (p={conv_test['p_value']:.2e})"
          f" | revenue diff = {rev_test['diff']:.2f} (p={rev_test['p_value']:.2e})")

    print("[4/7] segmented observed uplift (pre-treatment segments)")
    frame = segments.add_segments(frame)
    segment_table = segments.segment_uplift(frame, cfg.alpha)
    segment_table.to_csv(OUTPUT_DIR / "segment_uplift.csv", index=False)

    print("[5/7] T-learner uplift model (train/holdout split by user)")
    train, holdout = uplift.split_train_test(frame, cfg)
    models = uplift.train_t_learner(train, cfg.seed)
    holdout = holdout.copy()
    holdout["predicted_uplift"] = uplift.predict_uplift(models, holdout)
    train = train.copy()
    train["predicted_uplift"] = uplift.predict_uplift(models, train)

    model_curve = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["predicted_uplift"])
    oracle_curve = uplift.qini_curve(holdout["treatment"], holdout["converted"], holdout["true_tau"])
    random_curve = uplift.qini_curve(
        holdout["treatment"], holdout["converted"], np.random.default_rng(cfg.seed).random(len(holdout))
    )
    deciles = uplift.decile_uplift(holdout["treatment"], holdout["converted"], holdout["predicted_uplift"])
    model_curve.to_csv(OUTPUT_DIR / "qini_curve.csv", index=False)
    deciles.to_csv(OUTPUT_DIR / "uplift_deciles.csv", index=False)

    model_metrics = {
        "holdout_users": int(len(holdout)),
        "model_area": uplift.qini_area(model_curve),
        "oracle_area": uplift.qini_area(oracle_curve),
        "random_baseline_area": uplift.random_baseline_area(oracle_curve),
        "random_area": uplift.qini_area(random_curve),
        "normalized_auuc": uplift.normalized_auuc(model_curve, oracle_curve),
        "spearman": float(stats.spearmanr(holdout["predicted_uplift"], holdout["true_tau"]).statistic),
        "top_decile_uplift": float(deciles.iloc[0]["uplift"]),
        "predicted_ate": float(holdout["predicted_uplift"].mean()),
        "train_users": int(len(train)),
    }
    print(f"      holdout={model_metrics['holdout_users']:,}"
          f" | AUUC={model_metrics['normalized_auuc']:.3f}"
          f" | spearman={model_metrics['spearman']:.3f}"
          f" | top-decile uplift={model_metrics['top_decile_uplift'] * 100:.2f} pp")
    if model_metrics["normalized_auuc"] <= 0:
        raise SystemExit("uplift model is no better than random ordering; refusing to publish misleading results")

    print("[6/7] charts")
    figures = [
        charts.balance_plot(balance, FIG_DIR / "balance.png"),
        charts.qini_plot(model_curve, oracle_curve, FIG_DIR / "qini_curve.png", model_metrics["normalized_auuc"]),
        charts.decile_plot(deciles, FIG_DIR / "uplift_deciles.png"),
        charts.segment_plot(segment_table, FIG_DIR / "segment_uplift.png"),
    ]

    print("[7/7] report and summary")
    report_config = {
        "seed": cfg.seed,
        "n_users": cfg.n_users,
        "train_share": cfg.train_share,
        "features": list(FEATURE_COLUMNS),
    }
    report_path = report.build_report(
        truth=sim.truth,
        srm=srm,
        balance=balance,
        data_checks=checks,
        conv_test=conv_test,
        rev_test=rev_test,
        conv_boot=conv_boot,
        rev_boot=rev_boot,
        holm=holm,
        deciles=deciles,
        segments=segment_table,
        model_metrics=model_metrics,
        config=report_config,
        mde=mde,
    )

    within_ci = conv_test["ci_low"] <= sim.truth["ate_conversion"] <= conv_test["ci_high"]
    summary = {
        "config": report_config,
        "sample": {"n": srm["n"], "n_treated": srm["n_treatment"], "n_control": srm["n_control"]},
        "truth": {k: _jsonable(v) for k, v in sim.truth.items()},
        "srm": {k: _jsonable(v) for k, v in srm.items()},
        "max_abs_smd": float(balance["abs_smd"].max()),
        "conversion": {k: _jsonable(v) for k, v in conv_test.items()},
        "revenue": {k: _jsonable(v) for k, v in rev_test.items()},
        "bootstrap": {"conversion": {k: _jsonable(v) for k, v in conv_boot.items()},
                      "revenue": {k: _jsonable(v) for k, v in rev_boot.items()}},
        "holm": holm.to_dict(orient="records"),
        "mde_conversion": float(mde),
        "uplift_model": model_metrics,
        "truth_inside_ci": bool(within_ci),
        "artifacts": {"report": str(report_path), "figures": [str(p) for p in figures]},
        "runtime_seconds": round(time.time() - started, 2),
    }
    (OUTPUT_DIR / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone in {summary['runtime_seconds']}s | ATE {conv_test['diff'] * 100:.2f} pp"
          f" | 95% CI [{conv_test['ci_low'] * 100:.2f}, {conv_test['ci_high'] * 100:.2f}]"
          f" | truth {sim.truth['ate_conversion'] * 100:.2f} pp | truth inside CI: {within_ci}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
