#!/usr/bin/env python
"""Hillstrom (2008) e-mail experiment: three-arm analysis + T-learner uplift.

This is the **real-data** pipeline of the project (the simulated pipeline remains
for unit tests and for checking the estimators against a known truth).

Data: Kevin Hillstrom, MineThatData E-Mail Analytics And Data Mining Challenge
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Outputs: outputs/hillstrom_summary.json, reports/hillstrom_report.md,
reports/figures/hillstrom_*.png
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "hillstrom.csv"
OUTPUT = PROJECT_ROOT / "outputs"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"

NUM = ["recency", "history"]
CAT = ["channel", "zip_code", "history_segment"]
ARMS = ("Mens E-Mail", "Womens E-Mail")
CONTROL = "No E-Mail"
NAVY = "#1F3A5F"
LIGHT = "#8FB3F5"
RED = "#D64545"
GRAY = "#9AA5B1"


def ensure_data() -> None:
    if DATA.exists():
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, DATA)


def two_prop(x_t: int, n_t: int, x_c: int, n_c: int) -> dict:
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = diff / se if se else 0.0
    return {
        "treated": p_t,
        "control": p_c,
        "diff_pp": diff * 100,
        "ci_low_pp": (diff - 1.96 * se) * 100,
        "ci_high_pp": (diff + 1.96 * se) * 100,
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }


def srm_three_arm(frame: pd.DataFrame) -> dict:
    counts = frame["segment"].value_counts().reindex([CONTROL, *ARMS]).to_numpy()
    expected = np.full(3, len(frame) / 3)
    chi2, p = stats.chisquare(counts, expected)
    return {
        "n": int(len(frame)),
        "counts": {k: int(v) for k, v in zip([CONTROL, *ARMS], counts)},
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_detected": bool(p < 0.001),
    }


def balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    control = frame[frame.segment == CONTROL]
    rows = []
    for arm in ARMS:
        treated = frame[frame.segment == arm]
        for feature in NUM:
            pooled = np.sqrt((treated[feature].var(ddof=1) + control[feature].var(ddof=1)) / 2)
            smd = (treated[feature].mean() - control[feature].mean()) / pooled if pooled else 0.0
            rows.append({"arm": arm, "feature": feature, "level": "(numeric)", "smd": float(smd)})
        for feature in CAT:
            for level in sorted(frame[feature].unique()):
                p_t = float((treated[feature] == level).mean())
                p_c = float((control[feature] == level).mean())
                pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
                smd = (p_t - p_c) / pooled if pooled else 0.0
                rows.append({"arm": arm, "feature": feature, "level": str(level), "smd": float(smd)})
    table = pd.DataFrame(rows)
    table["abs_smd"] = table["smd"].abs()
    return table


def holm(p_values: dict[str, float]) -> list[dict]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, previous = [], 0.0
    for rank, (name, p) in enumerate(items, start=1):
        adjusted = min(1.0, (m - rank + 1) * p)
        adjusted = max(adjusted, previous)
        previous = adjusted
        rows.append({"metric": name, "p_value": p, "adjusted_p": adjusted, "rank": rank})
    return rows


def qini(treatment: np.ndarray, outcome: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-score, kind="mergesort")
    t, y = treatment[order], outcome[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = np.nan_to_num(y_t - y_c * (n_t / np.maximum(n_c, 1)))
    frac = np.arange(1, len(t) + 1) / len(t)
    return pd.DataFrame({"fraction": frac, "qini": q})


def area(curve: pd.DataFrame) -> float:
    x = np.concatenate([[0.0], curve["fraction"].to_numpy()])
    y = np.concatenate([[0.0], curve["qini"].to_numpy()])
    return float(np.trapezoid(y, x))


def deciles(treatment: np.ndarray, outcome: np.ndarray, score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(-score, kind="mergesort")
    rows = []
    for index, idx in enumerate(np.array_split(order, n_bins), start=1):
        t, y = treatment[idx], outcome[idx]
        n_t, n_c = int(t.sum()), int((1 - t).sum())
        rt, rc = float(y[t == 1].mean()), float(y[t == 0].mean())
        se = np.sqrt(rt * (1 - rt) / n_t + rc * (1 - rc) / n_c)
        rows.append({
            "decile": index,
            "n": int(len(idx)),
            "uplift_pp": (rt - rc) * 100,
            "ci_low_pp": (rt - rc - 1.96 * se) * 100,
            "ci_high_pp": (rt - rc + 1.96 * se) * 100,
        })
    return pd.DataFrame(rows)


def uplift_model(frame: pd.DataFrame, arm: str, seed: int = 20240101) -> dict:
    subset = frame[frame.segment.isin([arm, CONTROL])].copy()
    subset["treatment"] = (subset.segment == arm).astype(int)
    rng = np.random.default_rng(seed)
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
        part = train[train.treatment == arm_value]
        model.fit(part[NUM + CAT], part["conversion"])
        models[arm_value] = model

    score = models[1].predict_proba(test[NUM + CAT])[:, 1] - models[0].predict_proba(test[NUM + CAT])[:, 1]
    curve = qini(test["treatment"].to_numpy(), test["conversion"].to_numpy(), score)
    model_area = area(curve)
    random_area = 0.5 * float(curve["qini"].iloc[-1])
    dec = deciles(test["treatment"].to_numpy(), test["conversion"].to_numpy(), score)
    return {
        "arm": arm,
        "n": int(len(subset)),
        "holdout": int(len(test)),
        "qini_area": model_area,
        "random_baseline_area": random_area,
        "qini_lift_over_random": (model_area - random_area) / random_area if random_area else None,
        "overall_uplift_pp": float(
            test.loc[test.treatment == 1, "conversion"].mean()
            - test.loc[test.treatment == 0, "conversion"].mean()) * 100,
        "top_decile_uplift_pp": float(dec.iloc[0]["uplift_pp"]),
        "top_decile_ci_pp": [float(dec.iloc[0]["ci_low_pp"]), float(dec.iloc[0]["ci_high_pp"])],
        "curve": curve,
        "deciles": dec,
    }


def plot_lifts(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in ((axes[0], "conversion", "Conversion lift (pp)"),
                              (axes[1], "visit", "Visit lift (pp)")):
        names = list(results)
        values = [results[a][metric]["diff_pp"] for a in names]
        lower = [results[a][metric]["diff_pp"] - results[a][metric]["ci_low_pp"] for a in names]
        upper = [results[a][metric]["ci_high_pp"] - results[a][metric]["diff_pp"] for a in names]
        ax.bar(names, values, color=[NAVY, LIGHT], width=0.55)
        ax.errorbar(names, values, yerr=[lower, upper], fmt="none", ecolor=GRAY, capsize=4)
        ax.axhline(0, color=RED, linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qini(uplifts: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for arm, color in zip(ARMS, (NAVY, LIGHT)):
        curve = uplifts[arm]["curve"]
        ax.plot(curve.fraction, curve.qini, color=color, linewidth=2,
                label=f"{arm} (Qini +{uplifts[arm]['qini_lift_over_random'] * 100:.0f}% vs random)")
    ax.plot([0, 1], [0, 1], color=RED, linestyle=":", linewidth=1.2, label="Random targeting (illustrative)")
    ax.set_xlabel("Fraction targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Hillstrom uplift curves (holdout)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_deciles(uplifts: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.38
    for offset, arm, color in ((-width / 2, "Mens E-Mail", GRAY), (width / 2, "Womens E-Mail", NAVY)):
        dec = uplifts[arm]["deciles"]
        ax.bar(dec.decile + offset, dec.uplift_pp, width=width, color=color, label=arm)
    ax.axhline(0, color=RED, linewidth=0.8)
    ax.set_xlabel("Predicted-uplift decile (1 = highest)")
    ax.set_ylabel("Observed conversion uplift (pp)")
    ax.set_title("Uplift by decile (holdout)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            spec = formats.get(column)
            cells.append(format(value, spec) if spec and isinstance(value, (int, float)) and pd.notna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ensure_data()
    for directory in (OUTPUT, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATA)
    control = frame[frame.segment == CONTROL]
    srm = srm_three_arm(frame)
    balance = balance_table(frame)

    results: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for arm in ARMS:
        treated = frame[frame.segment == arm]
        conv = two_prop(int(treated.conversion.sum()), len(treated),
                        int(control.conversion.sum()), len(control))
        visit = two_prop(int(treated.visit.sum()), len(treated),
                         int(control.visit.sum()), len(control))
        spend_test = stats.ttest_ind(treated.spend, control.spend, equal_var=False)
        spend_diff = float(treated.spend.mean() - control.spend.mean())
        spend_se = float(np.sqrt(treated.spend.var(ddof=1) / len(treated) + control.spend.var(ddof=1) / len(control)))
        results[arm] = {
            "n": int(len(treated)),
            "conversion": conv,
            "visit": visit,
            "spend": {
                "treated": float(treated.spend.mean()),
                "control": float(control.spend.mean()),
                "diff": spend_diff,
                "ci_low": spend_diff - 1.96 * spend_se,
                "ci_high": spend_diff + 1.96 * spend_se,
                "p_value": float(spend_test.pvalue),
            },
        }
        p_values[f"{arm} / conversion"] = conv["p_value"]
        p_values[f"{arm} / visit"] = visit["p_value"]
        p_values[f"{arm} / spend"] = float(spend_test.pvalue)

    adjusted = holm(p_values)
    uplifts = {arm: uplift_model(frame, arm) for arm in ARMS}

    plot_lifts({arm: {"conversion": results[arm]["conversion"], "visit": results[arm]["visit"]} for arm in ARMS},
               FIGURES / "hillstrom_lift.png")
    plot_qini(uplifts, FIGURES / "hillstrom_qini.png")
    plot_deciles(uplifts, FIGURES / "hillstrom_deciles.png")

    summary = {
        "source": URL,
        "customers": int(len(frame)),
        "segments": {k: int(v) for k, v in frame.segment.value_counts().items()},
        "srm": srm,
        "max_abs_smd": float(balance.abs_smd.max()),
        "arms": {arm: {k: v for k, v in results[arm].items()} for arm in ARMS},
        "holm": adjusted,
        "uplift": {arm: {k: v for k, v in uplifts[arm].items() if k not in ("curve", "deciles")} for arm in ARMS},
        "control": {
            "n": int(len(control)),
            "conversion": float(control.conversion.mean()),
            "visit": float(control.visit.mean()),
            "spend": float(control.spend.mean()),
        },
    }
    (OUTPUT / "hillstrom_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                                    encoding="utf-8")

    report = f"""# Hillstrom 邮件实验：三组对比与增量客群分析

> **数据来源**：Kevin Hillstrom 公开邮件实验（MineThatData E-Mail Analytics Challenge，2008），
> 64,000 名顾客，男士邮件 / 女士邮件 / 无邮件三组，结局为访问（visit）、转化（conversion）和消费（spend）。
> 这是**真实公开数据**，不是模拟数据。

## 1. 分组校验

| 组 | 人数 |
| --- | --- |
{chr(10).join(f"| {k} | {v:,} |" for k, v in summary["segments"].items())}

- 三组 SRM 卡方：χ²={srm['chi2']:.2f}，p={srm['p_value']:.3f}，{'检测到异常' if srm['srm_detected'] else '未检测到异常'}
- 协变量最大 |SMD| = {summary['max_abs_smd']:.3f}（阈值 0.1）

## 2. 三组效果（vs 无邮件）

{md_table(pd.DataFrame([
    {
        "组": arm,
        "转化率": f"{results[arm]['conversion']['treated']*100:.2f}%",
        "转化提升": f"+{results[arm]['conversion']['diff_pp']:.2f} pp",
        "转化 95% CI": f"[{results[arm]['conversion']['ci_low_pp']:.2f}, {results[arm]['conversion']['ci_high_pp']:.2f}]",
        "访问提升": f"+{results[arm]['visit']['diff_pp']:.2f} pp",
        "人均消费提升": f"+${results[arm]['spend']['diff']:.2f}",
    } for arm in ARMS
]), ["组", "转化率", "转化提升", "转化 95% CI", "访问提升", "人均消费提升"])}

Holm 校正后（6 个检验）：

{md_table(pd.DataFrame(adjusted), ["metric", "p_value", "adjusted_p", "rank"], {"p_value": ".2e", "adjusted_p": ".2e"})}

## 3. 增量模型（T-learner）

对每个处理组单独与对照组建模，70/30 按用户切分，特征为 recency、history、channel、zip_code、history_segment。

| 组 | 测试集 | Qini 面积 | 随机基线 | 相对随机 | Top 10% 增量 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(f"| {arm} | {uplifts[arm]['holdout']:,} | {uplifts[arm]['qini_area']:.2f} | {uplifts[arm]['random_baseline_area']:.2f} | {uplifts[arm]['qini_lift_over_random']*100:+.1f}% | {uplifts[arm]['top_decile_uplift_pp']:.2f} pp |" for arm in ARMS)}

![增量曲线](figures/hillstrom_qini.png)

![十分位增量](figures/hillstrom_deciles.png)

## 4. 结论与局限

1. 男士邮件对转化、访问和消费的提升都更大且显著；女士邮件也有正效果，但幅度约为男士的一半。
2. 增量排序在女士邮件组优于随机（Qini 面积高约 {uplifts['Womens E-Mail']['qini_lift_over_random']*100:.0f}%），
   男士邮件组排序不稳定（Qini 低于随机），说明"知道有平均效果"不等于"能可靠挑出高增量人群"。
3. 局限：单零售商、2008 年、邮件渠道、短期窗口；没有成本字段，无法直接算 ROI；结论不能外推到其他市场。

![转化与访问提升](figures/hillstrom_lift.png)
"""
    (REPORTS / "hillstrom_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({"srm": srm, "max_abs_smd": summary["max_abs_smd"],
                      "arms": {a: {"conv_pp": results[a]["conversion"]["diff_pp"],
                                   "visit_pp": results[a]["visit"]["diff_pp"],
                                   "spend": results[a]["spend"]["diff"]} for a in ARMS},
                      "uplift": summary["uplift"]}, ensure_ascii=False, indent=2, default=float))
    print("report:", REPORTS / "hillstrom_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
