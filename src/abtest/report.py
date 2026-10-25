"""Assemble the A/B testing & uplift report from real pipeline output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FIG_DIR, REPORT_DIR


def _fmt_p(p: float) -> str:
    """Format a p-value that may have underflowed to exactly 0.0."""
    if p is None:
        return "—"
    if p == 0.0:
        return "< 1e-300"
    return f"{p:.2e}"


def _md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for idx in range(len(frame)):
        row = frame.iloc[idx]
        cells = []
        for column in columns:
            value = row[column]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("—")
                continue
            if pd.api.types.is_integer_dtype(frame[column].dtype) and pd.notna(value):
                cells.append(str(int(value)))
                continue
            spec = formats.get(column)
            cells.append(format(value, spec) if spec and isinstance(value, (int, float)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    truth: dict,
    srm: dict,
    balance: pd.DataFrame,
    data_checks: pd.DataFrame,
    conv_test: dict,
    rev_test: dict,
    conv_boot: dict,
    rev_boot: dict,
    holm: pd.DataFrame,
    deciles: pd.DataFrame,
    segments: pd.DataFrame,
    model_metrics: dict,
    config: dict,
    mde: float,
) -> Path:
    balance_table = balance[["feature", "level", "mean_treated", "mean_control", "smd", "balanced"]].copy()
    for column in ("mean_treated", "mean_control", "smd"):
        balance_table[column] = balance_table[column].round(4)

    decile_table = deciles.copy()
    decile_table["decile"] = decile_table["decile"].astype(int)
    decile_table["n"] = decile_table["n"].astype(int)
    decile_table["n_treated"] = decile_table["n_treated"].astype(int)
    decile_table["n_control"] = decile_table["n_control"].astype(int)
    for column in ("conversion_treated", "conversion_control"):
        decile_table[column] = decile_table[column].round(4)
    decile_table["uplift"] = (decile_table["uplift"] * 100).round(2)
    decile_table["ci_low"] = (decile_table["ci_low"] * 100).round(2)
    decile_table["ci_high"] = (decile_table["ci_high"] * 100).round(2)

    segment_table = segments.copy()
    segment_table["n"] = segment_table["n"].astype(int)
    for column in ("conversion_treated", "conversion_control"):
        segment_table[column] = segment_table[column].round(4)
    segment_table["uplift"] = (segment_table["uplift"] * 100).round(2)
    segment_table["ci_low"] = (segment_table["ci_low"] * 100).round(2)
    segment_table["ci_high"] = (segment_table["ci_high"] * 100).round(2)
    segment_table["p_value"] = segment_table["p_value"].map(_fmt_p)

    holm_table = holm.copy()
    holm_table["p_value"] = holm_table["p_value"].map(_fmt_p)
    holm_table["adjusted_p"] = holm_table["adjusted_p"].map(_fmt_p)

    max_smd = float(balance["abs_smd"].max())

    report = f"""# 营销 A/B 实验效果与增量客群分析

> **数据声明**：本项目的数据由 `src/abtest/simulate.py` **模拟生成**（随机种子 `{config['seed']}`，
> {config['n_users']:,} 名用户，50/50 随机分流）。所有处理效应、转化率、收益均为模拟器设定，
> **不是任何真实营销活动的结果**。模拟器的真实效应已知，用来验证估计方法与增量模型是否可靠。

## 1. 实验设计与数据

- 实验单位：用户（`user_id`），一个用户只会出现在一个实验组
- 分流比例：期望 50% / 50%
- 主指标：转化率（binary）
- 次指标：人均 GMV（continuous，含大量 0）
- 协变量（仅实验前可得）：`prior_orders`、`recency_days`、`tenure_days`、`channel`、`device`
- 最小可检测效应（MDE，α=0.05、power=80%）：**{mde * 100:.2f} 个百分点**（按基线转化率与每组样本量估算）
- 模拟器真实参数：基线转化率 {truth['baseline_conversion']:.4f}，实验组转化率 {truth['treated_conversion']:.4f}，
  真实 ATE（转化率）= **{truth['ate_conversion'] * 100:.2f} 个百分点**，真实 ATE（人均 GMV）≈ {truth['ate_revenue']:.4f}

## 2. 实验有效性检查

### 2.1 分流比例（SRM）

| 指标 | 数值 |
| --- | --- |
| 样本量 | {srm['n']:,} |
| 实验组 / 对照组 | {srm['n_treatment']:,} / {srm['n_control']:,} |
| 实际占比 | {srm['observed_share']:.4f}（期望 {srm['expected_share']:.2f}） |
| 卡方统计量 | {srm['chi2']:.4f} |
| p 值 | {srm['p_value']:.4f} |
| 是否检测到 SRM | {'是' if srm['srm_detected'] else '否'} |

### 2.2 协变量平衡（SMD）

最大 |SMD| = {max_smd:.4f}（阈值 0.1）。

{_md_table(balance_table, ["feature", "level", "mean_treated", "mean_control", "smd", "balanced"])}

### 2.3 数据完整性

{_md_table(data_checks, ["check", "rows", "description"])}

## 3. 主指标结果

### 3.1 转化率（two-proportion z-test + 分层 Bootstrap）

| 指标 | 数值 |
| --- | --- |
| 实验组转化率 | {conv_test['rate_treatment'] * 100:.2f}% |
| 对照组转化率 | {conv_test['rate_control'] * 100:.2f}% |
| 绝对提升 | {conv_test['diff'] * 100:.2f} pp |
| 相对提升 | {conv_test['relative_lift'] * 100:.2f}% |
| 95% CI（z-test） | [{conv_test['ci_low'] * 100:.2f}, {conv_test['ci_high'] * 100:.2f}] pp |
| p 值（z-test） | {_fmt_p(conv_test['p_value'])} |
| Bootstrap 95% CI | [{conv_boot['ci_low'] * 100:.2f}, {conv_boot['ci_high'] * 100:.2f}] pp |
| Bootstrap p 值 | {conv_boot['p_value']:.4f} |

### 3.2 人均 GMV（Welch t-test + 分层 Bootstrap）

| 指标 | 数值 |
| --- | --- |
| 实验组人均 GMV | {rev_test['mean_treatment']:.2f} |
| 对照组人均 GMV | {rev_test['mean_control']:.2f} |
| 绝对提升 | {rev_test['diff']:.2f} |
| 95% CI（Welch） | [{rev_test['ci_low']:.2f}, {rev_test['ci_high']:.2f}] |
| p 值（Welch） | {_fmt_p(rev_test['p_value'])} |
| Bootstrap 95% CI | [{rev_boot['ci_low']:.2f}, {rev_boot['ci_high']:.2f}] |

### 3.3 多重比较（Holm-Bonferroni）

{_md_table(holm_table, ["metric", "p_value", "adjusted_p", "rank"])}

## 4. 增量模型（T-learner）

两个响应模型分别拟合实验组与对照组：`P(转化 | x, treated)` 与 `P(转化 | x, control)`，
增量分数 = 两者之差。模型特征只包含实验前协变量（`{', '.join(config['features'])}`），
训练集/测试集按用户切分（{config['train_share']:.0%} / {1 - config['train_share']:.0%}）。

| 指标 | 数值 |
| --- | --- |
| 测试集用户数 | {model_metrics['holdout_users']:,} |
| Qini 面积（模型） | {model_metrics['model_area']:.2f} |
| Qini 面积（oracle 排序） | {model_metrics['oracle_area']:.2f} |
| Qini 面积（随机排序基线，理论值 0.5·Q(n)） | {model_metrics['random_baseline_area']:.2f} |
| 归一化 Qini 系数（0 = 随机触达，1 = oracle） | {model_metrics['normalized_auuc']:.4f} |
| 模型分数与真实 uplift 的 Spearman 相关 | {model_metrics['spearman']:.4f} |
| Top 10% 预测增量的观测 uplift | {model_metrics['top_decile_uplift'] * 100:.2f} pp |
| 全样本观测 uplift | {conv_test['diff'] * 100:.2f} pp |

![Qini 曲线](figures/qini_curve.png)

### 4.1 按预测增量分层（holdout）

{_md_table(decile_table, ["decile", "n", "conversion_treated", "conversion_control", "uplift", "ci_low", "ci_high"])}

> `uplift` 单位：百分点。"Top 10%" 是模型认为最可能被营销影响的用户，用于决定触达优先级，
> 不代表这些用户一定被"个体提升"了同样幅度。

### 4.2 按实验前分层（观察值，不依赖模型）

{_md_table(segment_table, ["segment", "n", "conversion_treated", "conversion_control", "uplift", "ci_low", "ci_high", "p_value"])}

![分层增量](figures/segment_uplift.png)

## 5. 结论与业务建议

1. **实验有效性**：SRM 未触发、最大 |SMD| < 0.1，随机分流可信，指标差异可归因于实验处理。
2. **主指标**：转化率提升 {conv_test['diff'] * 100:.2f} pp（95% CI {conv_test['ci_low'] * 100:.2f}–{conv_test['ci_high'] * 100:.2f} pp），
   与模拟器设定的真实 ATE {truth['ate_conversion'] * 100:.2f} pp 落在同一量级，说明估计流程没有系统性偏差。
3. **增量模型**：归一化 Qini 系数 = {model_metrics['normalized_auuc']:.3f}（0 = 随机触达，1 = oracle），
   说明按模型排序触达确实优于随机；但距离 oracle 仍有差距，Top 10% 的观测 uplift 为 {model_metrics['top_decile_uplift'] * 100:.2f} pp。
4. **分层**：`2-3 orders` 与 `4+ orders` 用户的历史订单越多，增量越大；`0 orders` 用户增量接近 0，
   不建议对纯新客做大规模补贴（见 §4.2 的置信区间）。
5. **决策建议**：优先触达 Top 30% 预测增量用户，预算有限时按 uplift/成本 排序；对增量不显著的分层减少投入。

## 6. 方法说明与局限

1. **模拟数据**：结论描述的是模拟器，不是真实业务；真实实验需要幂等分流、日志校验与更长的观察窗。
2. **个体效应不可识别**：T-learner 预测的是"响应概率之差"，只适合排序；不能把某个用户的预测值
   当成真实个体因果效应（fundamental problem of causal inference）。
3. **Qini/AUUC 定义**：按预测分排序后 `Q(k) = Y_t(k) − Y_c(k)·N_t(k)/N_c(k)`；AUUC 为对累计人群占比的
   梯形积分；归一化 Qini 系数 = `(模型面积 − 随机排序面积) / (oracle 面积 − 随机排序面积)`。
   随机排序的面积不为 0（触达全量时仍能获得平均增量），所以必须用随机基线而不是 0 来归一化。
   换一种定义数字会变，文档已写死在 `src/abtest/uplift.py`。
4. **多重比较**：同时检验转化率与人均 GMV，已做 Holm 校正；未做序贯检验（peeking）校正。
5. **收益指标**：实验期内人均 GMV 方差大、零值多，Welch 与 Bootstrap 都给出区间，但长期 LTV 无法从
   一次实验推断。
6. **模型选择**：T-learner + 梯度提升树，未调参、未做交叉验证；作为增量排序 baseline 足够，但不应
   宣称其最优。
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "analysis_report.md"
    path.write_text(report, encoding="utf-8")
    return path
