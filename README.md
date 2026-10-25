# A/B Testing & Uplift Modeling

随机实验分析 + T-learner 增量模型：分流校验（SRM）、协变量平衡、转化率/人均 GMV 假设检验、
分层 Bootstrap、Holm 多重比较，以及 Qini / AUUC 增量评估。
**模拟实验数据** —— 由 `src/abtest/simulate.py` 按已知机制生成，真实效应已知，用来检验估计流程而非描述真实业务。

| 项目 | 说明 |
| --- | --- |
| 语言 / 依赖 | Python 3.13（≥3.10）、pandas、numpy、scipy、scikit-learn、matplotlib |
| 数据 | 60,000 名模拟用户，50/50 随机分流，已知真实 ATE |
| 核心能力 | 实验设计校验、假设检验、Bootstrap、T-learner、Qini/AUUC、分层增量、可复现报告 |
| 测试 | `pytest` 39 项全部通过 |
| 报告 | [`reports/analysis_report.md`](reports/analysis_report.md)（4 张图，全部来自实际运行） |

## 1. 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
pytest                          # 39 passed
python scripts/run_pipeline.py  # 模拟 → 校验 → 检验 → 增量模型 → 图表 → 报告
```

完整流水线约 6 秒。产物：

- `outputs/balance.csv`、`outputs/data_checks.csv`：实验有效性检查
- `outputs/segment_uplift.csv`、`outputs/uplift_deciles.csv`、`outputs/qini_curve.csv`
- `outputs/multiple_testing.csv`、`outputs/pipeline_summary.json`
- `reports/analysis_report.md`、`reports/figures/*.png`

## 2. 目录结构

```
├── scripts/run_pipeline.py       # 端到端入口
├── src/abtest/
│   ├── config.py                 # 路径、模拟参数、特征白名单
│   ├── simulate.py               # 模拟实验（含真实个体效应 tau）
│   ├── quality.py                # SRM、SMD 平衡、数据完整性
│   ├── stats.py                  # z-test、Welch、分层 Bootstrap、Holm、MDE
│   ├── uplift.py                 # T-learner、Qini/AUUC、十分位增量
│   ├── segments.py               # 实验前分层的观察增量
│   ├── charts.py                 # 4 张图
│   └── report.py                 # Markdown 报告
├── tests/                        # 39 个测试
└── docs/
    ├── experiment_design.md      # 实验设计、假设与指标
    ├── methods.md                # 统计与增量模型方法说明
    ├── interview_notes.md        # 面试复核（10 问）
    └── acceptance.md             # 验收记录
```

## 3. 实验设计

| 维度 | 选择 |
| --- | --- |
| 实验单位 | 用户（`user_id`），互斥分流 |
| 分流 | 50/50，卡方 SRM 检验（p < 0.001 判定异常） |
| 主指标 | 转化率（binary）→ two-proportion z-test |
| 次指标 | 人均 GMV（连续、零膨胀）→ Welch t-test |
| 不确定性 | 分层 Bootstrap（臂内重采样，2000 次）+ 正态近似 CI |
| 多重比较 | Holm-Bonferroni |
| 增量评估 | T-learner（两个梯度提升分类器）+ Qini/AUUC + 十分位观测增量 |

方法选择原则：**按指标类型选检验**，不机械套用同一种检验；二值指标用比例检验，连续偏态指标用 Welch +
Bootstrap，而不是对两者都跑 t 检验。

## 4. 实际运行结果（seed = 20240101）

### 4.1 实验有效性

| 检查 | 结果 |
| --- | --- |
| 样本量 | 60,000（30,061 / 29,939） |
| SRM 卡方 p 值 | 0.6184（未触发） |
| 最大 \|SMD\| | 0.0156（阈值 0.1，平衡） |
| 数据完整性 | 6 项检查全部 0 违规 |
| MDE（α=0.05，power=80%） | 0.59 pp |

### 4.2 主指标

| 指标 | 实验组 | 对照组 | 差异 | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| 转化率 | 11.15% | 7.19% | **+3.96 pp** | [3.50, 4.42] | < 1e-300 |
| 人均 GMV | 15.26 | 9.93 | **+5.34** | [4.51, 6.16] | 5.43e-37 |

模拟器设定的真实 ATE = **3.98 pp**（转化率），落在估计的 95% CI 内 —— 说明估计流程无系统性偏差。

### 4.3 增量模型

| 指标 | 数值 |
| --- | --- |
| 训练 / 测试用户 | 42,081 / 17,919（按用户切分） |
| Qini 面积（模型 / oracle / 随机基线） | 234.72 / 254.14 / 180.97 |
| 归一化 Qini 系数（0 = 随机，1 = oracle） | **0.7346** |
| 模型分数与真实 uplift 的 Spearman 相关 | 0.8301 |
| Top 10% 预测增量用户的实际 uplift | **10.95 pp**（全样本 3.96 pp） |

### 4.4 实验前分层的观察增量

| 分层 | 用户数 | 转化率增量 | 95% CI |
| --- | --- | --- | --- |
| 0 orders | 18,047 | +0.99 pp | [0.31, 1.67] |
| 1 order | 21,613 | +3.32 pp | [2.58, 4.05] |
| 2-3 orders | 18,268 | +6.88 pp | [5.93, 7.83] |
| 4+ orders | 2,072 | +10.19 pp | [6.86, 13.52] |

> 数字来自 `outputs/pipeline_summary.json` 与 `reports/analysis_report.md`；全部为模拟数据结果。
> 统计检验结果跨环境稳定；模型指标（AUUC / Spearman / Top10%）会随 scikit-learn 版本略有差异
> （1.7.2：0.735 / 0.830 / 10.95 pp；1.9.1：0.756 / 0.830 / 11.01 pp）。

## 5. 关键设计决策

1. **先做有效性检查，再做检验**：SRM 触发或最大 |SMD| ≥ 0.1 时流水线直接失败，不允许在分流坏掉的数据上做分析。
2. **Bootstrap 分层重采样**：在每个实验臂内部重采样，保持分流比例，避免把"臂间样本量变化"混进区间估计。
3. **Qini 归一化用随机基线，而不是 0**：随机触达在覆盖全量人群时也能获得平均增量，其面积不为 0；
   归一化 Qini = (模型 − 随机) / (oracle − 随机)，1 = oracle，0 = 随机，负数 = 比随机还差。
4. **模型只用实验前特征**：`config.FEATURE_COLUMNS` 是白名单，`POST_TREATMENT_COLUMNS` 明确列出禁止项，
   测试断言两者不相交，防止把结果变量泄漏进特征。
5. **训练/测试按用户切分**：同一用户不会同时出现在训练集与测试集，避免同人泄漏。
6. **个体效应不可识别**：报告明确说明 T-learner 输出是排序分数，不是个体因果效应。

## 6. 测试

```bash
pytest -q          # 39 passed
```

| 测试文件 | 覆盖内容 |
| --- | --- |
| `test_stats.py` | z-test 已知算例、CI 宽度公式、Welch、Bootstrap 性质与可复现、Holm 单调性、MDE 公式 |
| `test_quality.py` | SRM 正常/人为失衡、SMD 平衡/人为偏移、数据完整性、期望分流比参数 |
| `test_simulate.py` | 同种子可复现、真实 ATE 与设计一致、效应异质性、收益只属于转化者、特征不含处理后变量 |
| `test_uplift.py` | 手算 Qini 面积、oracle = 1、随机基线线性、反向排序 < 0、十分位划分、T-learner 优于随机、训练/测试互斥 |
| `test_segments.py` | 分层完整划分、CI 覆盖、高频分层增量显著为正 |

## 7. 局限

1. 模拟数据：真实效应由设定决定，结论描述的是模拟器，不是真实营销活动。
2. 二值指标 + 有限样本下行级 Qini 曲线噪声较大，AUUC 只用于排序比较，不应解读为精确收益。
3. 未做序贯检验（peeking）校正；实际实验若多次查看，需 alpha spending。
4. 收益指标只观察实验窗口，不能推断长期 LTV；退款、复购等后续行为未建模。
5. T-learner 未调参、未做交叉验证，作为基线方法；更强的增量模型（X-learner、因果森林）可作为后续。

## 8. English summary

A randomised-experiment analysis toolkit on **simulated** data with a known treatment effect:
SRM and covariate-balance gates, type-appropriate tests (two-proportion z-test, Welch, stratified
bootstrap), Holm correction, and a T-learner whose ranking is graded with an explicitly defined
Qini/AUUC (random-baseline normalised). Leakage is prevented by a feature whitelist and a
user-level train/holdout split. The pipeline refuses to report results when randomisation is
broken, and the report states clearly that model scores are ranking signals, not individual causal
effects.
