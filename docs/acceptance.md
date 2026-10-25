# 项目二验收记录

验收时间：2026-10-25　｜　环境：Windows 11 + Python 3.13.3

## 1. 干净环境复现

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

| 检查项 | 结果 |
| --- | --- |
| 依赖安装 | 成功（venv：pandas 3.1.0 / numpy 2.5.4 / scipy 1.18.1 / scikit-learn 1.9.1 / matplotlib 3.11.2） |
| 测试 | `39 passed`（系统环境 scikit-learn 1.7.2 亦为 39 passed） |
| 完整流水线 | 成功，约 4–5 秒 |
| 统计结论可复现 | 转化率 +3.96 pp，95% CI [3.50, 4.42]，真实 ATE 3.98 pp 落在 CI 内（两环境一致） |

**已知环境敏感性**：模型相关指标随 scikit-learn 版本略有差异（1.7.2：归一化 Qini 0.735、Top10% 11.0 pp；
1.9.1：0.756、11.0 pp），属于正常的训练随机性差异，统计检验结果不受影响。

## 2. 验收清单

| 要求 | 证据 | 结论 |
| --- | --- | --- |
| 实验单位、处理组/对照组定义正确 | `docs/experiment_design.md`；`test_quality.py` 分流比与互斥性 | ✅ |
| 统计检验与指标类型匹配 | 转化率 → 两比例 z 检验；人均 GMV → Welch；`test_stats.py` 已知算例 | ✅ |
| 置信区间与 Bootstrap 经过验证 | z-test CI 宽度公式测试、Bootstrap 可复现/性质测试 | ✅ |
| 训练/测试严格隔离 | `test_uplift.py::test_train_and_holdout_are_disjoint`（user_id 不相交） | ✅ |
| 不存在明显数据泄漏 | 特征白名单，`test_simulate.py::test_features_are_pre_treatment_only` | ✅ |
| Qini/AUUC 定义清晰 | `src/abtest/uplift.py` 模块 docstring + `docs/methods.md`；手算面积测试 | ✅ |
| 模拟结果符合生成机制 | 真实 ATE 3.98 pp vs 估计 3.96 pp（CI 覆盖）；Spearman 0.83 | ✅ |
| README 数字来自实际运行 | `outputs/pipeline_summary.json` 与报告一致 | ✅ |
| 数据质量/分流检查能拦截问题 | 人为注入 55/45 分流后 `srm_detected=True`（测试）；流水线中止逻辑 | ✅ |
| 无凭证泄漏 / 虚构结论 | 无 API Key；所有结论标注为模拟数据 | ✅ |

## 3. 已知限制（不阻塞验收）

1. 二值指标 + 有限测试集导致 Qini 曲线噪声，AUUC 仅用于排序比较。
2. 未做 peeking 校正、新奇效应与干扰建模。
3. 模型未调参，作为基线方法；指标对 sklearn 版本敏感。

## 4. 验收结论

**通过。** 可以进入项目三开发。
