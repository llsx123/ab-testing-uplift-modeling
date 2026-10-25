# 项目二：面试复核笔记

## Q1. 实验有效性为什么要先看 SRM？p 值阈值为什么是 0.001？

**答题要点**：分流比例异常意味着随机化坏掉（埋点重复、分流哈希偏斜、部分用户被排除），两组不再可比，
此后的显著性没有意义。SRM 用卡方拟合优度检验，阈值取 0.001 是为了避免把正常波动当故障（护栏不是科研假设），
一旦触发流水线直接中止。

**代码位置**：`src/abtest/quality.py:27`（srm_test）、`scripts/run_pipeline.py` 第 2 步的中止逻辑。

## Q2. 转化率和人均 GMV 分别用什么检验？为什么不都用 t 检验？

**答题要点**：转化率是二值比例，用两比例 z 检验（非合并标准误）；人均 GMV 是连续、零膨胀、右偏，
用 Welch t 检验（不假设等方差）。对二值指标跑 t 检验在数学上不算错（0/1 变量的均值检验等价），
但比例检验的方差结构更明确，且报告口径更标准。

**代码位置**：`src/abtest/stats.py:20`、`src/abtest/stats.py:42`。

**追问**：GMV 严重右偏，t 检验还可靠吗？→ 大样本下 Welch 依中心极限定理近似正态，同时用 Bootstrap 交叉验证；
若样本小应改用 Mann-Whitney 或对数变换。

## Q3. Bootstrap 为什么要分层？p 值为什么可能是 0？

**答题要点**：在每个实验臂内部重采样，保持臂内样本量不变，避免把"臂间样本量波动"混入差值分布。
有限次重采样无法得到 p=0，因此当所有重采样差值同号时报告 `1/(B+1)` 作为分辨率下限。

**代码位置**：`src/abtest/stats.py:64`（bootstrap_difference）。

## Q4. 人均 GMV 的分母为什么是组内全部用户？

**答题要点**：如果只除以转化用户，就把"转化率提升"和"客单价提升"混在一起；人均 GMV 衡量的是"每个被触达
用户带来的收入"，分母必须是全部被分配的用户（含未转化记 0）。这也是零膨胀的来源。

**代码位置**：`src/abtest/stats.py:42`（输入是两组的 revenue 数组）、`src/abtest/simulate.py:83`（未转化 revenue=0）。

## Q5. 多个指标同时检验怎么控制假阳性？

**答题要点**：Holm-Bonferroni：p 值升序，第 k 个乘以 (m-k+1)，并保持单调；比 Bonferroni 温和，
控制 FWER。本项目一次检验转化率与人均 GMV 两个指标。

**代码位置**：`src/abtest/stats.py:113`、`outputs/multiple_testing.csv`。

## Q6. 什么是 T-learner？为什么用两个模型而不是一个？

**答题要点**：T-learner 分别在实验组和对照组上拟合响应模型 `μ1(x)`、`μ0(x)`，增量分数为二者之差。
两个模型让两臂的响应曲面各自自由（对照组没有处理效应），避免用一个含 treatment 交互项的模型强行共享结构。
代价是两模型误差叠加，小样本段不稳定。

**代码位置**：`src/abtest/uplift.py:67`（train_t_learner）、`src/abtest/uplift.py:78`（predict_uplift）。

## Q7. Qini 曲线怎么算？为什么随机排序的面积不是 0？

**答题要点**：按预测增量降序，对前 k 人计算 `Q(k) = Y_t(k) − Y_c(k)·N_t(k)/N_c(k)`；随机排序的 Q 在期望上
是从 0 到 Q(n) 的直线，面积 = 0.5·Q(n)。因此归一化必须减去随机基线，否则"随机"也能拿到约 0.5 的分数，
会高估模型。

**代码位置**：`src/abtest/uplift.py:85`（qini_curve）、`src/abtest/uplift.py:129`（random_baseline_area）、
`src/abtest/uplift.py:134`（normalized_auuc）。

**追问**：为什么不用 NTILE 分箱算 uplift？→ 分箱会丢掉排序信息，Qini 用的是曲线下面积。

## Q8. 增量模型的评估为什么必须用独立测试集？怎么防止数据泄漏？

**答题要点**：用训练集评估会高估（模型记住了噪声）。本项目按 `user_id` 切分训练/测试，且特征白名单只包含
实验前协变量，`config.POST_TREATMENT_COLUMNS` 明确禁止 `treatment/converted/revenue/true_tau`，
`tests/test_simulate.py::test_features_are_pre_treatment_only` 断言两者不相交。

**代码位置**：`src/abtest/config.py:19`（白名单）、`src/abtest/uplift.py:38`（split_train_test）。

## Q9. 预测的个体增量能不能直接说成"这个用户被提升了 x%"？

**答题要点**：不能。因果推断的根本问题是同一个用户只能观察到一种结果，`τ̂(x)` 是两个响应概率之差，
只适合排序和分组决策；把它当个体因果效应会犯"用预测值代替反事实"的错误。报告与 README 都写明这一点。

**代码位置**：`src/abtest/uplift.py` 模块 docstring、`reports/analysis_report.md` §6.2。

## Q10. 模拟器的真实 ATE 是怎么用的？为什么说估计流程没有系统性偏差？

**答题要点**：模拟器已知每个用户的 `p0/p1`，因此真实 ATE = mean(p1−p0) 可精确计算（3.98 pp）。
估计值 3.96 pp、95% CI [3.50, 4.42] 包含真值，说明检验流程没有系统性高估/低估。
注意这只验证了"流程正确"，不代表模拟设定等于真实业务。

**代码位置**：`src/abtest/simulate.py:83`（truth 计算）、`scripts/run_pipeline.py`（truth_inside_ci）。

---

## 自测清单

- [ ] 手写两比例 z 检验的 SE 与 CI
- [ ] 解释为什么 SRM 要在做主分析之前检查
- [ ] 说出分层 Bootstrap 与普通 Bootstrap 的区别
- [ ] 解释 Qini 公式中 `N_t/N_c` 缩放的作用
- [ ] 推导随机排序的 Qini 面积为什么是 0.5·Q(n)
- [ ] 区分"预测的个体增量"与"真实的个体因果效应"
- [ ] 说明 T-learner 与 S-learner 的差别与取舍
- [ ] 解释为什么训练/测试要按用户切分
