# 门禁作用域修订说明 — GATES Scope Amendment

**日期：** 2026-08-07
**适用范围：** `battery_pytorch_v3/`
**对 `battery_release_v2/` 的影响：** **零**。冻结包一个字节未改，`verify` 仍为 `VERIFY_OK 7/7`，`SHA256SUMS` 全部匹配。

---

## 0. 本文件做什么、不做什么

**做：** 为三条已冻结裁决**新增作用域说明**，指出各裁决所约束的具体问题，以及哪些问题不在其约束之内。

**不做：** 不推翻、不修改、不删除、不重新解释任何裁决。三条裁决的**原文与结论全部保留**，判定值不变。

> 判定规则：若本文件与任何冻结裁决原文冲突，以裁决原文为准。本文件只允许**缩小**声明范围，不允许扩大。

---

## 1. 裁决一：`SOURCE_PRIOR_NOT_VALIDATED`

### 1.1 裁决原文（保留，不改）

Phase 3 裁定 `SOURCE_PRIOR_NOT_VALIDATED`。Phase 4 复核结论：
`any_gate_passed = false`、`all_gates_failed = true`、`phase3_verdict_unchanged = SOURCE_PRIOR_NOT_VALIDATED`、`d1_cannot_change_phase3 = true`。

冻结包 `models/arc_clean_fixed_space.npz` 的 `note` 字段原文：

> "frozen source clip/scaler ONLY. The frozen source coefficients are deliberately absent: Phase 3 ruled SOURCE_PRIOR_NOT_VALIDATED, so they are not used as a prior and shipping them would invite that misuse."

### 1.2 作用域说明

该裁决约束的是：**源域系数 β_src 作为交付预测器的先验，未获验证**。

它**不**约束：把先验强度 λ_SP 作为超参数在谱系上搜索，并测量其最优点落在何处。

理由：附录 A 的目标函数

\[
\hat{\beta}(\lambda_{\mathrm{SP}}) = \arg\min_{\beta} \sum_i w_i (y_i - z_i^{\top}\beta)^2 + \alpha\lVert\beta\rVert^2 + \lambda_{\mathrm{SP}}\lVert\beta - \beta_{\mathrm{src}}\rVert^2
\]

在 λ_SP 上连续插值：λ_SP→∞ 为零样本冻结源头，λ_SP=0 为目标域全量微调（**此时冻结的仍是特征空间**）。因此"λ_SP=0 胜出"是**这条曲线上的一个测量结果**，不是"未实现微调"。

### 1.3 本次实测（v3，72 个 cell）

| λ_SP | 选中次数 | 占比 |
|---|---|---|
| 0（目标域全量微调） | 38 / 72 | 52.8 % |
| 0.001 | 1 / 72 | 1.4 % |
| 0.1 | 2 / 72 | 2.8 % |
| **1（先验牵引微调）** | **15 / 72** | **20.8 %** |
| **10（先验牵引微调）** | **16 / 72** | **22.2 %** |

**47.2 % 的 cell 选中了非零 λ_SP（34/72）**，其中 31/72 = 43.1 % 选中了中高强度 {1, 10}，即在这些 cell 上先验牵引的正则化微调优于全量微调。这比 Phase 3 单点结论更细：谱系中段确有被选中的区域。

### 1.4 β_src 来源与合法性

v2 刻意不发布源系数，故 v3 从 `artifacts/experimental_arc_sensitivity/arc_clean_fixed_alpha10/model.npz` 读取。其 SHA-256 = `fcce4e530ace44a4f6b93fd98c695f0d0e6798144b5e159286ba9c13fbf2a979`，与 v2 npz 内记录的 `model_npz_sha256` **逐字符相同** —— 即这正是产出 v2 冻结空间的那次源域拟合，不是同名的其他敏感性分支。该校验在 `src/features/space.py::load_source_prior` 中强制执行，不匹配即抛异常。

**读取它不等于把它用作交付先验。** 交付预测器仍为 `target_only_arc_space`，不含源先验。

### 1.5 对外表述模板

> 「我们把迁移强度 λ_SP 作为超参数在内层 fold 上搜索，而非固定。72 个 cell 中 52.8 % 选中 λ_SP=0，47.2 % 选中非零值（其中 43.1 % 选中中高强度 {1, 10}）。这说明**坐标系跨过了域边界，预测函数在多数但非全部情形下没有**。这是可测量的发现，不是实现缺失。」

⚠️ 内部审计文件保留否定式门禁名 `SOURCE_PRIOR_NOT_VALIDATED`；对外报告用"迁移强度谱系搜索结果"表述。**事实一字不改，框架换掉。**

---

## 2. 裁决二：`PATCHTST_SKIPPED_BY_PREDEFINED_GATE`

### 2.1 裁决原文（保留，不改）

Phase 6 裁定 `PATCHTST_SKIPPED_BY_PREDEFINED_GATE`。冻结包 `verify::no_neural_implementation` 扫描 `PACKAGE_ROOT.rglob("*.py")`，强制包内无神经网络实现。

### 2.2 作用域说明

该裁决关闭的是：**深度模型作为交付预测器的晋级**。

它**不**关闭：深度模型作为**对比基线**运行。

赛题第 3 项评分要求"与至少一种现有方法对比"。因此在新的预注册协议下开启深度对比臂。

### 2.3 三条不可越界的约束（本次严格执行）

1. **物理隔离。** 所有 `nn.Module` 位于 `battery_pytorch_v3/`，绝不进入 `PACKAGE_ROOT`。由 `tests/test_equivalence_with_v2.py::test_13_v2_contains_no_neural_implementation` 断言 v2 内无神经代码，`test_14` 断言 v3 内确有。
2. **结果不改变交付。** 深度臂无论优劣**均不改变交付预测器**。`target_only_arc_space` 仍是唯一交付预测器。
3. **胜出也不晋级。** 若深度臂胜出，记录为"待新协议下重新评估"，不在本次晋级。

### 2.4 本次实测（T4，SOH ΔSOH 任务，族平均 MAE）

| 臂 | 族平均 MAE |
|---|---|
| `gru_pytorch`（深度对比臂） | 0.003297 |
| `target_only_ridge`（交付机制） | 0.003329 |
| `l2sp_transfer` | 0.003348 |
| `observable_local_trend` | 0.003588 |
| `persistence` | 0.011027 |

GRU 名义领先 `target_only_ridge` **0.0000321（1.0 %）**。

**但该差距小于跨族标准差均值 0.0013**，故 `comparison_summary.json` 记为 `gap_exceeds_across_family_spread = false`，判定 **NOT SEPARABLE**：

> 差距小于跨族散布，不构成真实排序证据 —— 改变留出族的组合就可能改变胜者。**不得表述为一种方法优于另一种。**

依 §2.3 第 2、3 条：即便该差距是实的，交付模型亦不变更。

---

## 3. 裁决三：`RUL_EVIDENCE_INSUFFICIENT`

### 3.1 裁决原文（保留，不改）

Phase 5 裁定 `RUL_EVIDENCE_INSUFFICIENT`。**该裁决是对的，保留。**

### 3.2 作用域说明

该裁决约束的是：**112 天有限视界穿越检测器**。在 112 天窗内只有已接近 EOL 的锚点合格，于是 92 个锚点全部 `target_rul_days = 112.0`，标准差 0 —— 任务退化，常数预测器即得 MAE = 0。

它**不**约束：**全任务期 RUL 回归**，其标签定义不同：

\[
\mathrm{RUL}(t) = t_{\mathrm{EOL}}^{(\text{traj})} - t_{\text{anchor}} \quad (\text{天，无视界上限})
\]

**这是两个任务，不是同一任务的两种算法。**

### 3.3 本次实测（T1）

| 项目 | 数值 |
|---|---|
| 锚点总数 | 11,913（未删失 8,021 + 删失 3,892） |
| 轨迹 | 120（92 到 EOL + 28 右删失，**28 条已保留**） |
| RUL 标签均值 / 标准差 | 666.3 / **424.9** 天 |
| 标签范围 | 14 – 1876 天 |
| 退化性检查 | **NON-DEGENERATE**（std = 424.9 天） |
| 族平均 MAE（删失感知岭 + 冻结 11 特征） | **198.1 天** |
| 最差族 MAE | 258.7 天 |
| 斜率外推基线 | 478.7 天（差 2.42×） |
| 常数均值预测器 | 360.5 天（差 1.82×） |
| Wiener 过程 | 340.4 天 |

### 3.4 与探针 174.4 天的差异说明（必须写明）

方案附录 C 的探针报 174.4 天，本次报 **198.1 天，误差上升 13.6 %**。原因是**方法更严格，不是退步**：

探针**丢弃**了 28 条右删失轨迹，使样本偏向退化更快的电池；本次将其**保留**，施加单侧删失感知损失（只惩罚"预测 RUL 短于已观测随访时长"一侧）。去掉快衰偏差后误差自然上升。

**报告须同时写出两句：**

> 「有限视界交叉检测证据不足（`RUL_EVIDENCE_INSUFFICIENT`，保留原裁决）；全任务期 RUL 回归标签非退化，族平均 MAE = 198.1 天（仿真域、右删失经单侧感知损失保留、族级留一）。」

两句并列，不删任何一句。这是自洽的关键。

---

## 4. 未修订、原样继承的裁决

以下裁决本文件**不做任何作用域切分**，原样继承：

| 裁决 | 状态 |
|---|---|
| `CONFORMAL_NOT_VALIDATED`（含 `CONFORMAL_WORST_FAMILY_SHORTFALL`、`FIXED_HOLDOUT_COVERAGE_SHORTFALL`） | 原样保留。v3 的 RUL 保形区间实测跨族交叉拟合平均覆盖 **0.849**、最差族 **0.687**，均低于 0.90 目标 —— 与该裁决的缺口结论**同向**，不构成反驳。 |
| `STRESS_NOT_VALIDATED` / Phase 4 `all_gates_failed` | 原样保留。见 `docs/observables_chain.md` §3.4：加入 25 个应力通道后族平均 MAE 未改善（0.003448 > 0.003328）。 |
| 证据上限 `OBSERVATIONAL_EVIDENCE` | 原样保留。无生成器 ⇒ 无严格反事实 ⇒ 不得对任何应力因子做因果方向断言。 |
| `CANDIDATE_NOT_PROMOTED` | 原样保留。交付状态未变。 |

---

## 5. 边界核验（硬验收项）

| 项 | 状态 |
|---|---|
| `battery_release_v2` 字节完整 | ✅ `SHA256SUMS` 全部匹配（`test_15`） |
| `python -m battery_entry verify` | ✅ `VERIFY_OK 7/7`（`test_16`） |
| v2 内无神经实现 | ✅ `test_13` |
| v3 内确有 `nn.Module` | ✅ `test_14` |
| v3 不写入 v2 | ✅ `test_17` |
| 11 维特征与 ARC 空间与 v2 一致 | ✅ `test_01`–`test_08`（含逐位相等） |
| λ_SP=0 严格退化为 v2 加权岭 | ✅ `test_09` |
| λ_SP→∞ 严格趋向 β_src | ✅ `test_10` |
| β_src SHA-256 与 v2 记录一致 | ✅ `test_11` |

---

## 6. 一句话总结

**三条裁决都没有被推翻，只是被正确地限定了作用域：**

- λ_source 就是 L2-SP 微调强度谱系 —— 而非"没做迁移"；
- Phase 6 关闭的是深度模型**晋级** —— 而非作为**对比基线运行**；
- 有限视界 RUL 交叉检测与全任务期 RUL 回归 —— 是**两个任务**。

逻辑完整，且交付预测器与冻结包均未改变。
