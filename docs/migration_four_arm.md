# 电池迁移四臂对照与 Adaptive Transfer Guard

## 四臂对照（同一族级留一协议、同数据预算）

| 设置 | 源域 scaler | 源域 βsrc | 目标域训练 | 研究问题 |
|------|-----------|----------|-----------|---------|
| Target-only | × | × | ✓ | 目标域基线 |
| Feature-transfer | ✓ | × | ✓ | 坐标系是否可迁移 |
| Parameter-transfer | × | ✓ | ✓ | 权重是否有迁移价值 |
| Full L2-SP | ✓ | ✓ | ✓ | 两者结合是否最好 |

> **实测数字（填自 `results/l2sp_summary.json`、`results/comparison_table.csv` 与既有结论文档）：**
>
> 核心结论：**主要稳定收益来自退化表征坐标/预处理统计量迁移（6/6 族优于目标域原生，
> F0=0.003328 vs F4=0.004271，+22.1%）；参数迁移具场景依赖性。**
>
> - **Feature-transfer（特征空间迁移）**：`coordinate_space_control_F4_vs_F0`（lofo_100pct / T0_DELTA_REFERENCE）。
>   族平均 MAE：F0（ARC 迁移空间，交付）= **0.003328**（最差族 0.008706）vs F4（目标域原生空间）= **0.004271**（最差族 0.011855）。
>   提升 = (0.004271 − 0.003328) / 0.004271 ≈ **+22.1%**，**6/6 族一致优于目标域原生**。
>   来源：`docs/observables_chain.md` §3.4-§3.5；权威表述口径见 `RUN.md` §6。
> - **Parameter-transfer（参数迁移，λSP 谱系）**：来自 `results/l2sp_summary.json`
>   `lambda_sp_selection_counts`（合计 72 cell，NESTED_FAMILY_LOFO，选择器只见内层 fold）：
>   λSP=0 选中 **38/72 = 52.8%**；选中非零 **34/72 = 47.2%**（其中中高强度 {λSP=1, 10} 为 31/72 = 43.1%）。
>   来源：`results/l2sp_summary.json` + `docs/internal/GATES_scope_amendment.md` §1.3 + `RUN.md` §6。
> - **四臂结构与"同一族级留一协议、同数据预算"**：来源 `docs/internal/GATES_scope_amendment.md` §1.2-§1.3
>   （λSP 从 0=目标域全量微调到 ∞=零样本冻结源头的连续插值谱系，λSP=0 时冻结的仍是特征空间）。

## Adaptive Transfer Guard（负迁移保护）

λSP 在内层 fold 上按验证集选择：52.8% cell 选中 λSP=0（源参数无帮助时自动抑制），
47.2% 选中非零值（其中 43.1% 选中中高强度 {1,10}，有帮助时保留）。这不是"一半场景迁移失败"，
而是**系统不强迫源参数迁移，能够在跨域偏移较大时自动抑制负迁移**，同时保留已验证有效的特征空间迁移。

## 数字来源

| 数字 | 值 | 出处 |
|------|-----|------|
| F0（ARC 迁移空间，交付）族平均 MAE | 0.003328 | `docs/observables_chain.md` §3.4（`feature_ablation_results.csv`，lofo_100pct / T0_DELTA_REFERENCE） |
| F4（目标域原生空间）族平均 MAE | 0.004271 | `docs/observables_chain.md` §3.4 |
| 特征空间迁移提升 | +22.1% (6/6 族) | `docs/observables_chain.md` §3.5；`RUN.md` §6 权威表述 |
| λSP=0 选中比例 | 52.8% (38/72) | `results/l2sp_summary.json` `lambda_sp_selection_counts`；`RUN.md` §6 |
| λSP 非零选中比例 | 47.2% (34/72)；其中 {1,10} 中高强度 43.1% (31/72) | `results/l2sp_summary.json`；`docs/internal/GATES_scope_amendment.md` §1.3 |
| Target-only 基线（`target_only_ridge`）族平均 MAE | 0.003329 | `results/comparison_summary.json` `ranking_best_first.SOH_DELTA` |
| Full L2-SP（`l2sp_transfer`）族平均 MAE | 0.003348 | `results/comparison_summary.json` `ranking_best_first.SOH_DELTA` |

> 备注：`comparison_table.csv` 记录逐视界 MAE（如 h8/l2sp_transfer 族平均 0.005425 vs target_only 0.005421），
> 其 100pct 预算的族平均（0.003348）与上表一致；四臂表中的 λSP 选择分布以 `l2sp_summary.json` 为准。
> 以上均为**仿真域（STK 衍生孪生）结果，非真实卫星精度**；λSP 搜索与超参选择只使用内层 fold。
