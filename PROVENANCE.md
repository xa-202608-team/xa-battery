# PROVENANCE.md — 权威资产来源与可复现性

**日期：** 2026-08-11
**自检命令：** `$PY -m src.paths` → 应输出 `RESULT : SELF_CONTAINED`

---

## 1. 冻结输入资产（不可重生成，随包校验）

| 资产 | 包内路径 | 来源 | 可否重生成 | 校验 |
|------|---------|------|-----------|------|
| 冻结 ARC clip/scaler（11 维） | `reference/models/arc_clean_fixed_space.npz` | v2 冻结包 | ❌ 源域预处理链未完整归档 | `reference/SHA256SUMS` + `test_15` |
| β_src 源域模型 | `data/source_prior/arc_clean_fixed_alpha10_model.npz` | 研究树 `artifacts/experimental_arc_sensitivity/` | ❌ | `load_source_prior()` 强制 SHA-256 校验 |
| L2 参考健康点 | `reference/data/L2_reference_points_v2_min.csv` | v2 冻结包（STK 仿真产出） | ❌ 生成器缺失 | `reference/SHA256SUMS` |
| L3 窗口表 | `reference/data/L3_windows_v2_arc_clean_fixed_min.csv` | v2 冻结包 | ❌ | `reference/SHA256SUMS` |
| 划分清单 | `reference/data/split_manifest_v2.csv` | v2 冻结包 | ❌ | `reference/SHA256SUMS` |
| 冻结 alpha | `reference/models/frozen_alpha.json` | v2 冻结包（内层 fold 选择记录） | ❌ | `reference/SHA256SUMS` |
| 冻结保形半宽 | `reference/models/conformal_quantiles.json` | v2 冻结包 | ❌ | `reference/SHA256SUMS` |
| 生成配置 | `reference/docs/stk_metadata/generation_config.json` | v2 冻结包 | ❌ | `reference/SHA256SUMS` |

β_src SHA-256 = `fcce4e530ace44a4f6b93fd98c695f0d0e6798144b5e159286ba9c13fbf2a979`，
与 v2 冻结件 `arc_clean_fixed_space.npz::model_npz_sha256` 逐字符相同——即这正是产出 v2 冻结空间的那次源域拟合。

## 2. 公开数据集（随包，可重新清洗）

| 资产 | 包内路径 | 来源 | 大小 |
|------|---------|------|------|
| NASA PCoE 电池退化数据 | `data/source_public/B0005..B0056.npz` + `.meta.json` | NASA PCoE 公开数据集（34 电芯逐循环放电曲线） | 19 MB |

从 `.mat` 原始文件重新清洗为 npz 的流程在 `src/data/`（`mat_loader.py` + `clean.py`）内，可复现。

## 3. 随包遥测（部分可校验）

| 资产 | 包内路径 | 大小 | 可否重生成 |
|------|---------|------|-----------|
| L2 完整参考点（34 列，含环境应力通道） | `data/afterstk/L2_multi_scenario_reference_points.csv` | 7.3 MB | ❌ 生成器缺失 |
| L0 环境时序 | 不随包（184.7 MB） | 80.8 MB | ❌ 生成器缺失 |
| L1 电池遥测 | 不随包 | 103.9 MB | ❌ 生成器缺失 |

`src/twin/verify_chain.py` 可校验 L1→L2 聚合层的一致性（包内 L2 + 不随包 L1），但不能重新生成。

## 4. 结果可复现性

全部 7 个结果入口（pretrain / l2sp / mission_span / prognostic / comparators / fault_injection / fault_evaluation）在上述冻结输入上产生，确定性可复现（float64 CPU 单线程 + 全 RNG 固定）。完整复跑记录见 `docs/battery_reproduce_record.md`。

## 5. 不可重生成项与原因

数字孪生生成器原始代码不在项目内，但 `trajectory_manifest.csv` 完整记录了 120 条轨迹的逐轨迹老化参数，配合环境驱动可做**校准重建**（不是原始生成器源码）。详见 `src/twin/SPEC.md`（状态 `CALIBRATED_RECONSTRUCTION_NOT_ORIGINAL_SOURCE`）。

包内已实现三个孪生模块：
- `reconstruction.py` — 基于记录参数与环境驱动的 SOH 重积分（校准重建版）
- `verify_chain.py` — L1→L2 聚合前向校验
- `fault_injection.py` — L2 健康轨迹后验扰动鲁棒性测试

## 6. v2 冻结包只读保证

`reference/` 子目录是 v2 冻结包（原 `battery_release_v2/`），全程只读。完整性由以下断言保证：

| 断言 | 测试 |
|------|------|
| `reference/SHA256SUMS` 全部匹配 | `test_15` |
| `battery_entry verify` 仍 `VERIFY_OK 7/7` | `test_16` |
| v2 内无神经实现 | `test_13` |
| v3 输出路径不落在 v2 内 | `test_17` |
