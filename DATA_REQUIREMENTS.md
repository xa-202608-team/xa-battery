# DATA_REQUIREMENTS.md — 本包需要什么数据、哪些随包发布、哪些没有

**日期：** 2026-08-07
**打包口径：** 方案 B（不含 afterstk 遥测层）
**自检命令：** `python -m src.paths` → 应输出 `RESULT : SELF_CONTAINED`

---

## 0. 一句话

本包**随包发布**运行 7 个入口所需的全部数据（β_src 与 NASA PCoE 公开数据集在包内，目标域数据在 `reference/` 子目录内）。
**不随包发布**的是 184.7 MB 的 `afterstk` L0/L1 遥测层，它只被 `src/twin/verify_chain.py` 使用，且该模块的全部结论已落盘在 `results/` 与 `docs/` 内。

---

## 1. 随包发布的数据

### 1.1 包内（`battery_pytorch_v3/data/`，18.8 MB）

| 路径 | 内容 | 大小 | 谁需要 |
|---|---|---|---|
| `data/source_prior/arc_clean_fixed_alpha10_model.npz` | 冻结源域模型 = **β_src** 来源（`coef (10,11)`、`intercept (10,)`、`alpha=10.0`、`L=20`） | 0.1 MB | `run_l2sp`、`run_pretrain`、`comparators.run_all` |
| `data/source_public/B0005..B0056.npz` + `.meta.json` | **NASA PCoE 公开电池退化数据集**，34 个电芯，逐循环 capacity / voltage / current / temperature | 18.7 MB | `run_pretrain` |

**为什么 β_src 必须随包带：** 同级的 `reference/models/arc_clean_fixed_space.npz` **刻意不含源系数**——其 `note` 字段原文说明"shipping them would invite that misuse"（因 Phase 3 裁定 `SOURCE_PRIOR_NOT_VALIDATED`）。L2-SP 需要 β_src，故本包自带。

**完整性校验（自动执行，不匹配即抛异常）：**
`src/features/space.py::load_source_prior()` 校验该 npz 的 SHA-256 =

```
fcce4e530ace44a4f6b93fd98c695f0d0e6798144b5e159286ba9c13fbf2a979
```

必须等于 v2 npz 内记录的 `model_npz_sha256`。这确保它就是产出 v2 冻结 ARC 空间的那次源域拟合，而非同名的其他敏感性分支（研究树内有多个 lookalike `model.npz`）。复制入包后已复核 SHA 未变。

### 1.2 冻结包（`reference/` 子目录，29.9 MB）

本包以**只读**方式读取下列 v2 资产。v2 冻结包位于 `reference/` 子目录：

| 路径 | 内容 | 谁需要 |
|---|---|---|
| `models/arc_clean_fixed_space.npz` | 冻结 ARC clip/scaler（11 维），应用而不重拟合 | 全部入口 |
| `models/frozen_alpha.json` | 冻结 alpha 记录（参考） | 参考 |
| `models/conformal_quantiles.json` | 冻结保形半宽 | 参考 |
| `data/L2_reference_points_v2_min.csv` | L2 参考健康点（14,285 行） | `run_mission_span`、`run_prognostic`、`twin.*` |
| `data/L3_windows_v2_arc_clean_fixed_min.csv` | L3 窗口表（34,335 行 × 46 列） | `run_l2sp`、`comparators.run_all` |
| `data/split_manifest_v2.csv` | 120 轨迹族标识与右删失标记 | `run_mission_span`、`run_prognostic`、`twin.*` |
| `docs/stk_metadata/generation_config.json` | 阈值、步长、`adaptation_subsets` | `run_l2sp` |

⚠️ **v2 是冻结包，一个字节都不许改。** 本包从不写入 v2；由 `tests/test_equivalence_with_v2.py` 的 `test_13`–`test_17` 断言（含 `SHA256SUMS` 全量校验与 `verify` 7/7 复跑）。

---

## 2. **不**随包发布的数据

### 2.1 `afterstk` L0/L1 遥测层（184.7 MB）

L2 完整参考点（7.3 MB）已随包（`data/afterstk/L2_multi_scenario_reference_points.csv`）。以下 L0/L1 数据不随包：

| 路径（研究树内） | 内容 | 大小 |
|---|---|---|
| `data/afterstk/L0_multi_scenario_stk_environment.csv` | L0 环境时序，27 列 | 80.8 MB |
| `data/afterstk/L1_multi_scenario_battery_telemetry_72h.csv` | L1 电池遥测，29 列，含 `voltage_v` / `current_a` / `temperature_c` | 103.9 MB |

**唯一使用者：** `src/twin/verify_chain.py`（L1 → L2 聚合前向校验）。

**缺失时的行为：** 该模块**不崩溃**，打印 `STATUS: SKIPPED_INPUT_NOT_SHIPPED` 并 exit 0，同时指明结论落盘位置。由 `tests/test_self_contained.py::test_06` 断言此行为。

**为什么可以不带：** 该模块的全部数值结论已在研究树内跑过并落盘，无需重跑即可查阅：

| 落盘位置 | 内容 |
|---|---|
| `results/twin_chain_verification.json` | 完整数值输出，含 V/I/T 实测包络、逐族一致性、工况多样性 |
| `results/twin_l1_aggregates.csv` | 24 场景逐场景聚合 |
| `docs/observables_chain.md` §0 | V/I/T 实测包络表（G1 的直接答案） |

即 **G1 的证据在包内可查，不依赖这 253 MB**：

```
voltage_v      : 28.546 – 33.262 V
|current_a|    : 至 24.919 A
temperature_c  : 均值 21.73，最高 24.801 °C
dod_orbit      : 均值 0.2199，最大 0.2970
```

**如需重跑：** 把 `afterstk/` 放到 `<研究树>/data/afterstk`，或在原研究树内运行。

### 2.2 Phase 4 应力契约（2.7 MB）

`reports/stk_transfer_v2/04_exposure_stress/` — 29 字段观测性契约与消融结果。

**为什么不带：** 其全部结论已在 `docs/observables_chain.md` §3 **逐字段复述**（29 声明 / 25 准入 / 4 排除，含排除类别与理由、准入规则、消融对照表）。报告引用的是该文档，不是原始 JSON。

---

## 3. 入口可运行性对照表

| 入口 | 方案 B 打包下 | 需要 afterstk |
|---|---|---|
| `python -m src.rul.run_mission_span` | ✅ | — |
| `python -m src.metrics.run_prognostic` | ✅ | — |
| `python -m src.finetune.run_l2sp` | ✅ | — |
| `python -m src.comparators.run_all` | ✅ | — |
| `python -m src.pretrain.run_pretrain` | ✅ | — |
| `python -m src.twin.fault_injection` | ✅ | — |
| `python -m src.twin.run_fault_evaluation` | ✅ | — |
| `python -m src.twin.verify_chain` | ⚠️ 优雅跳过（exit 0） | ✅ 是 |
| `python -m pytest tests/` | ✅ 58 项全通过 | — |

**7/8 入口可跑，第 8 个优雅跳过且结论已留档。**

---

## 4. 路径解析规则（为什么不会"在我机器上能跑"）

`src/paths.py` 对 β_src 与公开数据集采用**包内优先、研究树回退**：

```
data/source_prior/...      ← 优先（随包发布）
   ↓ 不存在则回退
<研究树>/artifacts/...      ← 回退（原始位置）
```

回退是为了让本包在原研究树内也能原样运行（`results/` 里每个数字都是在那里产出的）。但回退**不是静默的**：

- `python -m src.paths` 报告每个输入**实际解析到哪里**，并判定 `SELF_CONTAINED` / `NOT_SELF_CONTAINED`
- `tests/test_self_contained.py::test_02` **断言**没有任何必需输入解析到 bundle 之外
- `tests/test_self_contained.py::test_08` **断言**除 `paths.py` 外无任何模块含硬编码 `D:/SUFAcopy` 路径

所以"因为开发机上恰好有研究树所以能跑"这种情况会被测试抓住，而不是流到评委手上。

---

## 5. 环境要求

| 项 | 值 |
|---|---|
| Python | 3.12（2026-08-10 起与相控阵组件统一） |
| numpy | 2.4.3 |
| pandas | 3.0.3 |
| **torch** | **2.11.0**（cu130 wheel；电池 `_runtime.py` 强制 device=cpu+float64） |
| pytest | 9.0.3 |

见 `requirements.lock`。torch 在本包是**硬依赖**（v2 中是注释掉的可选项）。

**为什么 torch 用 cu130 wheel 但强制 CPU：** 与相控阵组件统一环境（评审 GPU 环境零配置），但 `src/finetune/_runtime.py` 在每个 torch 模块导入前强制 `torch.device("cpu")` + float64 + 单线程 + `use_deterministic_algorithms(True)`。float64 单线程 CPU 路径是"梯度解 = 闭式解"等价性断言成立的前提。
