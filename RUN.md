# RUN.md — 电池组件运行说明 (battery_pytorch_v3)

**环境统一（2026-08-10）：与相控阵组件共用 Python 3.12 + torch 2.11。**
**六节：** 数据准备 / 训练 / 微调 / 预测 / 可信度验证 / 复现

---

## 0. 前置：解释器与环境

本包使用 conda env `pytorch_gpu`（与相控阵组件同一环境）：

```
/f/anaconda3/envs/pytorch_gpu/python.exe   (bash)
```

Python 3.12.13。依赖实测版本（见 `requirements.lock`）：

| 包 | 版本 |
|---|---|
| numpy | 2.4.3 |
| pandas | 3.0.3 |
| **torch** | **2.11.0** (cu130 wheel; 电池 `_runtime.py` 强制 device=cpu+float64) |
| **pytest** | **9.0.3** |

> 电池代码只用 numpy/pandas/torch/pytest 四个第三方库（grep import 核实）。
> torch 装 cu130 wheel（与相控阵统一，评审 GPU 环境零配置），但 `src/finetune/_runtime.py` 强制 `torch.device("cpu")` + float64 + 单线程，所以电池代码始终跑 CPU（float64 CPU 是梯度vs闭式解等价检验的前提）。

会话设置（后续命令沿用 `$PY`，在电池组件根目录执行）：

```bash
cd "XA-202608_最终交付/03_代码/components/battery"
PY=/f/anaconda3/envs/pytorch_gpu/python.exe
```

确定性设置由 `src/finetune/_runtime.py` 在**每个 torch 模块导入前**完成：float64、CPU、单线程、`use_deterministic_algorithms(True)`。该设置保证 gradient-vs-closed-form 等价性检验（`test_torch_matches_closed_form.py`）是确定性断言。

---

## 1. 数据准备

本包**不生成任何仿真数据**，只读取已有资产。所有输入均为只读。

### 1.1 目标域（仿真域，STK 数字孪生产物）

| 数据 | 路径 | 用途 |
|---|---|---|
| L2 参考健康点（5 列，14,285 行） | `reference/data/L2_reference_points_v2_min.csv` | RUL 锚点、预警阈值穿越 |
| L3 窗口表（46 列，34,335 行） | `reference/data/L3_windows_v2_arc_clean_fixed_min.csv` | SOH ΔSOH 任务设计矩阵 |
| 划分清单（120 轨迹 × 16 列） | `reference/data/split_manifest_v2.csv` | 族标识、右删失标记 |
| 冻结 ARC 空间 | `reference/models/arc_clean_fixed_space.npz` | clip/scaler，**应用而不重拟合** |
| 冻结 alpha | `reference/models/frozen_alpha.json` | 参考（v3 自行在内层 fold 选 alpha） |
| 生成配置 | `reference/docs/stk_metadata/generation_config.json` | 阈值、步长、adaptation_subsets |

### 1.2 源域（公开数据集）

| 数据 | 路径 | 说明 |
|---|---|---|
| NASA PCoE 电池退化数据 | `data/source_public/B0005..B0056.npz`（+ `.meta.json`） | **34 个电芯**，逐循环放电曲线（capacity/voltage/current/temperature） |
| 源域权威模型（β_src 来源） | `artifacts/experimental_arc_sensitivity/arc_clean_fixed_alpha10/model.npz` | SHA-256 与 v2 记录的 `model_npz_sha256` 一致，见 §3.2 |

源域 20 点参考窗构建（自检，不写文件）：

```bash
$PY -c "from src.data.source_dataset import build_windows, summarise; import json; print(json.dumps(summarise(build_windows(8)), indent=1))"
```

预期：`n_windows = 1670`，`n_cells_used = 25`，`n_cells_skipped = 9`（可用循环数 < 28 的电芯被跳过并逐个列出，不静默丢弃）。

### 1.3 遥测层（V/I/T，用于可观测量链路说明）

| 数据 | 路径 | 大小 | 随包 |
|---|---|---|---|
| L0 环境时序（27 列） | `data/afterstk/L0_multi_scenario_stk_environment.csv` | 80.8 MB | ❌ 不随包 |
| L1 电池遥测（29 列，含 `voltage_v`/`current_a`/`temperature_c`） | `data/afterstk/L1_multi_scenario_battery_telemetry_72h.csv` | 103.9 MB | ❌ 不随包 |
| L2 完整参考点（34 列，含环境应力通道） | `data/afterstk/L2_multi_scenario_reference_points.csv` | 7.3 MB | ✅ 随包 |

L0/L1 不随包（合计 184.7 MB），缺失时 `verify_chain` 优雅跳过（exit 0），结论已落盘在 `results/twin_chain_verification.json`。

L1 → L2 聚合前向校验（流式读取，不整体载入）：

```bash
$PY -m src.twin.verify_chain
```

数字孪生 SOH 重建（校准版，自包含输入）：

```bash
$PY -m src.twin.run_reconstruction --output-dir results/twin_reconstruction
```

原始生成器源码不在项目内，但逐轨迹参数完整记录，可做**校准重建**。详见 `src/twin/SPEC.md`（状态 `CALIBRATED_RECONSTRUCTION_NOT_ORIGINAL_SOURCE`）。

---

## 2. 训练（源域预训练）

```bash
$PY -m src.pretrain.run_pretrain
```

在 NASA PCoE 公开数据上用 PyTorch 训练线性头，并拟合 clip/scaler。

### 2.1 参数逐个说明

| 参数 | 位置 | 值 | 含义与为何如此 |
|---|---|---|---|
| `HORIZON_CYCLES` | `run_pretrain.py` | 8 | 源域视界，单位**循环**。取 8 以与目标域最长视界（8 个 14 天步）的步数可比。⚠️ 循环 ≠ 天，两域单位不同，代码中显式声明不混用。 |
| `CLIP_Q` | `run_pretrain.py` | (1.0, 99.0) | 1%–99% 分位裁剪，归档技术方案记载的源模型口径。 |
| `CONTEXT_LENGTH` | `frozen11.py` | 20 | 冻结路由上下文长度，来自源模型 manifest，非本处选择。 |
| `alpha` | 读自源模型 npz | 10.0 | 取自冻结源模型自身记录，**不重新调参**。 |
| `lr` | `train_l2sp_gradient` | 0.05 | AdamW 学习率。 |
| `max_epochs` | 同上 | 20000 | 上限；实际由早停终止。 |
| `patience` | 同上 | 800 | 早停耐心（以**目标函数**为准，见下）。 |
| `tol` | 同上 | 1e-14 | 目标函数改善阈值。 |
| `seed` | 同上 | 0 | 全 RNG 固定；本路径实际无随机性（零初始化、全批量、无 dropout）。 |
| `refine_lbfgs` | 同上 | True | AdamW 后接 LBFGS 精修，理由见 §3.4。 |

**早停为何以目标函数而非留出集为准：** 闭式解是该目标函数的已知最优点，因此"目标函数不再改善即停"是诚实判据。跨 λ_SP 的**超参选择**确实使用留出内层 fold（见 §3）。

### 2.2 输出

- `results/pretrain_source/source_pretrained.npz` — `beta_pytorch`、`beta_closed_form`、clip/scaler
- `results/pretrain_source/pretrain_report.json` — 含 `max_abs_gradient_minus_closed_form`

### 2.3 口径红线

本步**演示**"公开数据集 → PyTorch 预训练"可执行。**交付链使用的仍是 v2 的冻结 clip/scaler 与冻结源系数**。新拟合结果与冻结值不逐位相同（原始完整预处理链未记录在归档中），差异**如实报告**，不粉饰、不替换冻结件。

---

## 3. 微调（L2-SP 迁移强度谱系搜索）

```bash
$PY -m src.finetune.run_l2sp
```

### 3.1 目标函数

L2-SP：最小化 `sum_i w_i (y_i - z_i·beta)^2 + alpha·||beta||^2 + lambda_SP·||beta - beta_src||^2`

闭式解：`beta_hat = (Z'WZ + (alpha + lambda_SP)I)^-1 (Z'Wy + lambda_SP·beta_src)`

### 3.2 参数逐个说明

| 参数 | 值 | 含义与为何如此 |
|---|---|---|
| `LAMBDA_SP_GRID` | (0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e4) | 迁移强度谱系，**拟合前锁定**。含 0（目标域全量微调）与 1e4（趋近零样本冻结源头）两端，使搜索能真正到达任一端。 |
| `ALPHA_GRID` | (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100) | 岭强度网格，拟合前锁定。 |
| `HORIZONS` | (2, 4, 8) 步 = 28/56/112 天 | v2 支持的视界；不外推超出源模型支持（`for_horizon` 越界抛异常）。 |
| 权重 `w_i` | 轨迹等权 + 族平衡 | 归一化至均值 1.0，使 alpha 与 lambda_SP 保持与冻结无权拟合可比。 |
| 外层协议 | 族级留一（LOFO） | `environment_family_id` 是严格隔离键。 |
| 内层选择 | (alpha, lambda_SP) 联合，仅在内层 fold | `selector_saw_outer_test = False` 逐行记录。 |
| `N_GRADIENT_CHECKS` | 6 | 抽样做 AdamW 交叉核对的 cell 数（全网格跑 20k epoch 无额外信息量，专项测试已覆盖）。 |
| 预算阶梯 | 每族 2/5/10/20 条轨迹 | 即 10/25/50/100%。见 §3.4。 |

### 3.3 beta_src 来源（关键）

v2 的 `arc_clean_fixed_space.npz` **刻意不含源系数**（其 `note` 字段明示，因 Phase 3 裁定 `SOURCE_PRIOR_NOT_VALIDATED`）。故 beta_src 读自：

```
artifacts/experimental_arc_sensitivity/arc_clean_fixed_alpha10/model.npz
```

`load_source_prior()` **强制校验**其 SHA-256 等于 v2 npz 内记录的 `model_npz_sha256`（`fcce4e53...f2a979`），不匹配即抛异常——确保这就是产出 v2 冻结空间的那次拟合，而非同名的其他敏感性分支。

### 3.4 适配曲线的预算口径（必须说明）

v2 的 `generation_config.json` 记 `adaptation_subsets` 为 4/10/20/40，是对 40 条 `target_train` 轨迹池（2 族 × 20）的比例。本包在**族级留一**下评估，训练时有 **5 个族**，故同样的百分比按**每族**对该族 20 条轨迹表达：

| 标签 | 每族轨迹数 | 每折训练轨迹数 | v2 原配置值 |
|---|---|---|---|
| 10pct | 2 | 10 | 4 |
| 25pct | 5 | 25 | 10 |
| 50pct | 10 | 50 | 20 |
| 100pct | 20 | 100 | 40 |

百分比相同，只是分母用协议自身的口径表述；v2 原值一并写入输出便于追溯。**子集按 trajectory_id 排序确定，不重新抽样**（重抽会破坏与冻结 Phase 3/4 表的可比性）。

### 3.5 输出

- `results/l2sp_cells.csv` — 72 个 cell 逐行（含所选 alpha/lambda_SP、prior_retention、条件数）
- `results/adaptation_curve.csv` — 10/25/50/100% 精度曲线
- `results/l2sp_summary.json` — lambda_SP 选择分布、逐视界结果、梯度核对

### 3.6 梯度解 vs 闭式解的两类核对

| 核对 | 判据 | 说明 |
|---|---|---|
| **目标函数一致性** | 相对差 < 1e-6 | **主判据，与条件数无关**。两解算器到达同一最小值。 |
| **系数一致性** | ≤ 曲率界 | 界由曲率**推导**而非选定：目标函数严格二次，故把目标函数收敛到 dL 仍允许 `sqrt(dL / lambda_min)` 的系数误差，`lambda_min = alpha + lambda_SP`。良态 cell 上该界**紧于** 1e-6。 |

11 维几何特征近共线：alpha 小时法方程条件数可达 ~1e8，此时固定 1e-6 的**系数**匹配在数值上不可达。这是**设计矩阵的性质，不是优化器缺陷**，如实报告而非放宽容差或悄悄剔除该 cell。因此 AdamW 后接 LBFGS 精修（二阶方法利用曲率），两者最小化的是**同一目标函数**，精修只改数值路径不改目标。

---

## 4. 预测（RUL、提前性、对比臂）

### 4.1 全任务期 RUL

```bash
$PY -m src.rul.run_mission_span
```

| 参数 | 值 | 说明 |
|---|---|---|
| 标签 | `RUL(t) = t_EOL - t_anchor` | 天，**无视界上限**。与 112 天有限视界穿越检测器是**两个任务**。 |
| `ALPHA_GRID` | (1e-3 … 100) | 内层 fold 选择 |
| `LAMBDA_CENSORED` | 1.0 | 删失 hinge 权重：违约的删失下界与同等大小的观测误差等价代价。**声明值，非按结果调**。 |
| `COVERAGE` | 0.90 | 保形目标覆盖 |
| `MAX_ACTIVE_SET_ITERS` | 50 | 活动集迭代上限；触顶会**报告**而非静默 |
| 特征通道 | `soh_observed` | 部署可得 |
| EOL 标签通道 | `soh_true` | **仅评估** |

**右删失处理：** 28 条未到 EOL 的轨迹**保留**（探针曾丢弃，导致样本偏向快衰电池）。删失锚点不给点标签，而给下界 = 已观测随访时长，损失为**单侧**平方 hinge：只惩罚"预测 RUL 短于已观测随访时长"。求解用活动集迭代（`_solve_ridge_with_intercept` **联合优化 coef+intercept**，目标函数为凸分段二次，有限个活动集 ⇒ 收敛到全局最优），复用冻结加权岭闭式路径，不引入学习率/调度/容差三件套。此前版本分步求解（先 `solve_ridge_weighted` 无截距列，后补 `weighted_intercept`）不等价于联合优化，导致 test_08 失败被误判为"数值容差"；修复后 58 项全绿。

### 4.2 提前性指标

```bash
$PY -m src.metrics.run_prognostic
```

| 参数 | 值 | 说明 |
|---|---|---|
| `WARN_SOH` | 0.80 | 预警阈值 |
| `EOL_SOH` | 0.70 | 寿命终止阈值 |
| `WARN_HORIZON_STEPS` | 8（112 天） | 预警投影视界，v2 最长支持视界；再远即外推超出冻结源模型支持 |
| `ALPHA_PH` | 0.2 | PH 相对误差带 |
| 分辨率 | 14 天 | Δt_warn 仅在 ±14 天内有意义，按分布报告而非单一数字 |

报告三个**分开**的量：预警提前期 Δt_warn、严格 Saxena PH、格点下限 PH + 可信窗口。三者结论不同，不互相替代（详见 §6）。

### 4.3 对比臂

```bash
$PY -m src.comparators.run_all
```

七臂，同一族级留一协议：`persistence`、`observable_local_trend`、`wiener_process`、`double_exp_particle_filter`、`gru_pytorch`、`target_only_ridge`、`l2sp_transfer`（RUL 任务另有 `constant_mean`、`slope_extrapolation`、`censoring_aware_ridge_frozen11`）。

| 参数 | 值 | 说明 |
|---|---|---|
| Wiener `mu`/`sigma` | 逐轨迹 MLE | 由增量估计，无需训练集 ⇒ 公平的"现有方法"而非稻草人 |
| PF `n_particles` | 2000 | |
| PF `obs_sigma` | 0.005 | **声明值**，取 v2 L3 表锚点噪声残差量级 |
| PF 重采样 | ESS < n/2 时 | 含 roughening 抖动防粒子退化 |
| PF 子采样 | 每 10 个锚点 | 成本考虑，**事先声明且在输出中报告** |
| PF 种子 | `stable_seed(trajectory_id)` | 确定性，绝不用时钟播种 |
| GRU `HIDDEN` | 32 | 小网：训练集仅数千条 20 点序列，大网会过拟合使对比反向失真 |
| GRU 输入 | **原始 SOH 历史**，非 11 维特征 | 否则等于重参数化的岭回归，对比无意义 |
| GRU 验证集 | 一个**内层族** | 随机行切分会泄漏（同轨迹窗口高度重叠） |

⚠️ 深度臂是**对比基线，不是交付模型**。无论优劣均不改变交付预测器。见 `docs/internal/GATES_scope_amendment.md` §2。

### 4.4 故障注入（L2 健康轨迹后验扰动鲁棒性测试）

```bash
$PY -m src.twin.fault_injection
$PY -m src.twin.run_fault_evaluation
```

四类故障（内阻阶跃 `Rint×1.6`、容量跳变 `Q 突降`、热控偏置 `T+偏置`、深放电 `DoD↑`），各注入 15 条轨迹，**类间不重叠**（60/120）。注入作用于**已有的 L2 健康轨迹**（后验扰动），经 `src/twin/approx_propagation.py` 近似解析映射到 SOH 增量。幅值在任何模型跑之前声明，给定 `(trajectory_id, fault_type)` 完全确定。

⚠️ **此为 L2 健康轨迹层的后验扰动鲁棒性测试，不是物理级故障仿真**（SPEC.md §4 老化系数未记录，不重建完整老化方程传播）。用途：测试预测器对扰动的优雅降级。**不可用于**：声称已验证的故障传播模型、定量故障严重度推断、任何因果断言。

---

## 5. 复现

### 5.1 完整自检序列

在电池组件根目录（`XA-202608_最终交付/03_代码/components/battery`）执行：

```bash
# 硬验收项：v2 冻结包必须仍为 7/7
cd reference && PYTHONPATH="..:." $PY -m battery_entry verify
$PY -m battery_entry reproduce --mode quick --output /tmp/reproduce_check
cd ..

# v3 测试套件（58 passed 全绿；test_08 active-set 联合优化修复后不再 deselect）
$PY -m pytest tests/ -q

# v3 全流程
$PY -m src.pretrain.run_pretrain        # 源域预训练（torch LinearHead）
$PY -m src.finetune.run_l2sp            # L2-SP 迁移（闭式解+torch核对）
$PY -m src.rul.run_mission_span         # 全任务期 RUL
$PY -m src.metrics.run_prognostic       # 提前性指标
$PY -m src.comparators.run_all          # 七臂对比

# 可信度验证脚本（本次改造新增）
$PY scripts/verify_directionality.py    # 方向性校核（六项机理断言）
$PY scripts/verify_timescale.py         # 时间尺度分离（72h模板 vs 14d栅格）
$PY scripts/sensitivity_analysis.py     # ±20% 参数敏感性
$PY -m src.twin.fault_injection         # 故障注入（物理量+近似传播）
$PY -m src.twin.run_fault_evaluation    # 故障评估
```

> 完整复跑记录见 `docs/battery_reproduce_record.md`（八阶段 REPRODUCE_OK，总 wall time ≈ 28 s）。

### 5.2 确定性保证

| 机制 | 位置 |
|---|---|
| float64 / CPU / 单线程 / 确定性算法 | `src/finetune/_runtime.py`，**每个 torch 模块导入前**执行 |
| 全 RNG 固定（random / numpy / torch） | `RT.seed_everything(0)`，各入口开头 |
| 粒子滤波种子由 trajectory_id 派生 | `classical.stable_seed` |
| GRU 种子由 `(family, horizon)` 派生 | `stable_seed(f"{held}|{h}")` |
| 适配子集按 trajectory_id 排序 | 无 RNG |

### 5.3 v2 只读保证（测试断言，非口头承诺）

| 断言 | 测试 |
|---|---|
| v2 `SHA256SUMS` 全部匹配 | `test_15` |
| v2 `verify` 仍 `VERIFY_OK 7/7` | `test_16` |
| v2 内无神经实现 | `test_13` |
| v3 内确有 `nn.Module` | `test_14` |
| v3 输出路径不落在 v2 内 | `test_17` |

`src/paths.py::assert_v2_untouched` 提供 SHA-256 漂移检测。

### 5.4 Docker

Dockerfile（Python 3.12-slim + torch 2.11，与相控阵同环境；依赖全部走阿里云源避 pythonhosted PEP658 超时）。构建时门禁：`battery_entry verify` 7/7 + `pytest` 58 passed（test_08 已修复，不再 deselect）。

```bash
# 在 WSL2（docker 29.5.3）内，电池组件目录为上下文
cd "XA-202608_最终交付/03_代码/components/battery"
docker build -t xa-battery:latest .
docker run --rm xa-battery:latest verify
```

⚠️ **状态**：Docker 镜像于 2026-08-08 在 WSL2 内首次构建通过（v2 verify 7/7 + reproduce max|diff|=7.4e-16 + pytest 41 passed）。环境统一到 Python 3.12 + torch 2.11 后需重建（当前 conda env 下 58 passed 全绿）。Docker 是赛题 8 分建议项（非一票否决）。Dockerfile 配置已就绪，待最终打包前在干净环境构建补 clean-room 记录。

---

## 6. 结果口径红线（引用任何数字前必读）

**禁止出现的表述：**

- ❌ 把任何 MAE（0.003328 / 198.1 天 / 其他）描述为**真实卫星精度**
- ❌ 称保形区间为"112 天置信带"（它是**逐视界边际**区间，非联合覆盖）
- ❌ 称固定留出集为"从未见过的独立外部验证集"
- ❌ 对任何应力因子做**因果方向断言**（无生成器 ⇒ 无严格反事实，证据上限 `OBSERVATIONAL_EVIDENCE`）
- ❌ 引用 RUL 数字时不标注**仿真域、右删失处理方式、族级留一**
- ❌ 称 GRU 优于交付模型（差距 1.0% < 跨族散布 0.0013，判定 **NOT SEPARABLE**）
- ❌ 称数字孪生生成器可完整复现（代码不在项目内，老化系数未记录）
- ❌ 称 Docker 镜像在当前 Python 3.12 环境下已构建通过（2026-08-08 曾在 Python 3.9 环境构建通过，环境升级后待重建）

**允许且应当写的：**

- ✅ 特征空间迁移在 6/6 族上优于目标域原生基线（F0 = 0.003328 vs F4 = 0.004271，**已验证的预处理效应**）
- ✅ 有限视界交叉检测证据不足（保留原裁决）**且**全任务期 RUL 回归标签非退化、族平均 MAE = 198.1 天（仿真域、右删失单侧感知损失保留、族级留一）——**两句并列，不删任何一句**
- ✅ 迁移强度 lambda_SP 经内层 fold 搜索：52.8% cell 选中 0，47.2% 选中非零值（其中 43.1% 选中中高强度 {1,10}）

详见 `docs/internal/GATES_scope_amendment.md`。
