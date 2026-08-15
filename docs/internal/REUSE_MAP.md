# REUSE_MAP.md — Phase A 历史资产检索结果

**检索日期：** 2026-08-07
**检索范围：** `d:\SUFAcopy\`（历史工作区，只读）+ `D:\SUFAcopy_archive_pre_phase0_20260804_160617\archived_files`（归档区，只读，按确切路径读取）
**检索方式：** 全量文件遍历 + 任务书 10 个关键词族计数 + SHA256 比对定位权威冻结件
**结论：** 8 项能力可复用（直接用/改造），1 项能力确认不存在（经典退化模型），1 项能力物理参数不完整（数字孪生生成器 → PENDING）

> 本文件是"禁止从零重写"规则的执行证据。凡标 `NEW` 的行，都对应下表"不存在"检索结论。

---

## 0. 运行环境（检索副产物，非任务要求但影响全部结论）

| 事实 | 值 |
|---|---|
| CLAUDE.md 记载"当前环境没有 pytest" | **在本次使用的解释器上不成立** |
| 实际使用解释器 | `D:\Anaconda\Anaconda\envs\cv\python.exe`（Python 3.9.25，与 v2 包 `__pycache__` 的 `cpython-39` 标记一致） |
| 该环境依赖 | numpy 2.0.2 / pandas 2.3.3 / scipy 1.13.1 / **torch 2.8.0+cpu** / **pytest 8.4.2** / sklearn 1.6.1 |
| `python` 裸命令 | 不在 PATH（指向 WindowsApps 存根），必须用绝对路径调用 |
| Docker CLI | **缺失**（`docker --version` 失败），故 T7 标 `PENDING_DOCKER_EXECUTION` |

CLAUDE.md 的"13 项直接执行测试脚本"口径保留；本包新增的 pytest 测试在上述 `cv` 环境下运行，不改动任何环境、不安装任何依赖。

---

## 1. 复用表

格式：`能力 | 历史文件绝对路径 | 可复用程度 | 迁入后的新路径 | 备注`

### 1.1 PyTorch 训练/微调循环

```
L2-SP 闭式解（torch float64，含 lambda_source/lambda_bias 双先验项与确定性 fallback 阶梯） | d:\SUFAcopy\src\stk_transfer_v2\models\source_prior_ridge.py | 直接用 | battery_pytorch_v3/src/finetune/source_prior_ridge.py | 【最高价值复用】该文件的目标函数与方案附录 A 的 L2-SP 逐项一致：sum w_i||Z_i W+b-Y_i||^2 + lambda_source||W-W_src||_F^2 + lambda_bias||b-b_src||^2 + alpha||W||_F^2。已含 numpy 独立参考实现 solve_numpy_reference()，可直接作为 test_torch_matches_closed_form.py 的第三方交叉验证。原文件依赖 src.stk_transfer_v2.models._runtime_p3，迁入时需替换为本包自有 runtime 设置。
torch 单线程 float64 CPU 确定性设置（含 NumPy-MKL/torch OpenMP 初始化顺序修正） | d:\SUFAcopy\src\stk_transfer_v2\models\_runtime_p3.py | 直接用 | battery_pytorch_v3/src/finetune/_runtime.py | 注释记录了该顺序是实测得出的，不得重排。
AdamW + 早停 + warmup-cosine + grad clip + 最佳ckpt恢复 的完整训练循环 | d:\SUFAcopy\src\training\stage2_arc_supervised.py | 改造 | battery_pytorch_v3/src/pretrain/train_loop.py | 【训练循环骨架来源】原文件是 PatchTSTSOH 的训练循环（Huber + AdamW + LambdaLR warmup-cosine + patience 早停 + best_state 恢复 + reload 校验）。T3 需要的是 LinearHead 而非 PatchTST，故复用其循环结构与早停/调度逻辑，替换模型与损失。
DataLoader/Dataset 封装、train-only scaler 拟合、target/recover 变换 | d:\SUFAcopy\src\training\arc_data.py | 改造 | battery_pytorch_v3/src/data/arc_dataset.py | ArcSohDataset + fit_scaler + transform + to_target + recover_soh。本包源域预训练直接复用其 "scaler 仅在 train fold 拟合" 的约束。
torch 闭式岭解（lambda_source=0 极限）+ numpy 等价性 | d:\SUFAcopy\battery_release_v2\battery_entry\model.py::solve_ridge_weighted_torch | 直接用 | 不迁入（v2 只读，v3 import 其逻辑的等价重写） | v2 一个字节不许改，故 v3 在 src/finetune/ 内重写同一法方程并由 test_equivalence_with_v2.py 断言逐位一致。
```

### 1.2 深度序列模型（对比臂用，非交付模型）

```
PatchTST 主干（patch embedding + RevIN 开关 + level_skip 旁路 + SOH 头） | d:\SUFAcopy\src\models\patchtst.py | 改造 | battery_pytorch_v3/src/comparators/patchtst_arm.py | 11878 字节，build_backbone() + PatchTSTSOH。作为 T4 深度对比臂候选。注意：不得放入 battery_release_v2（verify::no_neural_implementation 扫描 PACKAGE_ROOT）。
PatchTST-Delta 变体（delta 目标空间） | D:\SUFAcopy_archive_pre_phase0_20260804_160617\archived_files\src\legacy_v1_1\patchtst_delta.py | 仅参考 | — | 5033 字节。delta 目标的处理方式可参考，但 v2 的 T0_DELTA_REFERENCE 语义已由 L3 表固化。
PatchTST 归一化变体 | D:\...\archived_files\src\legacy_v1_2\patchtst_norm.py | 仅参考 | — | 6304 字节。
Phase 6 深度模型门禁裁决原文（PATCHTST_SKIPPED_BY_PREDEFINED_GATE） | d:\SUFAcopy\reports\stk_transfer_v2\06_patchtst\phase6_decision.md | 直接用（原文引用） | battery_pytorch_v3/GATES_scope_amendment.md 内原文保留 | 裁决保留原文，只新增作用域说明。
```

### 1.3 源域公开数据集 / 预训练

```
NASA PCoE 公开电池退化数据集，34 个电芯已清洗为 npz | d:\SUFAcopy\data\processed\clean\arc\B0005..B0056.npz (+ .meta.json) | 直接用 | battery_pytorch_v3/data/source_public/（软引用，不复制 34MB） | 【G6 关键资产：公开数据集真实存在】每个 npz 含 cycle_idx / ambient_temperature / capacity / voltage / current / temperature / current_load / voltage_load / time，逐循环放电曲线为 object 数组。B0005 有 168 个循环。这是 ARC 空间的真实来源，源域预训练可用真实数据而非占位。
NASA .mat 原始加载器（放电段提取、容量序列构建） | d:\SUFAcopy\src\data\mat_loader.py | 直接用 | battery_pytorch_v3/src/data/mat_loader.py | 7372 字节。
公开数据集清洗（异常循环剔除、容量单调性处理） | d:\SUFAcopy\src\data\clean.py | 直接用 | battery_pytorch_v3/src/data/clean.py | 4258 字节。
20 点参考窗构建 / 滑窗 | d:\SUFAcopy\src\data\windowing.py | 直接用 | battery_pytorch_v3/src/data/windowing.py | 5595 字节，配 tests/test_windowing.py。
SOH 标签构建（容量→SOH 归一化，EOL 定义） | d:\SUFAcopy\src\data\labels.py | 直接用 | battery_pytorch_v3/src/data/labels.py | 5213 字节。
权威 11 维源模型（coef (10,11) + intercept(10) + clip_lo/hi + scaler_mean/std + alpha=10 + L=20 + H=10 + EOL=0.7） | d:\SUFAcopy\artifacts\experimental_arc_sensitivity\arc_clean_fixed_alpha10\model.npz | 直接用 | battery_pytorch_v3/data/source_prior/beta_src.npz（派生副本） | 【beta_src 唯一权威来源，已由 SHA256 确证】该文件 sha256=fcce4e53...f2a979，与 v2 冻结件 models/arc_clean_fixed_space.npz::model_npz_sha256 记录值逐字符相同 → 这就是产出 v2 冻结 ARC 空间的那个源模型。v2 的 npz 刻意剔除了源系数（note 字段明示"deliberately absent"以防误用），故 L2-SP 所需的 beta_src 必须从此处取。
源域预测提取（把源模型套到目标域 L3 上产生 source_pred） | d:\SUFAcopy\src\stk_transfer\extract_source_predictions.py | 改造 | battery_pytorch_v3/src/pretrain/extract_source_predictions.py | 12914 字节。
Phase 3 源先验完整引擎（lambda_source 网格 + 内层选择 + 保留比统计） | d:\SUFAcopy\src\stk_transfer_v2\source_prior\engine_p3.py + run_phase3_source_prior.py + contracts_p3.py | 改造 | battery_pytorch_v3/src/finetune/l2sp_engine.py | 17350 + 52191 + 17799 字节。lambda_source 网格搜索与"44/72 cell 选中 0"的结论即出自此处，T3 的 lambda_SP 谱系搜索直接沿用其网格与内层选择协议。
Phase 3 逐折 L2-SP 拟合结果（fold0..5 × 10/25/50/100pct） | d:\SUFAcopy\artifacts\stk_transfer_v2\03_source_prior\arc_clean_fixed_source_prior\lofo_{10,25,50,100}pct\fold{0..5}_model.npz | 直接用（作为回归基准） | battery_pytorch_v3/results/ 内引用 | 可用于交叉核对 v3 的 L2-SP 复现是否与冻结结果一致。
```

### 1.4 ARC 空间与源系数

```
冻结 ARC clip/scaler（11 维，clip_lo/clip_hi/scaler_mean/scaler_std + 4 个 sha256 + context_length=20） | d:\SUFAcopy\battery_release_v2\models\arc_clean_fixed_space.npz | 直接用（只读 import） | v3 直接读 v2 路径，不复制 | 【冻结坐标系，applied never refitted】v3 的 test_equivalence_with_v2.py 用它断言逐位一致。
冻结 11 维特征提取器（权威实现） | d:\SUFAcopy\battery_release_v2\battery_entry\features.py | 直接用 | battery_pytorch_v3/src/features/frozen11.py（逐字节复制 + 等价性测试） | FEATURE_NAMES 顺序 load-bearing；feature_names_sha256() 可比对。同一实现另有两个副本：src/production_model/features.py（原始权威）与 src/stk_transfer_v2/data/frozen_features_v2.py。
FrozenSpace.apply / 加权岭闭式解 / 轨迹等权+族平衡权重 | d:\SUFAcopy\battery_release_v2\battery_entry\model.py | 直接用（等价重写） | battery_pytorch_v3/src/features/space.py + src/finetune/weights.py | trajectory_equal_weights() 是 T1 冻结协议要求的权重方案，WEIGHTING_RULE 字符串记录了规则。
frozen_alpha（LOFO 18 cell + fixed_holdout 3 视界的已选 alpha，含 selector_saw_outer_test=false 证据） | d:\SUFAcopy\battery_release_v2\models\frozen_alpha.json | 直接用 | v3 读 v2 路径 | alpha 网格锁定在 configs/stk_transfer.yaml::p3.alpha_grid（7 候选），不得加宽。
Phase 2B 加权岭引擎（_solve_ridge_weighted 原实现 + 权重归一到 mean 1.0） | d:\SUFAcopy\src\stk_transfer_v2\baselines\engine_p2b.py | 直接用 | battery_pytorch_v3/src/finetune/ridge_core.py | 27522 字节。v2 的 model.py 明示"mirrors engine_p2b._solve_ridge_weighted exactly"。
```

### 1.5 数字孪生生成器

```
【生成器代码本体】STK 导出 → 环境时序 → 退化积分 → L0/L1/L2/L3 的可执行实现 | 检索结论：不存在 | 不存在 | battery_pytorch_v3/src/twin/SPEC.md（PENDING） | 【PENDING】全量检索 d:\SUFAcopy 与归档区所有 .py/.md，正则匹配 (L1_multi_scenario|L0_multi_scenario|rint_ohm_true|q_max_ah_true|def .*degrad|def .*aging|arrhenius|calendar_aging|cycle_aging|to_csv.*L[01])，命中的全部是 read-only 消费者（DATA_CARD.md / STK_SCENARIOS.md / labels_v2.py / contracts_p2a.py / golden_time_p4.py / _p7_build_package.py）与审计脚本，无任何写 L0/L1 的生成代码。与方案 G3「生成器代码不在项目内」一致。
【生成器产出物】L0 环境时序（27 列，含 eclipse_state/solar_power_w/beta_angle/thermal_boundary_c/mission_mode/load_power_w） | d:\SUFAcopy\data\afterstk\L0_multi_scenario_stk_environment.csv (80.8 MB) | 直接用（只读） | 不复制，v3 按路径读 | 生成器缺失但其输出完整存在，故 L0→L1→L2→L3 链路可被"复现校验"（verify 已有输出的一致性），但不可被"重新生成"。
【生成器产出物】L1 电池遥测（29 列，含 voltage_v / current_a / temperature_c 及其 _true 对偶、soc、dod_orbit、efc、ah/wh_throughput、q_max_ah_true、soh_true、rint_ohm_true、sensor_quality_flags） | d:\SUFAcopy\data\afterstk\L1_multi_scenario_battery_telemetry_72h.csv (103.9 MB) | 直接用（只读） | 不复制，v3 按路径读 | 【G1 关键资产：V/I/T 遥测真实存在】方案 G1 称"报告里看不到 V/I/T"——实际 L1 层有真实的 voltage_v / current_a / temperature_c 秒级遥测（30 s 步长，72 h 模板），且带观测/真值对偶通道。T6 的可观测量链路图因此有实测数据支撑，不是纸面声明。
【生成器产出物】L2 参考健康点（34 列，含 capacity_ah/rint_ohm/mean_temp_c/max_temp_c/mean_c_rate/dod/efc/eclipse_h_per_day 等环境应力通道） | d:\SUFAcopy\data\afterstk\L2_multi_scenario_reference_points.csv (7.3 MB) | 直接用（只读） | 不复制 | v2 包内的 L2_reference_points_v2_min.csv 只保留 5 列（trajectory_id/reference_index/reference_day/soh_observed/soh_true），环境通道在此原始 L2 内。
【生成器产出物】STK 原始导出（24 场景 × Beta角/光照时段/太阳翼面积/太阳翼功率 + Battery_ExternalData） | d:\SUFAcopy\data\stk_report\T001..T024\*.csv + d:\SUFAcopy\data\afterstk\stk_external_data\T001..T024_*_Battery_ExternalData.csv | 直接用（只读） | 不复制 | 每场景约 1.23 MB ExternalData。注意：`.sc` 场景文件确实 0 个（与 G3 一致），存在的是导出的 CSV 报告。
物理参数表（28 V 母线 / 100 Ah 初始容量 / SOC 0.35–0.95 / 目标单轨 DoD 20–35% / EOL 0.70 / 预警 0.80 / 地影功率反求公式 / 单步 10 步积分流程 / 域随机化清单） | D:\SUFAcopy_archive_pre_phase0_20260804_160617\archived_files\电池这块的技术方案.md §7 | 改造（作为 SPEC 依据） | battery_pytorch_v3/src/twin/SPEC.md | 【部分可用】§7.1 给出名义电池参数表与地影功率反求公式，§7.2 给出每时刻 10 步积分流程（功率平衡→ECM→热→日历+循环老化→Qmax/Rint 反馈→safe mode 触发→每轨聚合→7-30 天参考点），§7.3 给出域随机化清单。**但缺少老化方程的具体系数**（日历老化的 Arrhenius 前因子/活化能、循环老化的 DoD 指数与 C-rate 系数、Rint 增长率），因此无法逐位复现已有的 soh_true 轨迹。按任务书要求：不编造，写 SPEC.md + 标 PENDING。
故障注入实现（内阻阶跃/容量跳变/热控偏置/深放电） | 检索结论：不存在 | 不存在 | battery_pytorch_v3/src/twin/fault_injection.py（NEW） | 【NEW】fault_inject 关键词族命中的全部是审计产物（buckets.json/protected_*.json/freeze_before.json）与 arc_anomaly_sensitivity（那是 ARC 空间的异常敏感性分析，不是故障注入）。v2 实测 safe_mode_points_total = 0，温度仅 19.4–24.5 °C，DoD 0.09–0.27，确认无故障注入。故障注入作用在**已有的 L2 参考点轨迹**上（后处理式注入），不依赖缺失的老化系数，因此可实现且可跑。
ARC 空间异常敏感性（clip 边界扰动下的稳健性，含 47201 字节主实现） | d:\SUFAcopy\src\stk_transfer\run_arc_anomaly_sensitivity.py + d:\SUFAcopy\data\arc_anomaly_sensitivity\ | 仅参考 | — | 是 clip/scaler 敏感性分析，与故障注入不同源，但其扰动施加方式可参考。
```

### 1.6 故障注入

见 1.5 末两行：**实现不存在（NEW）**，但注入靶点（L2 轨迹）与观测链（L1 V/I/T）都在。

### 1.7 经典退化模型

```
Wiener 过程 / 粒子滤波 / 卡尔曼 / 双指数容量衰减 + PF-RUL | 检索结论：不存在 | 不存在 | battery_pytorch_v3/src/comparators/classical.py（NEW） | 【NEW，检索最干净的一条】关键词族 (wiener|particle_filter|particle filter|kalman|exponential_fit|double_exp) 在 d:\SUFAcopy 与归档区全部 .py/.md/.json/.yaml 中，达到 >=3 次命中的文件**只有 1 个，且就是任务书自己的 battery_branch_closeout_plan.md**。逐个关键词单独检索亦无任何实现。与方案 G10「无现有寿命预测或退化建模方法」一致。故 T4 的经典臂必须新写。
非学习基线：observable_local_trend（斜率外推）+ persistence | d:\SUFAcopy\battery_release_v2\battery_entry\model.py::observable_local_trend / persistence | 直接用 | battery_pytorch_v3/src/comparators/baselines.py | Phase 2B 最低跨族 std 的高稳定性基线。
目标域原生 Ridge（无迁移消融臂） | d:\SUFAcopy\src\stk_transfer_v2\baselines\baselines_p2b.py（target_only_*_space） | 直接用 | battery_pytorch_v3/src/comparators/target_only_ridge.py | 14606 字节。即 v2 的交付模型 target_only_arc_space，作为 T4 第四臂。
扩展基线集合 | d:\SUFAcopy\src\stk_transfer\run_extended_baselines.py | 仅参考 | — | 8505 字节。
legacy 基线集合 | D:\...\archived_files\src\legacy_v1_1\baselines.py | 仅参考 | — | 4753 字节。
```

### 1.8 保形与 RUL

```
Phase 5 RUL 实现（有限视界穿越检测 + 删失处理 + rul_lower_bound） | d:\SUFAcopy\src\stk_transfer_v2\rul\rul_p5.py | 改造 | battery_pytorch_v3/src/rul/censoring.py | 19017 字节。【作用域注意】该实现是 112 天有限视界穿越检测器（RUL_EVIDENCE_INSUFFICIENT 的作用域），T1 要做的是全任务期 RUL 回归——两个不同任务。复用其 censored / rul_lower_bound_days 的删失记账方式，不复用其标签定义。
RUL 标签与删失记账（rul_and_censoring：EOL 穿越索引、右删失下界） | d:\SUFAcopy\src\stk_transfer_v2\data\labels_v2.py | 直接用 | battery_pytorch_v3/src/rul/labels.py | 6132 字节。build_l3_v2.py 调用 LB.rul_and_censoring(tru, day, tid, ce) 产出 target_rul_days / censored / rul_lower_bound_days。
v2 包内 RUL 模块 | d:\SUFAcopy\battery_release_v2\battery_entry\rul.py | 仅参考 | — | 只读；其 warning/0.80 逻辑对 T2 有参考价值。
保形区间（族平衡 cross-fit，逐视界边际，含 gate 判定与 shortfall 标记） | d:\SUFAcopy\src\stk_transfer_v2\uncertainty\conformal_p5.py + contracts_p5.py + report_p5.py | 改造 | battery_pytorch_v3/src/rul/conformal.py | 27466 + 48923 + 31757 字节。T1 的 RUL 保形区间复用其 FAMILY_BALANCED_CROSSFIT 方法与"逐视界边际、不声称联合覆盖"的口径。
冻结保形半宽（12 个 PROTOCOL×MODEL×level×horizon 组合） | d:\SUFAcopy\battery_release_v2\models\conformal_quantiles.json | 直接用（只读） | — | 含 gate_status=CONFORMAL_NOT_VALIDATED 与 per_horizon_marginal_only=true。红线：不得称其为"112 天置信带"。
Group cross-fitted conformal（早期实现） | D:\...\archived_files\src\final_core_v1_group_conformal\gcconformal.py | 仅参考 | — | 13961 字节。
conformal 早期实现 | D:\...\archived_files\src\final_core_v1_conformal\conformal.py | 仅参考 | — | 10547 字节。
早期 RUL 实现 | D:\...\archived_files\src\final_core_v1\rul.py | 仅参考 | — | 11079 字节。
全任务期 RUL 可行性探针（8021 锚点 / 标签非退化 / 族平均 MAE 174.4 天） | d:\SUFAcopy\files\rul_longhorizon_feasibility.py | 改造 | battery_pytorch_v3/src/rul/run_mission_span.py | 【T1 起点】6722 字节。已含 build_anchors / fit_ridge / slope_extrapolation。缺：右删失感知损失、冻结 ARC 空间、轨迹等权+族平衡权重、保形区间。
```

### 1.9 适配曲线

```
adaptation_subsets 定义（10pct=4 / 25pct=10 / 50pct=20 / 100pct=40 条目标轨迹） | d:\SUFAcopy\battery_release_v2\docs\stk_metadata\generation_config.json | 直接用 | v3 读 v2 路径 | 方案 G12 称"已定义未使用"——实际 Phase 3/4 已按 lofo_{10,25,50,100}pct 跑过（见 artifacts/stk_transfer_v2/03_source_prior/.../lofo_*pct/），故适配曲线是"已有结果重新组织"而非从零跑。
适配比例诊断 | d:\SUFAcopy\src\stk_transfer\diagnose_adaptation_fraction.py | 改造 | battery_pytorch_v3/src/finetune/adaptation_curve.py | 8332 字节。
L3 内的 adapt 标记列（adapt_rank_within_target_train / in_adapt_{10,25,50,100}pct） | d:\SUFAcopy\src\stk_transfer_v2\data\build_l3_v2.py | 直接用 | battery_pytorch_v3/src/data/adapt_subsets.py | 子集成员关系已固化在 L3 表内，v3 不重新抽样（否则破坏可比性）。
Phase 4 逐 budget 结果（budget_label=lofo_{10,25,50,100}pct × feature_group × target_semantics 的 family_macro_mae/worst_family_mae） | d:\SUFAcopy\reports\stk_transfer_v2\04_exposure_stress\feature_ablation_results.csv | 直接用 | battery_pytorch_v3/results/adaptation_curve.csv 的交叉核对源 | 40 行 × 15 列，已含 budget_pct 列。
```

### 1.10 提前性

```
预警提前期 Δt_warn / Prognostic Horizon (Saxena 2008, alpha=0.2) | 检索结论：不存在 | 不存在 | battery_pytorch_v3/src/metrics/prognostic.py（NEW） | 【NEW】lead_time 关键词族命中的全部是 warning_threshold=0.80 的阈值定义与 RUL gate 判定（rul_p5.py / battery_entry/rul.py / report_p5.py / gate_decision.json），**无 PH 或预警提前期的指标实现**。与方案 G9「提前性指标完全缺失」一致。所需输入（110/120 轨迹跨 0.80、92 条到 EOL）已在 L2 内，故可实现。
warning_threshold=0.80 / eol_threshold=0.70 的权威定义 | d:\SUFAcopy\battery_release_v2\docs\stk_metadata\generation_config.json | 直接用 | — | mission_days=2190, reference_step_days=14（Δt_warn 的分辨率下界）。
EOL 穿越与预警点统计 | d:\SUFAcopy\battery_release_v2\data\split_manifest_v2.csv | 直接用 | — | 含 eol_crossed_on_soh_true / right_censored_on_soh_true / manifest_eol_reached 逐轨迹标记，可直接得 92 到 EOL / 28 右删失。
EOL 告警诊断（误报/漏报计数） | d:\SUFAcopy\src\eval\diagnostics.py::eol_alarm | 改造 | battery_pytorch_v3/src/metrics/alarm.py | 3415 字节。已有 false_alarm / missed_alarm 计数逻辑，扩展为 Δt_warn 分布。
```

### 1.11 可观测量链路（T6）

```
29 个应力字段观测性契约（25 admitted / 4 excluded：3 family_proxy + 1 duplicate_collinear；含 signal_to_noise>=1.0 admission 规则与 declared_before_any_fit=true） | d:\SUFAcopy\reports\stk_transfer_v2\04_exposure_stress\stress_observability_contract.json | 直接用（原文复述） | battery_pytorch_v3/docs/observables_chain.md | 【T6 关键资产：29 字段结论已审计】任务书要求"复述 29 个应力字段结论"——contract 内 n_fields_declared=29 / n_fields_admitted=25 / n_fields_excluded=4 逐字段可查，另有 absorbed_by_geometry11_audit 对 25 个 obs_hist_* 列逐列记录被 11 维几何吸收的程度。obs_hist_ 前缀的历史窗口聚合规则（mean/max/last/cum，窗口严格 <= context_end）亦在其中。
应力特征构建实现（obs_hist_* 派生，含 raw stress_ 前缀被 BARRED 的理由） | d:\SUFAcopy\src\stk_transfer_v2\exposure_stress\stress_features_p4.py | 直接用 | battery_pytorch_v3/src/features/stress_hist.py | 25854 字节。
应力消融结果（40 行：feature_group × target_semantics × budget，含 family_macro_mae/worst_family_mae/across_family_std） | d:\SUFAcopy\reports\stk_transfer_v2\04_exposure_stress\feature_ablation_results.csv | 直接用 | battery_pytorch_v3/results/ 内引用 | B0_OBSERVABLE_LOCAL_TREND vs F0_TARGET_ARC_SPACE_GEOMETRY11 vs 含应力组 的逐组对比。
应力可识别性 / 反事实设计 / 显著性 | d:\SUFAcopy\reports\stk_transfer_v2\04_exposure_stress\stress_identifiability.csv + counterfactual_design.json + counterfactual_results.csv | 直接用 | — | 红线：不得对应力因子做因果方向断言（证据上限 OBSERVATIONAL_EVIDENCE）。
应力诊断三件套 | d:\SUFAcopy\src\stk_transfer\diagnose_stress{,_mechanism,_pruned,_significance}.py | 仅参考 | — | 9508 + 3521 + 5226 + 9702 字节。
L1 V/I/T → SOH 估计链的数据支撑 | d:\SUFAcopy\data\afterstk\L1_multi_scenario_battery_telemetry_72h.csv | 直接用（只读） | docs/observables_chain.md 的实测依据 | 见 1.5：voltage_v/current_a/temperature_c + soc/dod_orbit/efc/ah_throughput 齐备，库仑计数与内阻脉冲两条路径均有对应列（ah_throughput / rint_ohm_true）。
星上 SOH 估计链的文字描述（§8 真实遥测如何进入方案） | D:\...\archived_files\电池这块的技术方案.md §8 | 改造 | docs/observables_chain.md | 含"不得将异常标签等价成失效寿命标签"等红线，与 LIMITATIONS.md 一致。
```

### 1.12 Docker 与文档（T7）

```
Dockerfile（v2 基础版，torch 注释掉） | d:\SUFAcopy\battery_release_v2\Dockerfile + requirements.lock + .dockerignore | 改造 | battery_pytorch_v3/docker/Dockerfile + requirements.lock | 需解注释 torch CPU wheel。v2 原件一字节不动，v3 用副本改。
复现说明 | d:\SUFAcopy\battery_release_v2\REPRODUCIBILITY.md | 改造 | battery_pytorch_v3/RUN.md | 补训练/微调两节。
门禁裁决原文（frozen_verdicts，verify 第 6 组扫描对象） | d:\SUFAcopy\battery_release_v2\docs\GATES.md | 直接用（原文保留） | battery_pytorch_v3/GATES_scope_amendment.md 内引用 | 裁决一字不改，只新增作用域说明。
声明清单 / 限制清单 | d:\SUFAcopy\battery_release_v2\CLAIMS_MANIFEST.json + LIMITATIONS.md | 直接用（继承红线） | v3 各 results/ 输出的限定语来源 | 报告口径红线的权威出处。
```

---

## 2. NEW 清单（检索确认不存在，允许新写）

| # | 能力 | 检索证据 | 新路径 |
|---|---|---|---|
| N1 | Wiener 过程 / 双指数 + 粒子滤波 RUL | `(wiener\|particle_filter\|kalman\|exponential_fit\|double_exp)` 全库 >=3 命中文件数 = **1**，且该文件是任务书本身 | `src/comparators/classical.py` |
| N2 | 预警提前期 Δt_warn + Prognostic Horizon(α=0.2) | `lead_time` 族命中全是 0.80 阈值定义与 gate 判定，无指标实现 | `src/metrics/prognostic.py` |
| N3 | 故障注入四类（内阻阶跃/容量跳变/热控偏置/深放电） | `fault_inject` 族命中全是审计产物；v2 实测 safe_mode_points_total=0 | `src/twin/fault_injection.py` |
| N4 | 右删失感知 RUL 损失（单侧 hinge） | 现有 rul_p5 只做删失**记账**（censored 标记 + 下界），无删失感知**损失** | `src/rul/censored_loss.py` |
| N5 | LinearHead(nn.Module) + AdamW 梯度训练与闭式解等价性测试 | 现有 torch 路径全是 `linalg.solve` 闭式解；PatchTST 有梯度训练但不是线性头 | `src/pretrain/linear_head.py`, `tests/test_torch_matches_closed_form.py` |

---

## 3. PENDING 清单

| # | 条目 | 缺什么 | 处置 |
|---|---|---|---|
| P1 | 数字孪生生成器可复跑实现 | 老化方程系数：日历老化 Arrhenius 前因子/活化能、循环老化 DoD 指数与 C-rate 系数、Rint 增长率。归档 §7 只有名义电池参数与流程，无系数 | 已写 `src/twin/SPEC.md`（生成方法 + 参数表 + 伪代码），标 `PENDING_AGING_COEFFICIENTS`，**未编造系数**。**追加实测证据**：归档 §7.1 记标称容量 100 Ah，而 L1 实测 `q_max_ah_true` 起始为 **40.0 Ah** —— 归档是"演示值"，与实际生成参数不符，直接证明不能照抄归档重建。已实现两个**不依赖缺失系数**的替代模块：`verify_chain.py`（L1→L2 聚合前向校验，已跑通）与 `fault_injection.py`（四类故障后处理注入，已跑通） |
| P2 | Docker build/run 实跑 | 本机 Docker CLI 与 Docker Desktop **均未安装**（`docker` 不在 PATH；`C:\Program Files\Docker` 等 4 个路径均不存在；`com.docker.service` 未注册）。WSL2 有 Ubuntu-22.04 但未注册 Docker 引擎 | 保留静态 `docker/Dockerfile`（torch 已解注释）+ `docker/requirements.lock`（`torch==2.8.0` 为硬依赖），标 `PENDING_DOCKER_EXECUTION`。`docker/build_run.log` 记录的是**确认引擎缺失的 5 项探测过程，不是伪造的构建输出** |
| P3 | `.sc` STK 场景文件 | 项目内 0 个（与 G3 一致）；存在的是 24 场景的 CSV 导出报告 | 已在 `SPEC.md` §0 与 `docs/observables_chain.md` §5 说明证据链止于 CSV 导出层 |

---

## 5. 最终交付状态（自检后回填，2026-08-07）

### 5.1 硬验收项

| 项 | 结果 |
|---|---|
| `battery_release_v2` `python -m battery_entry verify` | ✅ **VERIFY_OK 7/7**（全部 v3 工作完成后复跑仍为 7/7） |
| `SHA256SUMS` 全量匹配 | ✅ 由 `test_15` 断言 |
| v2 内无 `nn.Module` | ✅ 由 `test_13` 断言 |
| v2 目录内未新增任何文件 | ✅ 已核查（v2 自身的 `guard_output_write` 拒绝了包内写入，reproduce 输出改至 v3 目录） |
| `pytest tests/ -v` | ✅ **58 passed** |
| 11 维特征与 ARC 空间与 v2 一致 | ✅ `test_01`–`test_08`（含随机历史逐位相等） |
| 梯度解 vs 闭式解 | ✅ 良态问题 < 1e-6（`test_04`，7 个 λ_SP 值）；真实 L3 设计矩阵见 `test_07` |

### 5.2 已跑通的入口

| 命令 | 状态 |
|---|---|
| `python -m src.rul.run_mission_span` | ✅ exit 0 |
| `python -m src.metrics.run_prognostic` | ✅ exit 0 |
| `python -m src.finetune.run_l2sp` | ✅ exit 0 |
| `python -m src.comparators.run_all` | ✅ exit 0 |
| `python -m src.pretrain.run_pretrain` | ✅ exit 0 |
| `python -m src.twin.verify_chain` | ✅ exit 0 |
| `python -m src.twin.fault_injection` | ✅ exit 0 |
| `python -m src.twin.run_fault_evaluation` | ✅ exit 0 |

### 5.3 复用比例（按文件计）

| 类别 | 数量 | 说明 |
|---|---|---|
| 直接复用（逐字节复制或只读引用） | 12 项 | 冻结 11 维特征提取器（SHA-256 与 v2 逐位相同）、冻结 ARC 空间、frozen_alpha、conformal_quantiles、L2/L3/split 数据、NASA PCoE 34 电芯、β_src 源模型、generation_config、29 应力字段契约、应力消融结果 |
| 改造复用 | 9 项 | L2-SP 闭式解（`source_prior_ridge.py`）、torch 确定性 runtime（`_runtime_p3.py`）、AdamW 训练循环骨架（`stage2_arc_supervised.py`）、加权岭引擎（`engine_p2b.py`）、RUL 删失记账（`labels_v2.py` / `rul_p5.py`）、保形方法（`conformal_p5.py`）、适配诊断、EOL 告警诊断、RUL 探针（`rul_longhorizon_feasibility.py`） |
| **新写（NEW）** | **5 项** | 见 §2：N1 经典退化模型、N2 提前性指标、N3 故障注入、N4 删失感知损失、N5 LinearHead + 等价性测试 |
| PENDING | 3 项 | 见 §3 |

**每一项 NEW 都对应 §2 表中的检索证据。** 未出现"检索到但仍重写"的情况。

---

## 4. 边界遵守声明

- `battery_release_v2/` 全程**只读**：v3 通过绝对路径 import/读取，从不写入。任务结束时复跑 `python -m battery_entry verify` 必须仍为 `VERIFY_OK 7/7`。基线已实测确认为 7/7。
- 所有 `nn.Module` 一律位于 `battery_pytorch_v3/`，不进入 `PACKAGE_ROOT`，避免 `verify::no_neural_implementation` 由 7/7 掉到 6/7。
- 归档区与 `SUFAcopy/` 历史树**只读**：需要的实现按确切路径复制到 v3 再改，不原地修改；`archive_manifest.json` 未整体载入，只做脚本化查询。
- 冻结裁决（`SOURCE_PRIOR_NOT_VALIDATED` / `PATCHTST_SKIPPED_BY_PREDEFINED_GATE` / `RUL_EVIDENCE_INSUFFICIENT` / `CONFORMAL_NOT_VALIDATED` / `STRESS_NOT_VALIDATED`）**原文保留**，仅在 `GATES_scope_amendment.md` 新增作用域说明。
