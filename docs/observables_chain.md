# 可观测量链路：V/I/T 遥测 → 星上 SOH 估计 → 本模型输入

**日期：** 2026-08-07
**对应缺口：** 方案 G1（可观测量与赛题不匹配）、G4（退化机理假设未成节）
**状态：** 已实测支撑，非纸面声明

---

## 0. 结论先行

赛题点名「基于电压、电流、温度等遥测量」。方案 G1 判断「11 维特征全部由 `soh_observed` 单通道导出，报告里看不到 V/I/T」。

**实际情况：V/I/T 遥测真实存在于 L1 层，本次已实测。** 本工作定位在链路**后半段**（SOH → 多视界预测 / RUL / 预警）；前半段（V/I/T → SOH）由数字孪生的 ECM + 热模型 + 库仑计数实现，其输出完整保存在 L1。

实测包络（`src/twin/verify_chain.py` 从 103.9 MB L1 流式聚合：24 场景 / 207,384 行 / 30 s 步长 / 72 h 模板）：

| 遥测量 | L1 列名 | 实测范围 |
|---|---|---|
| 电池端电压 | `voltage_v` | 28.546 – 33.262 V |
| 充放电电流 | `current_a` | 幅值至 24.919 A（充正放负） |
| 电池温度 | `temperature_c` | 均值 21.73，最高 24.801 °C |
| 荷电状态 | `soc` | 0.653 – 0.950 |
| 单轨放电深度 | `dod_orbit` | 均值 0.2199，最大 0.2970 |
| 等效完整循环 | `efc` | 累积通道 |
| 安时 / 瓦时吞吐 | `ah_throughput` / `wh_throughput` | 累积通道 |

L1 另带**观测/真值对偶通道**（`voltage_v` vs `voltage_v_true`、`current_a` vs `current_a_true`、`temperature_c` vs `temperature_c_true`），即传感器噪声在生成时被显式建模，观测链不是理想化的。

---

## 1. 链路图

```
┌─── STK（Ansys Systems Tool Kit）───────────────────────────────┐
│  24 场景 × 6 环境族（h{500,550} × i{070,053} × raan{000,090,180}）│
│  导出：光照/地影时段、Beta 角、太阳翼面积、太阳翼功率、任务模式    │
│  产物：data/stk_report/T001..T024/, stk_external_data/          │
└───────────────────────────┬────────────────────────────────────┘
                            ↓  L0（27 列，80.8 MB）
        eclipse_state · solar_power_w · beta_angle_deg
        panel_effective_area_m2 · load_power_w · thermal_boundary_c · mission_mode
                            ↓
┌─── 电池数字孪生（生成器代码不在项目内，见 src/twin/SPEC.md）─────┐
│  ① 功率平衡与 BMS    P_batt = η_reg·P_sa − P_load               │
│  ② ECM              V = OCV(SOC) − I·R_int                     │
│  ③ 热模型            dT/dt = (I²R_int − h(T−T_bnd)) / C_th      │
│  ④ 日历 + 循环老化   ← 系数未记录 ⇒ PENDING                     │
│  ⑤ Q_max / SOH / R_int 反馈至 ①                                │
│  ⑥ 保护触发（欠压 / 过温 / SOC / safe mode）                    │
└───────────────────────────┬────────────────────────────────────┘
                            ↓  L1（29 列，103.9 MB）★ V/I/T 在此
   voltage_v · current_a · temperature_c（+ _true 对偶）
   soc · charge_state · dod_orbit · ah_throughput · wh_throughput · efc
   q_max_ah_true · soh_true · rint_ohm_true · sensor_quality_flags
                            ↓
┌─── 星上 SOH 估计器（标准实现，三条独立证据融合）──────────────────┐
│  (a) 库仑计数    SOH_Q = ∮I dt / Q_nominal，由 ah_throughput 支撑 │
│  (b) 电压平台    放电曲线特征点 → 容量映射，由 voltage_v 支撑      │
│  (c) 内阻脉冲    ΔV/ΔI 阶跃辨识 → R_int 增长，由 rint_ohm 支撑     │
│  三者互校，输出带噪声的 soh_observed                             │
└───────────────────────────┬────────────────────────────────────┘
                            ↓  L2（34 列）每 14 天一个参考健康点
   soh_observed（模型输入）· soh_true（仅评估）
   capacity_ah · rint_ohm · mean_temp_c · max_temp_c · mean_c_rate · dod
   efc · eclipse_h_per_day · mission_comm_ratio · mean_beta_deg ...
                            ↓
┌─── ★ 本工作范围从这里开始 ★ ────────────────────────────────────┐
│  20 点参考窗（CONTEXT_LENGTH = 20）                             │
│         ↓                                                      │
│  冻结 11 维几何特征（features.py::common_features）              │
│    last, diff1, diff2, diff_long, slope5, slope_full,          │
│    curvature, ctx_mean, ctx_std, ctx_min, ctx_max              │
│         ↓                                                      │
│  冻结 ARC 空间 clip/scaler（源域拟合，应用而不重拟合）            │
│         ↓                                                      │
│  L2-SP 微调头（λ_SP 谱系搜索）                                   │
│         ↓                                                      │
│  多视界 ΔSOH（h = 2/4/8 步 = 28/56/112 天）                     │
│    + 全任务期 RUL（删失感知）                                    │
│    + 保形区间（逐视界边际）                                      │
│    + 0.80 阈值预警（实测族平均提前 124.1 天）                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 为什么模型只吃 SOH 单通道——这是**设计选择**，且已被实测检验

11 维特征只读 `soh_observed` 历史，不读 V/I/T、不读环境应力、不读族标识。理由有二：

1. **预测时可得性。** 提取器不消费任何未来点、任何隐藏真值通道、任何环境列，因此该 11 维空间在部署时可安全计算。
2. **已实测：加入应力通道并未改善。** 见 §3。

这不是「没做」，而是「做过并测量了」。

---

## 3. 29 个应力字段的已审计结论（复述 Phase 4，原文口径不变）

来源：`reports/stk_transfer_v2/04_exposure_stress/stress_observability_contract.json`（`declared_before_any_fit = true`）

### 3.1 声明与准入

| 项 | 数量 |
|---|---|
| 声明字段 `n_fields_declared` | **29** |
| 准入 `n_fields_admitted` | **25** |
| 排除 `n_fields_excluded` | **4** |

排除明细（4 个）：

| 排除类别 | 字段 | 理由 |
|---|---|---|
| `excluded_family_proxy`（3） | `obs_hist_const_altitude_km`、`obs_hist_const_inclination_deg`、`obs_hist_const_raan_deg` | 在族内取常值，等价于泄漏 `environment_family_id`——而族是严格隔离键 |
| `excluded_duplicate_collinear`（1） | `obs_hist_cum_wh_throughput` | 与 `obs_hist_cum_ah_throughput` 相关 \|r\| ≥ 0.9999，同一个数两次 |
| `excluded_not_identifiable`（0） | — | 无 |

准入规则（**拟合前声明**）：`min_signal_to_noise = 1.0`（跨族有效变异必须超过测量噪声地板）、`near_constant_std = 1e-8`、`max_distinct_for_family_proxy = 12`、`max_abs_correlation = 0.9999`。

### 3.2 原始 `stress_*` 前缀被 BARRED 的理由（重要，避免误读）

实测：每个原始 `stress_*` 列是其 L2 对应量在 `context_end` 处的**点值**（最大绝对误差 0 至 5.8e-11），**不是历史窗口摘要**。

- 无未来泄漏：点值切片吻合，严格未来切片不吻合 → 原始列不泄漏未来。
- 但仍被禁用：赛题要求的是**历史摘要**（历史平均/最高温度、历史 DoD、累积 EFC）。截止点单点值是**另一个量**，会悄悄回答另一个问题。
- 故 Phase 4 自行派生 `obs_hist_*` 列（窗口严格 ≤ `context_end`），并禁用 `stress_` 前缀，使两者不可能混淆。
- 例外：`stress_efc` / `stress_ah_throughput` / `stress_wh_throughput` 天然累积，其截止点值**就是**累积至今的历史，派生列已验证相等。

窗口审计：`n_rows = 34335`、`n_unmatched_rows = 0`、`max_reference_index_offset_vs_context_end = 0`、`window_bound_respected = true`、`aggregation_is_prefix_only = true`。

### 3.3 25 个准入字段（`obs_hist_` 前缀，聚合口径 mean/max/last/cum）

温度族（5）：`mean_mean_temp_c`、`max_mean_temp_c`、`last_mean_temp_c`、`mean_max_temp_c`、`max_max_temp_c`
DoD 族（3）：`mean_dod`、`max_dod`、`last_dod`
C-rate 族（3）：`mean_mean_c_rate`、`max_mean_c_rate`、`last_mean_c_rate`
累积吞吐（2）：`cum_efc`、`cum_ah_throughput`
地影/任务（4）：`mean_eclipse_h_per_day`、`last_eclipse_h_per_day`、`mean_mission_comm_ratio`、`last_mission_comm_ratio`
光照几何（4）：`mean_mean_solar_power_w`、`mean_max_solar_power_w`、`mean_mean_effective_area_m2`、`mean_mean_abs_beta_deg`
电池状态（4）：`last_capacity_ah`、`mean_rint_ohm`、`last_rint_ohm`、`last_age_days`

其中 CORE 组 6 列：`mean_mean_temp_c`、`max_max_temp_c`、`mean_dod`、`mean_mean_c_rate`、`cum_efc`、`cum_ah_throughput`。

### 3.4 消融结果（`feature_ablation_results.csv`，lofo_100pct / T0_DELTA_REFERENCE）

| 特征组 | 族平均 MAE | 最差族 MAE | 跨族 std |
|---|---|---|---|
| **F0_TARGET_ARC_SPACE_GEOMETRY11**（交付） | **0.003328** | 0.008706 | 0.001026 |
| F3_FULL_OBSERVABLE_HISTORY_STRESS（+25 应力） | 0.003448 | 0.008576 | 0.000658 |
| F2_CORE_OBSERVABLE_HISTORY_STRESS（+6 应力） | 0.003577 | 0.010012 | 0.001031 |
| B0_OBSERVABLE_LOCAL_TREND（非学习） | 0.003588 | 0.008540 | 0.001106 |
| F4_TARGET_NATIVE_GEOMETRY11（目标域原生空间） | 0.004271 | 0.011855 | 0.001411 |
| PERSISTENCE | 0.011027 | 0.024239 | 0.002852 |

**结论（Phase 4 原文口径）：`any_gate_passed = false`，`all_gates_failed = true`，`phase3_verdict_unchanged = SOURCE_PRIOR_NOT_VALIDATED`。**

即：加入 25 个应力通道后族平均 MAE **未改善**（0.003448 > 0.003328）。这是**阴性结果，不是执行失败**（契约明示 `negative_result_is_not_execution_failure = true`）。

### 3.5 一条**可以**说的正面结论

F0（ARC 迁移空间，0.003328）优于 F4（目标域原生空间，0.004271）。这是坐标空间对照 `coordinate_space_control_F4_vs_F0` 的结果，即**特征空间迁移带来的已验证预处理效应**。

⚠️ 红线：不得由此对任何应力因子做**因果方向断言**。无生成器 ⇒ 无严格反事实，证据上限为 `OBSERVATIONAL_EVIDENCE`。

---

## 4. 退化机理假设（G4）

| 机理 | 方程形式 | 可观测映射 | 系数状态 |
|---|---|---|---|
| 容量衰减（日历） | dQ/dt = −A_cal·exp(−Ea/RT)·f(SOC)·t^z | `temperature_c` → `q_max_ah_true` → `capacity_ah` → SOH | ❌ 未记录 |
| 容量衰减（循环） | dQ/dN = −B_cyc·DoD^p·C_rate^q·exp(−Eb/RT) | `dod_orbit`、`mean_c_rate`、`efc` → SOH | ❌ 未记录 |
| 内阻增长 | R_int(t) = R_0(1 + γ(1−SOH)^m) | `rint_ohm_true` → 放电压降增大、温升加剧 | ❌ 未记录 |
| 端电压响应 | V = OCV(SOC) − I·R_int | `voltage_v` | ✅ 形式已知 |
| 热响应 | dT/dt = (I²R_int − h(T−T_bnd))/C_th | `temperature_c`、`thermal_boundary_c` | ✅ 形式已知 |

**方程形式已知，系数未记录。** 归档 `电池这块的技术方案.md` §7.1 给出的是**演示参数**（标称容量 100 Ah），而 L1 实测 `q_max_ah_true` 起始为 **40.0 Ah** —— 二者不符，直接证明归档参数不能用于重建生成器。详见 `src/twin/SPEC.md` §4。

因此：机理**假设**成节可写（本节），机理**复现**标 PENDING。不反解系数冒充原始生成参数。

---

## 5. 证据域限定

- 全链路证据为 **SIMULATION_ONLY**（STK 导出 + Python 数字孪生），**不是真实卫星精度**。
- L1 为 72 h 高分辨率模板（30 s 步长），非全 2190 天秒级遥测；长寿命轨迹由模板重放 + 多年加速退化推进（归档 §6.7 的显式设计）。
- 星上 SOH 估计器的三条路径（库仑计数 / 电压平台 / 内阻脉冲）在本项目中由孪生实现，**未在真实遥测上标定**。
- `.sc` STK 场景文件项目内 0 个；证据链止于 24 场景的 CSV 导出层。
- 不得将异常标签等价成失效寿命标签；不得在无目标域校准标签时声称区间在真实卫星上有效。
