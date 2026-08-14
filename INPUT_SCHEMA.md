# Battery 输入遥测约定

> 本文件定义电池组件的输入遥测字段规范。校验器 `scripts/validate_input_schema.py`
> 按此约定对 CSV 输入做字段存在性、单位一致性与数据类型检查。
>
> 数据来源：STK 仿真的 LEO 通信卫星锂离子电池组（6 个轨道环境族 × 多种子轨迹）。
> 参见 `docs/battery_data_dictionary.csv` 与 `reference/examples/example_input.csv`。

## L1 遥测层（raw telemetry，30 s 采样）

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `timestamp_utc` | ISO 8601 | 30 s | 否 | UTC 时间戳，仅作输出溯源，不参与特征计算 |
| `voltage_v` | V | 30 s | 是 | 电池端电压，来自电池模型遥测 |
| `current_a` | A | 30 s | 是 | 充放电电流，来自电源平衡遥测 |
| `temperature_c` | °C | 30 s | 是 | 电芯表面温度，来自热模型遥测 |

## L2 派生层（derived metrics，每轨/14 d 采样）

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `soc` | — (无量纲) | 30 s | 否 | 荷电状态 (State of Charge)，BMS/仿真在线估计，范围 (0, 1] |
| `dod_orbit` | — (无量纲) | 每轨 | 否 | 每轨放电深度 (Depth of Discharge)，累计派生量 |
| `efc` | — (cycles) | 每轨 | 否 | 等效满循环数 (Equivalent Full Cycles)，累计派生量 |
| `ah_throughput` | Ah | 14 d | 否 | 安时吞吐量累计 |
| `wh_throughput` | Wh | 14 d | 否 | 瓦时吞吐量累计 |
| `mean_temp_c` | °C | 14 d | 否 | 窗口均温（老化模型输入） |
| `max_temp_c` | °C | 14 d | 否 | 窗口峰温（老化模型输入） |
| `mean_c_rate` | — (1/h) | 14 d | 否 | 窗口平均 C-rate（老化模型输入） |

## L3 预测输入层（predictor input，14 d 网格）

预测接口 (`battery_entry predict`) 的最小必需列：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `battery_id` | — | — | 是 | 电池标识符（字符串），用于按电芯分组 |
| `reference_index` | — (整数序号) | 14 d | 是 | 参考点序号，必须严格递增（整数） |
| `soh_observed` | — (无量纲) | 14 d | 是 | 在线观测 SOH，范围 (0, 1.5]；>1.5 判为百分尺度误用 |

### 禁止列（leakage guard）

以下列在预测输入中被拒绝（详见 `reference/battery_entry/schema.py`）：

- `_true` 后缀列（hidden-truth 通道：`soh_true`, `rint_ohm_true` 等）
- `target_*` 前缀列（未来/标签侧信息）
- `future_realized_*` / `future_planned_*`（oracle / exposure 通道）
- `environment_family_id`, `split`, `trajectory_id`（身份/元数据代理）
- `altitude_km`, `inclination_deg`, `raan_deg`（轨道三元组 → 族身份映射）

## 内部状态真值（hidden truth，仅用于标签/审计，不进输入）

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `q_max_ah_true` | Ah | 14 d | 否（仅标签） | 最大容量真值，老化模型产出 |
| `rint_ohm_true` | Ω | 14 d | 否（仅标签） | 内阻真值，老化模型产出 |
| `soh_true` | — (无量纲) | 14 d | 否（仅标签） | SOH 真值 = Q/Q₀ |
| `rul_days` | day | 14 d | 否（仅标签） | 剩余寿命真值（EOL 判据派生） |

## 数值约定

- `soh_observed` 必须严格 > 0 且 ≤ 1.5；>1.5 时校验器报"疑似百分尺度 (0..100)，请除以 100"
- `reference_index` 必须为整数，且同一 `battery_id` 下严格递增
- 每条电池历史至少 20 个参考点（ARC route context length）
