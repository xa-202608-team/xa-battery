# 数字孪生任务期老化重建规范

**版本：** reconstruction v1，2026-08-10  
**状态：** `CALIBRATED_RECONSTRUCTION_NOT_ORIGINAL_SOURCE`  
**运行入口：** `python -m src.twin.run_reconstruction`

## 1. 结论与边界

原交付包曾写成 `PENDING_AGING_COEFFICIENTS`。补充核查完整 `afterstk`
数据包后，发现这个判断需要更正：

- 原始生成器源码仍然不存在；
- 但 `trajectory_manifest.csv` 完整记录了 120 条轨迹的逐轨迹老化参数；
- `annual_environment_family_calendar.csv` 完整记录了 6 个环境族、365 天的环境驱动；
- 完整 L2 表可作为结构校准与留族检验的参考真值。

因此现在能够重新积分并生成新的 `soh_true`，但必须准确称为：

> 基于记录参数与环境驱动的校准重建版生成器，而不是丢失的原始生成器源码。

不得声称逐位恢复了原程序或其随机数序列，也不得把仿真结果表述为真实卫星精度。

## 2. 自包含输入

`data/twin_reconstruction/` 包含：

| 文件 | 作用 | 是否含 SOH 真值 |
|---|---|---:|
| `stress_driver_v1.csv` | 14,285 行环境/暴露驱动 | 否 |
| `trajectory_parameters_v1.csv` | 120 条逐轨迹参数 | 否 |
| `annual_environment_family_calendar.csv` | 6 族年度环境日历 | 否 |
| `calibration_reference_truth.csv` | 结构校准与验收 | 是，仅验证读取 |
| `reconstruction_config_v1.json` | 四个全局校正量、协议和指标 | 否 |

生成函数 `reconstruct()` 从不读取 `calibration_reference_truth.csv`。测试会拒绝在
stress driver 中出现 `soh_true`、容量、内阻或 EOL 等输出字段。

## 3. 记录参数

逐轨迹参数来自完整 `afterstk/trajectory_manifest.csv`，包括：

- 实际初始容量 `q0_ah`：36.13–44.00 Ah；
- 初始内阻 `initial_rint_ohm`：0.0381–0.0539 Ω；
- 日历/循环系数 `k_cal`、`k_cyc`；
- DoD、C-rate 指数；
- 温度敏感度；
- 膝点起始损失与增益；
- 内阻指数与传感器噪声标准差。

原归档的 100 Ah 是演示值，重建器不会读取或使用它。验收测试明确断言所有
`q0_ah` 均落在实测记录范围内且不存在 100 Ah。

## 4. 老化方程

对每个 14 天参考步，先计算未校正的日历与循环损失：

\[
\Delta L_{cal}=k_{cal}\,\Delta\sqrt{t}\,
\exp[k_T(T_{max}-25)](1+\beta_{cal}\,SOC_{mean})
\]

\[
\Delta L_{cyc}=k_{cyc}\,\Delta EFC\,DoD^p\,C_{rate}^q\,
\exp[k_T(T_{mean}-25)]
\left(1+\beta_{cyc}\frac{|\beta|}{90}\right)
\]

全局校正后累计基础损失：

\[
L_{raw}(t)=\sum_{\tau\le t}
(s_{cal}\Delta L_{cal}+s_{cyc}\Delta L_{cyc})
\]

膝点加速：

\[
L(t)=L_{raw}(t)+s_k g_k
\max[0,L_{raw}(t)-(L_k+\delta_k)]
\]

最终：

\[
SOH_{true}=\mathrm{clip}(1-L,0,1),\qquad
Q_{max}=Q_0SOH_{true}
\]

`s_cal`、`s_cyc`、`s_k`、`delta_k` 是唯一的全局重建校正量，完整保存在
`reconstruction_config_v1.json`。其余参数均直接使用 manifest 记录值。

## 5. 校准协议和新颖族验收

全局校正量只使用：

- `target_train`：2 个环境族；
- `target_calibration`：2 个环境族。

以下两个 `target_test` 环境族不进入校准：

- `h500_i070_raan000`；
- `h550_i053_raan180`。

当前固定验收门槛：

| 指标 | 门槛 | 当前结果 |
|---|---:|---:|
| 留出测试族 SOH MAE | ≤ 0.0035 | 0.002539 |
| 留出测试族 \(R^2\) | ≥ 0.995 | 0.996942 |
| SOH 单调性违规 | 0 | 0 |

这些是对历史仿真轨迹的重建一致性指标，不是真实卫星验证。

## 6. 反事实能力的准确口径

重建器支持三个单因素方向性检查：

- 温度 +5 °C；
- DoD ×1.20；
- C-rate ×1.20。

在固定其他输入时，三者均不得令最终 SOH 上升。该检查说明**方程实现符合预先
规定的单调方向**，可以用于软件级反事实和敏感性测试；它不证明这些方向或幅度
已经由真实卫星因果实验验证。

故障注入仍分两层陈述：

- `reconstruction.py`：方程级应力扰动，可重新积分 SOH；
- `fault_injection.py`：既有轨迹上的后处理式异常注入，用于预测器鲁棒性。

## 7. 复现命令

```bash
python -m src.twin.calibrate_reconstruction
python -m src.twin.run_reconstruction
python -m pytest tests/test_twin_reconstruction.py -q
```

正常标志：

```text
CALIBRATION_OK
TWIN_RECONSTRUCTION_OK
5 passed
```

## 8. 仍未恢复的内容

- 原始生成器源码和逐位随机数序列；
- 完整 STK `.sc` 场景文件；
- 真实卫星在轨老化标定；
- 内阻通道的精确重建。当前内阻是全局近似，SOH 才是主验收状态。

这些限制不妨碍任务期 `soh_true` 的自包含重积分，但必须继续保留在报告边界中。

