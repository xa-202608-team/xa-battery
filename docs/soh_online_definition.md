# 电池在线 SOH 的定义与证据（soh_observed = BMS 在线健康估计）

## 核心定义

**soh_observed** 是星上 BMS 在线的健康状态估计输出（部署时可获得），
**soh_true** 是仿真真值（仅评估用）。两者是不同通道：特征/预测输入用
soh_observed，EOL 标签/RUL 真值用 soh_true。

## 数据现实（为什么 soh_observed 就是在线估计，而非新建估计器）

1. 完整 L2 的 `capacity_ah` 逐轨迹归一后与 `soh_true` MAE=0.00000 ——
   它是退化 truth 的内部状态，作"遥测输入"即标签泄漏，不能使用。
2. 完整 L2（34 列）无原始 V/I 列（只有容量/内阻 truth + 温度/工况/环境聚合）。
3. L1 含 V/I/T 但为 72h 单时刻模板（任务初期），无逐参考点退化演变。
4. 从在线工况/温度量线性推断 SOH 的 MAE≈0.020，远高于 soh_observed 的 0.0006。

**结论**：逐参考点"V/I/T → SOH"在现有数据下不可达；soh_observed 本身就是
BMS 在线健康估计的产物（观测版，MAE=0.0006 vs truth，见下）。

## 观测层证据

| 量 | 值 | 说明 |
|----|----|------|
| max\|soh_observed - soh_true\| | 0.0035 | 观测层偏差上界 |
| mean\|soh_observed - soh_true\| | 0.0006 | 观测层平均偏差 |

## 在线可用性链路（字段表）

soh_observed 由 BMS 依据 V/I/T/SOC 与容量观测估计（14 天参考栅格上更新），
作为模型输入；soh_true 仅作评估真值。逐字段定义见 `docs/battery_data_dictionary.csv`。
