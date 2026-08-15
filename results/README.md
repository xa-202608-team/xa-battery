# results/ 本地结果槽位

本目录是电池组件的本地实验结果工件槽位，**不承载任何受 Git 跟踪的结果文件**。

## 用途

- 实验输出（对比表、prognostic 指标、conformal 覆盖率、孪生验证 JSON 等）一律写入
  本地槽位，由交付包 `05_结果/reference/battery` 提供冻结参考（映射到 `results/reference/`），
  路径映射见 `handoff/artifact-map.yaml`。
- `public_summary.json` 只登记已经公开报告且逐项核实过的冻结摘要；
  当前数字来源尚未逐项核实，状态为 `NOT_YET_VERIFIED`、`metrics` 为空，
  **不得编造指标或哈希**。
- `expected_metrics.json` 登记复现验收的期望指标；在核实并冻结前同样保持
  `NOT_YET_VERIFIED`、空 `metrics`。

## 边界约束

- 除本 README、`public_summary.json`、`expected_metrics.json` 外，`results/` 下任何文件
  不得被 Git 跟踪（由 `tests/test_repository_boundary.py` 与 `.gitignore` 强制）。
- 结果必须可由源码 + 本地数据/权重复现；本地工件缺失时相关验收显式记为
  `LOCAL_ARTIFACT_REQUIRED` / `NOT_YET_VERIFIED`，不得伪造。
