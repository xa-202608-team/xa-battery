# checkpoints/ 本地权重槽位

本目录是电池组件的本地模型权重/校准工件槽位，**不承载任何受 Git 跟踪的权重文件**。

## 用途

- 预训练/微调权重、冻结参数（如 `frozen_alpha.json`、conformal 分位数）等
  通过交付包 `03_代码/components/battery/checkpoints` 落位到本地，
  路径映射见 `handoff/artifact-map.yaml`。
- 每次填充本地权重后，须在 `checkpoint_manifest.json` 登记条目
  （相对路径 + SHA256 + 训练配置摘要），状态从 `LOCAL_ARTIFACT_REQUIRED` 迁移到实际状态。

## 边界约束

- 除本 README 与 `checkpoint_manifest.json` 外，`checkpoints/` 下任何文件不得被 Git 跟踪
  （由 `tests/test_repository_boundary.py` 与 `.gitignore` 强制，`*.pt/*.pth/*.ckpt` 亦被全局忽略）。
- 权重必须可由源码 + 配置复现；本地工件缺失时相关验收显式记为
  `LOCAL_ARTIFACT_REQUIRED`，不得伪造。
