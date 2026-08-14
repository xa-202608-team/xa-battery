# 电池组件 RC 移交说明

本文件描述 `xa-battery` 公开仓库代码基线与本地大型工件之间的移交契约。

## 槽位与交付包映射

`handoff/artifact-map.yaml` 是唯一权威映射，当前为：

| 本地槽位 | 交付包路径 | 角色 |
|----------|-----------|------|
| `data/` | `04_数据/battery` | dataset |
| `checkpoints/` | `03_代码/components/battery/checkpoints` | checkpoint |
| `results/reference/` | `05_结果/reference/battery` | reference_result |

规则：

- 映射只允许相对路径；本地不存在时 RC 打包必须失败而不是生成空包。
- 公开仓库不携带数据/权重/结果工件；对应槽位只保留 README 与 manifest。
- 填充本地工件后须同步更新 `data/data_manifest.json`、
  `checkpoints/checkpoint_manifest.json`、`results/public_summary.json`
  （登记相对路径 + SHA256 + 来源），并把状态从 `LOCAL_ARTIFACT_REQUIRED` /
  `NOT_YET_VERIFIED` 迁移到实际状态。

## 契约与门禁

- 契约版本：`component-contract-v1.1.0`（槽位 manifest schema 1.1.0）。
- 边界由 `tests/test_repository_boundary.py` 强制：`data/`、`results/`、`checkpoints/`
  只有白名单文件可被跟踪，`reference/data/`、`reference/models/`、`reference/results/`
  不得有受跟踪文件。
- 运行说明见根目录 `RUN.md`；输入契约见 `INPUT_SCHEMA.md`；来源与许可见 `PROVENANCE.md`；
  数据获取要求见 `DATA_REQUIREMENTS.md`。
