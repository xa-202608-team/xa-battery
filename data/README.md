# data/ 本地数据槽位

本目录是电池组件的本地数据工件槽位，**不承载任何受 Git 跟踪的数据文件**。

## 用途

- 数据集（NASA B0005/B0006/... 公开电芯退化 npz、STK 场景参考点、孪生重建输入等）
  通过交付包 `04_数据/battery` 落位到本地，路径映射见 `handoff/artifact-map.yaml`。
- 每次填充本地数据后，须在 `data_manifest.json` 登记条目（相对路径 + SHA256 + 来源与许可），
  状态从 `LOCAL_ARTIFACT_REQUIRED` 迁移到实际状态。
- 数据获取方式与字段级要求见根目录 `DATA_REQUIREMENTS.md` 与 `INPUT_SCHEMA.md`。

## 边界约束

- 除本 README 与 `data_manifest.json` 外，`data/` 下任何文件不得被 Git 跟踪
  （由 `tests/test_repository_boundary.py` 与 `.gitignore` 强制）。
- 不得为了通过测试把实际工件重新提交入库；本地工件缺失时相关验收显式记为
  `LOCAL_ARTIFACT_REQUIRED`，不得伪造。
