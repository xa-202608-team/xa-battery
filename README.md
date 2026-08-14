# XA-202608 Battery Component

本私有仓库只承载电池组件。当前契约版本：`component-contract-v1.1.0`（槽位 manifest schema 1.1.0）。
正式组件版本只能由 `@xa-202608-team/integrators` 签发。
源码迁移、测试和 Docker 门禁将在电池迁移计划中完成。

## 运行说明（自 XA-202608 最终交付快照导入）

本基线由快照导入工具按白名单策略从最终交付的电池组件复制而来，
不含数据、权重与结果工件（见下方"本地工件槽位"）。

- **如何运行**：见 `RUN.md`（端到端入口、Docker 用法、输出目录约定）。
- **输入契约**：见 `INPUT_SCHEMA.md` 与 `reference/schemas/predict_input_schema.json`。
- **来源与许可**：见 `PROVENANCE.md` 与 `reference/DATA_CARD.md`。
- **数据获取**：见 `DATA_REQUIREMENTS.md`（NASA 公开电芯退化数据下载与派生要求）。
- **冻结参考实现**：`reference/`（battery_entry CLI、release 配置、模型/数据卡与复现文档）。
- **依赖锁定**：`requirements.lock`（reference 侧另有 `reference/requirements.lock`）。

## 本地工件槽位

`data/`、`checkpoints/`、`results/` 是本地工件槽位，除 README 与 manifest 外
不入库（`tests/test_repository_boundary.py` 强制）。填充方式与 RC 打包映射见
`handoff/HANDOFF.md` 与 `handoff/artifact-map.yaml`。当前状态：
`LOCAL_ARTIFACT_REQUIRED`（数据/权重）、`NOT_YET_VERIFIED`（公开指标摘要）。
