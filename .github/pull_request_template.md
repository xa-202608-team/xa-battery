# Pull Request

## 关联 Issue

<!-- 必填：Closes #<编号>，或说明无对应 Issue 的原因 -->

## 变更范围

<!-- 必填：本次改动的目录/模块列表，如 configs/、src/train/ -->

## 明确不做内容

<!-- 必填：本次 PR 明确不涉及、留待后续的内容，防止范围蔓延 -->

## 契约版本

<!-- 必填：本次变更遵循/变更的组件契约版本（如 component-contract-v1.1.0）；涉及槽位 manifest schema 变更须注明新旧版本 -->

## 单指标影响

<!-- 必填：本次变更影响的单一核心指标及方向（如 RMSE、覆盖率、PHM 分数），无影响填"无"并说明理由 -->

## 是否需要 RC

<!-- 必填：本次变更是否触发新的 RC 打包（是/否）；若是，列出 handoff/artifact-map.yaml 涉及的本地槽位 -->

## 无标签泄漏

<!-- 必填：确认监督标签派生与 train/val 划分无同轨迹/同个体泄漏（机台/电芯/器件级划分），或说明不涉及 -->

## 公开扫描

<!-- 必填：scan_public_repo --root . 的退出码与结果；若新增可能含敏感内容的文件须注明已复查 -->

## 数据/配置版本

<!-- 必填：涉及的数据集名称/版本/SHA256，配置文件版本或 config_sha256 -->

## 验收命令

<!-- 必填：在组件仓库根目录执行的验收命令及退出码，如 verify / reproduce --mode quick --output /outputs -->

## metrics.json

<!-- 必填：粘贴本次运行的 metrics.json 关键内容（或链接），须符合 schemas/metrics.schema.json -->

## manifest.json

<!-- 必填：粘贴本次运行的 manifest.json 关键内容（或链接），须符合 schemas/manifest.schema.json -->

## 风险

<!-- 必填：本次变更可能引入的风险（复现性、性能、兼容性等）及影响面 -->

## 回退方式

<!-- 必填：出问题时如何回退（revert commit / 版本号 / 数据回滚等） -->
