# 电池组件 — PyTorch 权威路径声明

> 赛题硬门槛：代码基于 PyTorch 并可完整复跑。本文件界定**权威结果来源**与
> numpy 可移植版本的定位，避免"哪个是正式算法"的歧义。

## 一句话结论

**正式作品的全部权威结果由 v3 PyTorch pipeline（`src/`）生成；v2 冻结的
纯 numpy 可移植版（`reference/battery_entry/`）只读保留，仅用于数值一致性回归验证。**

## 两条路径的界定

| 路径 | 位置 | 角色 | 是否产出权威结果 |
|------|------|------|----------------|
| **v3 PyTorch pipeline** | `src/`（当前交付代码） | 权威：预训练 + 迁移求解 + RUL + 评估 | ✅ 唯一权威 |
| **v2 numpy 可移植版** | `reference/battery_entry/`（冻结只读） | 数值一致性回归对照（SHA-256 守卫） | ❌ 仅回归验证 |

### v3 PyTorch pipeline 中各环节的求解器

| 环节 | 模块 | 求解器 | PyTorch 的角色 |
|------|------|--------|---------------|
| 源域预训练 | `src/pretrain/run_pretrain.py` | `LinearHead(nn.Module)` + AdamW + 早停 | **torch 训练权威** |
| L2-SP 迁移 | `src/finetune/run_l2sp.py` | 凸二次规划**闭式解**（`solve_l2sp_closed_form`） | torch 梯度解做交叉核对（`max_abs_gradient_minus_closed_form` 检验两解一致） |
| RUL（全任务期） | `src/rul/run_mission_span.py` | 活动集迭代（凸分段二次全局最优） | 确定性求解 |
| 评估/对比 | `src/comparators/run_all.py` | 七臂对比 + 闭式/迭代求解 | GRU 臂为 torch 实现 |

### 关于"闭式解"的诚实说明

L2-SP 迁移的主求解器是**凸二次规划的解析闭式解**（`beta_hat = (Z'WZ + (alpha+λ)I)^-1 (Z'Wy + λ·βsrc)`），
不是 torch 训练。这是**刻意的确定性选择**：

- 目标函数严格凸二次 → 闭式解是全局最优点，无学习率/调度/随机性，可复现性最强；
- PyTorch 梯度解（AdamW + LBFGS 精修）与该闭式解交叉核对，
  `tests/test_torch_matches_closed_form.py` 断言两者到达同一最优点；
- 因此"权威结果由 PyTorch pipeline 生成"的准确含义是：**全流程由 v3 链产出，
  PyTorch 覆盖模型训练与数值验证，凸优化环节用确定性解析解并由 torch 梯度证明其最优**。

## 全流程复跑入口（全部来自 v3）

```bash
# 环境：Python 3.12 + torch 2.11+cpu（见 requirements.lock，与相控阵组件同环境）
python -m src.pretrain.run_pretrain        # 源域预训练（torch）
python -m src.finetune.run_l2sp            # L2-SP 迁移（闭式解 + torch 核对）
python -m src.rul.run_mission_span         # 全任务期 RUL
python -m src.metrics.run_prognostic       # 提前性指标
python -m src.comparators.run_all          # 七臂对比
python -m src.twin.fault_injection         # 故障注入（物理量 + 近似传播）
python -m pytest tests/ -q                 # 全量测试
```

## v2 numpy 可移植版的定位（回归验证）

- `reference/battery_entry/` 是**只读冻结**的 v2 交付（SHA-256 守卫
  `tests/test_self_contained.py` 断言其未被改动）；
- 它提供 `verify`（7/7）与 `reproduce --mode quick`（逐位复现 max|diff| < 1e-12），
  用于证明 v3 的闭式/迭代求解与 v2 的 numpy 可移植实现**数值一致**；
- **不产出**报告中的任何权威指标，只作一致性回归。

## 确定性保证

float64 / CPU / 单线程（`OMP_NUM_THREADS=1` 等）/ 确定性算法，见
`src/finetune/_runtime.py` 与 Dockerfile 的单线程环境变量。

## 环境统一说明（2026-08-10）

电池组件与相控阵组件**共用同一基础环境**：Python 3.12 + torch 2.11（见
`requirements.lock`，版本与 conda env `pytorch_gpu` 对齐）。v2 冻结包
（`battery_entry`）已在此环境下验证 `verify 7/7` 通过；v3 全流程（pretrain/l2sp/
rul/prognostic/comparators）在此环境下复跑结果与原 3.9.25 环境一致（闭式解
确定性，版本升级不影响解析解）。
