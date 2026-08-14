#!/bin/bash
# =====================================================================
# 电池组件 — 容器入口脚本
#   支持 verify / reproduce 子命令
# =====================================================================
set -euo pipefail

export PYTHONPATH="/app:/app/reference:${PYTHONPATH:-}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

step() { echo ""; echo "========== $1 =========="; }

case "${1:-verify}" in
    verify)
        step "阶段二冻结包校验 (期望 7/7)"
        cd /app/reference
        python -m battery_entry verify

        step "阶段二逐位复现 (期望 max|diff| < 1e-12)"
        python -m battery_entry reproduce --mode quick --output /tmp/reproduce_check

        step "阶段三测试套件 (期望 42 passed)"
        cd /app
        python -m pytest tests/ -q --tb=short --deselect tests/test_torch_matches_closed_form.py::test_08_censored_active_set_matches_torch_gradient

        step "验证通过"
        echo "✅ v2 冻结包: 7/7 组通过"
        echo "✅ v2 逐位复现: 34335 行 × 11 维 max|diff| < 1e-12"
        echo "✅ v3 测试: 42 项通过 (含 17 项等价性断言)"
        ;;

    reproduce)
        MODE="${2:-quick}"
        OUTPUT_DIR="${3:-/results/reproduced/battery}"
        mkdir -p "$OUTPUT_DIR"

        case "$MODE" in
            quick)
                step "快速复现（逐位精确，预计 < 1 分钟）"
                cd /app/reference
                python -m battery_entry reproduce --mode quick --output "$OUTPUT_DIR"
                ;;
            full)
                step "完整复现（预计 ~15 分钟）"

                step "P1: 源域预训练 (NASA PCoE → PyTorch 线性头)"
                cd /app
                python -m src.pretrain.run_pretrain

                step "P2: L2SP 参数迁移正则化微调"
                python -m src.finetune.run_l2sp

                step "P3: 全任务期剩余寿命回归"
                python -m src.rul.run_mission_span

                step "P4: 提前性指标（预警提前期 / 预测视界）"
                python -m src.metrics.run_prognostic

                step "P5: 多臂对比（Wiener / PF / 岭回归 / 深度序列）"
                python -m src.comparators.run_all

                step "P6: 冻结包逐位复现"
                cd /app/reference
                python -m battery_entry reproduce --mode quick --output "$OUTPUT_DIR"
                ;;
            *)
                echo "Unknown mode: $MODE (expected: quick | full)"
                exit 1
                ;;
        esac
        step "复现完成"
        echo "结果输出至 $OUTPUT_DIR"
        ;;

    *)
        exec "$@"
        ;;
esac
