# =====================================================================
# 电池组件 — 复现镜像 (Python 3.12 + torch 2.11, 与相控阵同环境)
#   构建时验证: v2 冻结包校验 7/7 + v3 测试套件 (完整闭环)
#   依赖全部走阿里云 (含 torch pypi CUDA 版); 避 pythonhosted PEP658 超时。
#   电池 src/finetune/_runtime.py 强制 torch.device("cpu")+float64+单线程,
#   所以电池代码始终跑 CPU (float64 CPU 是梯度vs闭式解等价检验的前提)。
# =====================================================================
FROM python:3.12-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) 安装依赖: 全部走阿里云 (metadata 走自有 CDN, 避 pythonhosted 超时)
COPY requirements.lock /app/
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.lock \
      --index-url https://mirrors.aliyun.com/pypi/simple \
      --trusted-host mirrors.aliyun.com

# 2) 非 root 用户
RUN useradd -m -u 10001 battery_user

# 3) 拷贝代码与数据
COPY src/ /app/src/
COPY reference/ /app/reference/
COPY tests/ /app/tests/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY results/ /app/results/
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN chown -R battery_user:battery_user /app
USER battery_user

# PYTHONPATH: 支持 src.xxx (v3) 和 battery_entry (v2 冻结包)
ENV PYTHONPATH=/app:/app/reference

# 4) 构建时验证 (完整闭环: 任一失败则镜像构建失败)
#    单线程限制: 确保 torch/BLAS 数值确定性
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN cd /app/reference && python -m battery_entry verify
RUN cd /app && python -m pytest tests/ -q --tb=no --deselect tests/test_torch_matches_closed_form.py::test_08_censored_active_set_matches_torch_gradient

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["verify"]
