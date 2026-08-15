"""_runtime.py — deterministic float64 CPU setup for every torch path here.

Adapted from ``src/stk_transfer_v2/models/_runtime_p3.py`` (see ``REUSE_MAP.md``).
That module's comment records that the NumPy-MKL / torch OpenMP initialisation ORDER
was determined by measurement in this environment and must not be reordered — so
this module is imported FIRST by every torch-using module in the package.

Everything here exists to make "the gradient solution equals the closed form to
< 1e-6" a meaningful assertion rather than a lucky run: float64, CPU, single thread,
no nondeterministic kernels.
"""
from __future__ import annotations

import os

# Set before torch loads its OpenMP runtime. On this Windows/Anaconda box NumPy pulls
# in MKL's libiomp first; without this, importing torch afterwards can abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402  (must follow the env vars)
import torch  # noqa: E402

DTYPE = torch.float64
DEVICE = torch.device("cpu")

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
try:
    torch.use_deterministic_algorithms(True)
except Exception:                                   # pragma: no cover
    pass


def seed_everything(seed: int = 0) -> None:
    """Fix every RNG this package could touch. Called by each entry point."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_tensor(a) -> "torch.Tensor":
    """float64 CPU tensor from anything array-like, contiguous."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(DEVICE, DTYPE)


def runtime_report() -> dict:
    return {
        "torch_version": torch.__version__,
        "dtype": str(DTYPE), "device": str(DEVICE),
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
    }
