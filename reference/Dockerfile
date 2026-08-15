# Minimal CPU image for battery_release_v2.
#
# Deliberately does NOT contain:
#   * STK itself — the package depends only on data ALREADY EXPORTED from STK, so a
#     downstream reproduction never needs an STK licence or installation;
#   * the parent research project — the package is self-contained, and
#     `python -m battery_entry verify` proves it mechanically;
#   * CUDA / GPU wheels — every result in this package was produced on CPU;
#   * any network access at run time — every input is inside the image.
#
# Build:
#   docker build -t battery-soh-v2 .
# Verify:
#   docker run --rm battery-soh-v2 python -m battery_entry verify
# Quick reproduction, writing to a mounted host directory:
#   docker run --rm -v "$PWD/out:/out" battery-soh-v2 \
#       python -m battery_entry reproduce --mode quick --output /out

# Pinned by digest-free exact tag: the Python minor version is load bearing because the
# frozen numbers were produced on 3.9.
FROM python:3.9.25-slim-bookworm

# PYTHONDONTWRITEBYTECODE keeps the image and any mounted volume free of .pyc files,
# which would otherwise appear as untracked files in a SHA256SUMS check.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so the (large, slow) wheel layer is cached independently of the
# package contents.
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock

# Then the package itself.
COPY battery_entry/ ./battery_entry/
COPY configs/      ./configs/
COPY models/       ./models/
COPY data/         ./data/
COPY schemas/      ./schemas/
COPY results/      ./results/
COPY tests/        ./tests/
COPY docs/         ./docs/
COPY examples/     ./examples/
COPY README.md MODEL_CARD.md DATA_CARD.md LIMITATIONS.md REPRODUCIBILITY.md ./
COPY CLAIMS_MANIFEST.json release_manifest.json SHA256SUMS ./

# Run as a non-root user, so files written into a mounted output directory are not
# root-owned on the host. UID/GID 10001 is outside the usual host range to reduce the
# chance of colliding with a real host user.
RUN groupadd --gid 10001 battery \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin battery \
 && mkdir -p /out \
 && chown -R battery:battery /app /out
USER battery:battery

# Fail the build if the package is not internally consistent. This runs the manifest,
# SHA256, isolation, model-consistency and frozen-verdict checks, so a broken image
# cannot be produced silently.
RUN python -m battery_entry verify

VOLUME ["/out"]

# Default entry point runs the quick reproduction into the mounted volume.
ENTRYPOINT ["python", "-m", "battery_entry"]
CMD ["reproduce", "--mode", "quick", "--output", "/out"]
