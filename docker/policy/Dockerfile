# behavior26 submission policy: arm B (pi05_b1k_frozen_vlm_progress, 6,000 steps, fine-tuned
# from the organisers' released turning_on_radio checkpoint), served over the BEHAVIOR-1K
# evaluator's websocket protocol on port 8000, with GET /healthz.
#
# This is the SAME server that produced our public-test results: openpi's
# scripts/b1k/serve_b1k.py at the pinned commit plus our patches, started with the arguments
# scripts/serve_baseline.sh passed during evaluation. Build with scripts/build_policy_image.sh,
# which stages the checkpoint into the build context.
#
# Runtime needs: one NVIDIA GPU with >= 24 GB (uses ~6 GB for weights), recent NVIDIA driver
# (CUDA 12 user-space libraries ship inside the image via jax[cuda12]), the NVIDIA container
# runtime. The model computes in bfloat16.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    OPENPI_DATA_HOME=/opt/openpi-cache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates build-essential clang ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.8.17/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ARG OPENPI_COMMIT=0cc8e355f7bac0976db1cc3139b1ff0379feea60
RUN git clone --filter=blob:none https://github.com/wensi-ai/openpi.git /opt/openpi \
    && git -C /opt/openpi checkout -q ${OPENPI_COMMIT}

WORKDIR /opt/openpi
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen 2>&1 | tail -3 \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . 2>&1 | tail -2

# Our patches (robot-name compatibility, the progress-head config, task prompts = task names).
COPY scripts/apply_openpi_patches.sh /opt/behavior26/scripts/apply_openpi_patches.sh
COPY training/patches /opt/behavior26/training/patches
RUN OPENPI_ROOT=/opt/openpi bash /opt/behavior26/scripts/apply_openpi_patches.sh

# The tokenizer is fetched at first use; bake it in so the container needs no internet.
RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(48); print('TOKENIZER_OK')"

# Arm B's checkpoint: params + its norm stats (assets/turning_on_radio). No optimizer state.
COPY checkpoint /opt/checkpoint

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD curl -sf http://127.0.0.1:8000/healthz || exit 1
CMD ["uv", "run", "--no-sync", "scripts/b1k/serve_b1k.py", \
     "--robot", "b1k/R1Pro", "--task", "b1k/turning_on_radio", "--repo-id", "turning_on_radio", \
     "--port", "8000", "--policy.config", "pi05_b1k_frozen_vlm_progress", "--policy.dir", "/opt/checkpoint"]
