# Policy server image for challenge submission.
#
# Build this in WEEK 1, not week 7. It is a required submission artifact and it is how
# you prove the 24GB constraint holds before it matters.
#
# Constraint: the submitted model must run on a SINGLE 24GB VRAM GPU.
# Final evaluation runs on RTX 3090 / A5000 / TitanRTX.
#
# The organizers run OmniGibson OUTSIDE this container and connect over the websocket
# protocol -- so the image serves the policy only. Do not put the simulator in here.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TODO: install the policy stack (openpi behavior fork, or GR00T) and its deps.
# Keep the simulator out -- it runs on the host side.
COPY requirements-tools.txt .
RUN pip3 install --no-cache-dir -r requirements-tools.txt

COPY policy/ ./policy/
COPY configs/ ./configs/

# TODO: COPY the checkpoint, or mount it at runtime with -v.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/healthz || exit 1

CMD ["python3", "-m", "policy.server", "--host", "0.0.0.0", "--port", "8000"]
