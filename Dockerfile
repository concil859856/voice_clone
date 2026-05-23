# syntax=docker/dockerfile:1.7
#
# voice_clone — Qwen3-TTS voice-cloning HTTP server for the Vocence
# /studio/ops fleet manager. Runs on NVIDIA RTX 4090 (24 GB).
#
# Build:
#   docker build -t docker.io/<ns>/voice_clone:latest .
# Run (on rented box):
#   docker run -d --gpus all --restart=unless-stopped -p 8113:8113 \
#     -e VOICE_CLONE_API_KEY=<key> \
#     -v hf_cache:/cache/hf \
#     docker.io/<ns>/voice_clone:latest

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HOST=0.0.0.0 \
    PORT=8113

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the slow torch/qwen-tts install layer caches
# across iterations of the Python source.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the actual server.
COPY main.py ./
COPY bt_voices/ ./bt_voices/
COPY voice_clone.py ./
# (clone/ package is optional and not used by main.py; left out to keep
# the image lean.)

# Build-time import smoke test — verifies the full import chain the
# entrypoint exercises is internally consistent. Cheap (no GPU, ~5 s)
# but catches torch/torchvision ABI mismatches, missing modules, and
# similar import-time failures BEFORE the image reaches Docker Hub.
# Runs on the CPU-only GHA runner. A broken image that imports cleanly
# at build time is a contradiction.
RUN python3 -c "import torch; print('torch', torch.__version__)" \
    && python3 -c "import main; print('main import OK')"

# Persistent HuggingFace cache. Mount a host volume here so the ~3.4 GB
# Qwen3-TTS weights aren't re-downloaded on every container recreation.
VOLUME /cache/hf

EXPOSE 8113

# 120 s start-period covers the lazy first-request model load (~30-60 s
# on cold cache, ~10 s warm). HEALTHCHECK uses /health (always open) so
# Docker can mark unhealthy without needing the bearer token.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8113}/health || exit 1

CMD ["python3", "main.py"]
