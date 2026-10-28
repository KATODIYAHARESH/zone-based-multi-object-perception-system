# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# ZBMOPS perception engine. CPU image by default. CUDA build:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t zbmops:gpu .
#   docker run --gpus all -p 8765:8765 -v $PWD/data:/app/data zbmops:gpu
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/data/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install "opencv-python-headless>=4.9.0"

COPY . .
RUN useradd --create-home --uid 10001 zbmops \
    && mkdir -p /app/data \
    && chown -R zbmops:zbmops /app
USER zbmops

EXPOSE 8765 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8765/healthz || exit 1

CMD ["python", "main.py", "run", "--headless"]
