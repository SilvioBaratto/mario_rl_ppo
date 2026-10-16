# Super Mario Bros RL with PPO
# Single-stage image; torch comes from the CPU-only wheel index to keep it small

FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for OpenCV
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Required for video encoding
    ffmpeg \
    # Required for SDL (headless mode)
    libsdl2-dev \
    libsdl2-image-dev \
    # Build dependencies
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mario
WORKDIR /app
# WORKDIR is created as root; evaluate and record write their outputs into it as `mario`.
RUN chown mario:mario /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies. torch/torchvision come from the CPU-only index
# first: the PyPI Linux wheels pull the multi-GB CUDA toolkit, and the
# containers train with --device cpu.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 torchvision==0.29.1 && \
    pip install -r requirements.txt

# Copy application code, then register the package and its `mario` command. The dependencies are
# already in the layer above, so --no-deps keeps this step instant.
COPY --chown=mario:mario . .
RUN pip install --no-deps -e .

# Set SDL to headless mode
ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# Switch to non-root user
USER mario

# Default command: show help
CMD ["mario", "--help"]

# Example usage:
# Build:
#   docker build -t mario-rl-ppo .
#
# Train (10M timesteps):
#   docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
#     mario train --total-timesteps 10000000 --device cpu --exp-name docker_run
#
# Evaluate:
#   docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
#     mario evaluate --model-path runs/docker_run/checkpoints/best_model --n-episodes 10
#
# Record video:
#   docker run --rm -v $(pwd)/runs:/app/runs -v $(pwd)/videos:/app/videos mario-rl-ppo \
#     mario record --model-path runs/docker_run/checkpoints/best_model
