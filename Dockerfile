# Super Mario Bros RL with PPO
# Multi-stage build for optimized image size

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for OpenCV
    libgl1-mesa-glx \
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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=mario:mario . .

# Set SDL to headless mode
ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# Switch to non-root user
USER mario

# Default command: show help
CMD ["python", "train.py", "--help"]

# Example usage:
# Build:
#   docker build -t mario-rl-ppo .
#
# Train (10M timesteps):
#   docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
#     python train.py --total-timesteps 10000000 --device cpu --exp-name docker_run
#
# Evaluate:
#   docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
#     python evaluate.py --model-path runs/docker_run/checkpoints/best_model --n-episodes 10
#
# Record video:
#   docker run --rm -v $(pwd)/runs:/app/runs -v $(pwd)/videos:/app/videos mario-rl-ppo \
#     python play_and_record.py --model-path runs/docker_run/checkpoints/best_model
