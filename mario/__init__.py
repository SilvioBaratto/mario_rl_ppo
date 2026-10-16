"""Super Mario Bros reinforcement learning: a PPO agent that learns from pixels."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mario-rl-ppo")
except PackageNotFoundError:  # a checkout that was never installed
    __version__ = "0+unknown"
