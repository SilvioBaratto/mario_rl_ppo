"""Helpers the commands share: headless SDL setup and the choice of torch device."""

from __future__ import annotations

import os


def configure_sdl(video: bool = False) -> None:
    """Point SDL at its dummy drivers so the emulator runs without a display or a sound card.

    This has to run before pygame or the emulator initialises, which is why every command calls
    it before it imports the heavy modules.

    Args:
        video: Keep the real video driver, as ``evaluate --render`` and ``evaluate
            --record-video`` need; only the audio driver is then replaced.
    """
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    if not video:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


def select_device(device: str) -> str:
    """Resolve 'auto' to the fastest available device (cuda, then mps, then cpu).

    Args:
        device: A torch device name, or 'auto'.

    Returns:
        ``device`` unchanged unless it was 'auto'.
    """
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
