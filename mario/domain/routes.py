"""Demonstration-route arithmetic: a route's level and the frames a restart may begin from."""

from __future__ import annotations

import re
from typing import Any

from mario.domain.config import DEMO_STRIDE, FRAME_SKIP
from mario.domain.levels import level_key


def route_level(route: dict[str, Any]) -> str:
    """Return the 'W-S' label of the level a demonstration route was recorded on."""
    match = re.fullmatch(r"SuperMarioBros-(\d)-(\d)-v\d+", route["env_id"])
    if not match:
        raise ValueError(f"Demo route env_id {route['env_id']!r} is not a single-stage id")
    return level_key(match.group(1), match.group(2))


def demo_warmup_frames(frame_stack: int) -> int:
    """Return how many route frames a demo restart replays before the agent takes over.

    VecFrameStack zero-fills the stack on every reset, and from a zero-padded
    stack the policy acts as it does at a level start (it usually lets go of
    A). Mid-jump that is fatal in SMB1: one frame without A locks in falling
    gravity for the rest of the jump. Replaying the route for the first
    ``frame_stack - 1`` agent steps hands control over with a full stack of
    real frames, as in a normal episode.

    Args:
        frame_stack: Frames per observation stack.

    Returns:
        Raw NES frames covering ``frame_stack - 1`` agent steps.
    """
    return (frame_stack - 1) * FRAME_SKIP


def demo_start_frames(
    route: dict[str, Any], stride: int = DEMO_STRIDE, warmup_frames: int = 0
) -> list[int]:
    """Return the route frames a reverse curriculum can start from, earliest first.

    Args:
        route: A route from ``load_demo_route``.
        stride: Frames between start states.
        warmup_frames: Route frames replayed before the agent takes over; a
            start frame needs that many frames of route before it.

    Returns:
        Every ``stride``-th frame (1-based from the route's reset) up to the
        route's last frame that leaves room for the warm-up.
    """
    return [
        frame for frame in range(stride, len(route["frames"]) + 1, stride) if frame > warmup_frames
    ]
