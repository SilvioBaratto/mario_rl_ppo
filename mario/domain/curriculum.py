"""The reverse curriculum that moves a level's start point backward along its demonstration."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from mario.domain.routes import demo_start_frames, route_level


class ReverseCurriculum:
    """Move each level's demonstration start point backward as the agent succeeds from it.

    Start-state curriculum from a single demonstration (Salimans & Chen,
    2018): episodes first start near the end of a recorded route, past the
    obstacle, and the start moves one state earlier each time the agent
    reaches the route's end often enough from the current one. Each step back
    adds only a few frames of the demonstrated behaviour for the agent to
    discover itself, until it starts at the level start.

    Lives in the main process: workers hold their own (unpicklable) emulator
    states, report each finished demo episode as ``info['demo_outcome']`` of
    the following reset, and receive the current start frame through
    ``FrontierStart.set_demo_cursor``.

    Args:
        routes: Demonstration routes to run a curriculum on, at most one per level.
        warmup_frames: The workers' warm-up length (``demo_warmup_frames``);
            it decides which frames can be start frames.
        threshold: Fraction of the last ``window`` episodes from the current
            start that must reach the route's end before stepping back.
        window: Number of recent outcomes from the current start considered.
        start_frames: Optional 'W-S' level -> route frame to begin at instead of
            the route end, e.g. the last logged ``curriculum/<level>_start_frame``
            of a run being resumed (the position is not saved in checkpoints).
            A frame between start frames rounds down to the earlier one; one
            before the first start frame (the logged 0) marks the level done.

    Raises:
        ValueError: If ``start_frames`` names a level without a route.
    """

    def __init__(
        self,
        routes: Sequence[dict[str, Any]],
        warmup_frames: int,
        threshold: float = 0.5,
        window: int = 10,
        start_frames: dict[str, int] | None = None,
    ):
        self.window = window
        self.required = max(1, int(np.ceil(threshold * window)))
        self._frames = {
            route_level(route): demo_start_frames(route, warmup_frames=warmup_frames)
            for route in routes
        }
        self._index = {level: len(frames) - 1 for level, frames in self._frames.items()}
        for level, frame in (start_frames or {}).items():
            if level not in self._frames:
                raise ValueError(
                    f"no demo route for level {level} (routes: {', '.join(self._frames)})"
                )
            self._index[level] = sum(f <= frame for f in self._frames[level]) - 1
        self._outcomes = {level: deque(maxlen=window) for level in self._frames}

    def cursors(self) -> dict[str, int | None]:
        """Return each level's current start frame, or None once its curriculum is done."""
        return {
            level: self._frames[level][index] if index >= 0 else None
            for level, index in self._index.items()
        }

    def record(self, outcome: dict[str, Any]) -> tuple[int, int] | None:
        """Count a finished demo episode and step the start back if it has been mastered.

        Episodes that started from a later (rehearsal) state, or from a start
        the cursor has already left, are ignored.

        Args:
            outcome: A worker's ``demo_outcome`` dict with 'level', 'frame' and 'success'.

        Returns:
            (successes, episodes) counted at the old start when the level's
            start frame moved back, otherwise None.
        """
        level = outcome["level"]
        if level not in self._index or outcome["frame"] != self.cursors()[level]:
            return None
        outcomes = self._outcomes[level]
        outcomes.append(bool(outcome["success"]))
        if sum(outcomes) < self.required:
            return None
        counts = (sum(outcomes), len(outcomes))
        self._index[level] -= 1
        outcomes.clear()
        return counts
