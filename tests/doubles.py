"""Hand-written stand-ins for the emulator, so wrapper logic is tested without a ROM."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Row = tuple[float, bool, bool, dict[str, Any]]


def level_info(world: int = 1, stage: int = 1, **fields: Any) -> dict[str, Any]:
    """Return a gym-super-mario-bros style info dict, overridden by ``fields``."""
    info: dict[str, Any] = {
        "world": world,
        "stage": stage,
        "life": 2,
        "x_pos": 40,
        "y_pos": 79,
        "score": 0,
        "flag_get": False,
    }
    info.update(fields)
    return info


class ScriptedEnv(gym.Env):
    """A gymnasium env that answers each ``step`` with the next prepared row.

    Once the rows run out the last one repeats. ``actions`` records what it was asked to do,
    ``reset_seeds`` the seed of every reset.
    """

    observation_space = spaces.Box(0, 255, (4, 4, 3), np.uint8)
    action_space = spaces.Discrete(7)

    def __init__(self, start_info: dict[str, Any], rows: list[Row]) -> None:
        super().__init__()
        self.start_info = start_info
        self.rows = rows
        self.actions: list[int] = []
        self.reset_seeds: list[int | None] = []
        self._cursor = 0

    def _frame(self) -> np.ndarray:
        return np.zeros((4, 4, 3), np.uint8)

    def reset(self, *, seed=None, options=None):
        """Rewind the script and return the start info."""
        self.reset_seeds.append(seed)
        self._cursor = 0
        return self._frame(), dict(self.start_info)

    def step(self, action):
        """Return the next scripted row."""
        self.actions.append(int(action))
        reward, terminated, truncated, info = self.rows[min(self._cursor, len(self.rows) - 1)]
        self._cursor += 1
        return self._frame(), reward, terminated, truncated, dict(info)


class FakeNes(gym.Env):
    """A one-dimensional stand-in for the NES env: a joypad byte moves Mario that many pixels."""

    observation_space = spaces.Box(0, 255, (4, 4, 3), np.uint8)
    action_space = spaces.Discrete(256)

    def __init__(self, start_x: int = 40) -> None:
        super().__init__()
        self.start_x = start_x
        self.x = start_x
        self.ram = np.zeros(2048, np.uint8)
        self._is_dying = False
        self.received: list[int] = []
        self.restored: list[int] = []
        self.synced = 0

    @property
    def _x_position(self) -> int:
        return self.x

    def _info(self) -> dict[str, Any]:
        return level_info(x_pos=self.x)

    def reset(self, *, seed=None, options=None):
        """Put Mario back at the start."""
        self.x = self.start_x
        return np.zeros((4, 4, 3), np.uint8), self._info()

    def step(self, action):
        """Move Mario by ``action`` pixels and remember the byte."""
        self.received.append(int(action))
        self.x += int(action)
        return np.zeros((4, 4, 3), np.uint8), 0.0, False, False, self._info()

    def dump_state(self) -> int:
        """Return a snapshot of the position."""
        return self.x

    def load_state(self, state: int) -> None:
        """Restore a snapshot taken by ``dump_state``."""
        self.restored.append(state)
        self.x = state

    def _did_reset(self) -> None:
        self.synced += 1


def route(frames: list[int], goal_x: int, env_id: str = "SuperMarioBros-1-1-v0") -> dict[str, Any]:
    """Return a minimal demonstration route over ``frames`` that ends at ``goal_x``."""
    return {
        "env_id": env_id,
        "frames": frames,
        "waypoints": [{"frame": len(frames), "x": goal_x}],
    }
