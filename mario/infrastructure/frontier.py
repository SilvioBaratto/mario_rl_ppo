"""Restarting training episodes from saved emulator states: frontier snapshots and demo routes."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper

from mario.domain.config import DEMO_STRIDE, SMB1_FLOAT_STATE
from mario.domain.levels import level_key
from mario.domain.routes import demo_start_frames, route_level
from mario.infrastructure.wrappers import StepReturn


def replay_demo_route(
    nes, route: dict[str, Any], stride: int = DEMO_STRIDE, warmup_frames: int = 0
) -> dict[int, tuple[Any, list[int], int]]:
    """Replay a demonstration route and collect the states a reverse curriculum can start from.

    Args:
        nes: The raw gym-super-mario-bros env of the route's level (``env.unwrapped``);
            it is reset and left at the route's end.
        route: A route from ``load_demo_route``.
        stride: Frames between start states.
        warmup_frames: Route frames replayed after the restore before the
            agent's first decision (see ``demo_warmup_frames``).

    Returns:
        {frame: (state, joypad bytes, x at the frame)} for every frame from
        ``demo_start_frames``, 1-based from the route's reset. Restoring the
        state and playing the bytes reproduces the frame exactly; the first
        byte redraws the screen, the remaining ``warmup_frames`` are forced
        over the agent's actions. Empty, with a warning, when a waypoint's x
        does not match.
    """
    starts = {
        frame - warmup_frames: frame for frame in demo_start_frames(route, stride, warmup_frames)
    }
    expected = {w["frame"]: w["x"] for w in route["waypoints"]}
    nes.reset()
    saved = {}
    states = {}
    for frame, byte in enumerate(route["frames"], start=1):
        if frame in starts:
            saved[starts[frame]] = (frame, nes.dump_state())
        nes.step(byte)
        x_pos = int(nes._x_position)
        if frame in saved:
            first, state = saved.pop(frame)
            states[frame] = (state, route["frames"][first - 1 : frame], x_pos)
        if frame in expected and x_pos != expected[frame]:
            warnings.warn(
                f"demo route {route['env_id']} diverged at frame {frame} "
                f"(x={x_pos}, expected {expected[frame]}); ignoring it",
                stacklevel=2,
            )
            return {}
    return states


class FrontierStart(Wrapper):
    """Restart a fraction of episodes from saved emulator states just behind the frontier.

    Episodes otherwise always begin at the level start, so an obstacle deep
    in a level is attempted once per long run-up the policy has already
    mastered. While Mario stands on solid ground, this wrapper snapshots the
    emulator once per ``bucket`` pixels of x. The frontier of a level is the
    median furthest x of its last ``window`` episodes; on a ``prob`` fraction
    of resets, a snapshot ``back_min``..``back_max`` pixels behind it is
    restored. The frontier moves forward by itself as the policy improves.
    Meant for training only: evaluation must start at the real level start.

    A level with a demonstration route instead runs a reverse curriculum on
    those restarts while ``ReverseCurriculum`` (main process) keeps a start
    frame set through ``set_demo_cursor``: the restart begins at that frame of
    the route, or with probability ``1 - demo_exact`` at a later one to keep
    the already-learned part rehearsed. The restore lands ``warmup_frames``
    before the start frame and forces the route's bytes until it, so the
    agent's first decision there sees a full frame stack. Route states are
    captured every ``DEMO_STRIDE`` frames, mid-jump included, by replaying the
    route once in-process (native snapshots cannot be pickled across workers). The next
    reset's info reports how a demo episode went as ``demo_outcome``; demo
    episodes are left out of the frontier statistics.
    """

    _NOOP_BYTE = 0

    def __init__(
        self,
        env: gym.Env,
        prob: float = 0.5,
        back_min: int = 50,
        back_max: int = 400,
        bucket: int = 32,
        window: int = 20,
        min_episodes: int = 5,
        demo_routes: Sequence[dict[str, Any]] | None = None,
        demo_exact: float = 0.75,
        warmup_frames: int = 0,
    ):
        super().__init__(env)
        self.prob = prob
        self.back_min = back_min
        self.back_max = back_max
        self.bucket = bucket
        self.min_episodes = min_episodes
        self.demo_exact = demo_exact
        self.warmup_frames = warmup_frames
        self._window = window
        self._snapshots: dict[tuple[Any, Any], dict[int, Any]] = {}
        self._episode_ends: dict[tuple[Any, Any], deque] = {}
        self._rng = np.random.default_rng()
        self._level = None
        self._episode_max_x = 0
        self._episode_steps = 0
        self._demo_routes = {route_level(route): route for route in demo_routes or []}
        self._demo_states: dict[str, dict[int, tuple[Any, list[int], int]]] = {}
        self._demo_cursor: dict[str, int | None] = {}
        self._episode_demo: dict[str, Any] | None = None
        self._forced: deque = deque()

    def set_demo_cursor(self, level: str, frame: int | None) -> None:
        """Set the route frame that demo restarts on ``level`` begin from.

        Args:
            level: 'W-S' label; calls for levels this worker doesn't play are kept but unused.
            frame: A frame from ``demo_start_frames``, or None to stop demo restarts.
        """
        self._demo_cursor[level] = frame

    def _demo_states_for(self, level: str) -> dict[int, tuple[Any, list[int], int]]:
        if level not in self._demo_states:
            route = self._demo_routes.get(level)
            self._demo_states[level] = (
                replay_demo_route(self.env.unwrapped, route, warmup_frames=self.warmup_frames)
                if route
                else {}
            )
        return self._demo_states[level]

    def frontier(self, level) -> float | None:
        """Return the median furthest x of recent episodes on ``level``, None while warming up."""
        ends = self._episode_ends.get(level)
        if not ends or len(ends) < self.min_episodes:
            return None
        return float(np.median(ends))

    def _pick_demo_frame(self, level: str) -> int | None:
        cursor = self._demo_cursor.get(level)
        if cursor is None:
            return None
        states = self._demo_states_for(level)
        if cursor not in states:
            return None
        later = [frame for frame in states if frame > cursor]
        if not later or self._rng.random() < self.demo_exact:
            return cursor
        return later[int(self._rng.integers(len(later)))]

    def _pick_start(self, level: str, level_tuple):
        """Return (state, joypad bytes to replay, demo frame or None), or None to start normally."""
        demo_frame = self._pick_demo_frame(level)
        if demo_frame is not None:
            state, joypad, _ = self._demo_states[level][demo_frame]
            return state, joypad, demo_frame
        snapshot = self._pick_snapshot(level_tuple)
        if snapshot is None:
            return None
        return snapshot, [self._NOOP_BYTE], None

    def _pick_snapshot(self, level):
        frontier = self.frontier(level)
        if frontier is None:
            return None
        low, high = frontier - self.back_max, frontier - self.back_min
        candidates = [
            snap
            for key, snap in self._snapshots.get(level, {}).items()
            if low <= key * self.bucket <= high
        ]
        if not candidates:
            return None
        return candidates[int(self._rng.integers(len(candidates)))]

    def _restore(self, state, byte: int):
        """Load ``state`` and play one frame with ``byte``.

        Returns:
            (obs, info), or None when that frame ended the episode.
        """
        nes = self.env.unwrapped
        nes.load_state(state)
        # gym-super-mario-bros keeps reward trackers (last time, max x,
        # score, coins) in Python; resync them with the restored RAM or the
        # first step pays e.g. a huge time penalty.
        nes._did_reset()
        # The frame redraws the screen buffer so obs matches the state. Demo
        # states replay the route's own byte: releasing A for even one frame
        # mid-ascent makes SMB1 switch to falling gravity for the rest of the jump.
        obs, _, terminated, truncated, info = nes.step(byte)
        if terminated or truncated:
            return None
        return obs, info

    def _finish_episode(self) -> dict[str, Any] | None:
        """Book the episode that just ended; return its demo outcome, if it was a demo episode."""
        # Skip resets with no steps since the last one (the construction-time
        # seeding reset, a VecEnv's first reset): they'd record the start x as
        # an episode end and drag the frontier back toward the level start.
        if self._level is None or self._episode_steps == 0:
            return None
        if self._episode_demo is None:
            self._episode_ends.setdefault(self._level, deque(maxlen=self._window)).append(
                self._episode_max_x
            )
            return None
        demo = self._episode_demo
        return {
            "level": demo["level"],
            "frame": demo["frame"],
            "x": demo["x"],
            "success": self._episode_max_x >= demo["goal_x"],
        }

    def reset(self, *, seed=None, options=None):
        """Reset normally, then maybe jump to a demo state or a snapshot behind the frontier.

        Args:
            seed: Reseeds both the env and the restart RNG when given.
            options: Forwarded to the wrapped env's reset.

        Returns:
            The (obs, info) tuple of the state the episode actually starts
            from. ``info['frontier_start']`` tells whether a state was
            restored and ``info['demo_start']`` whether it came from a demo
            route; ``info['demo_outcome']`` (level, start frame, start x,
            success) is present when the previous episode was a demo episode.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        outcome = self._finish_episode()

        obs, info = self.env.reset(seed=seed, options=options)
        level_tuple = (info.get("world"), info.get("stage"))
        level = level_key(*level_tuple)
        if level in self._demo_routes and level not in self._demo_states:
            # The replay drives the emulator from its own reset, so redo the
            # episode reset afterwards.
            self._demo_states_for(level)
            obs, info = self.env.reset(options=options)

        self._episode_demo = None
        self._forced.clear()
        start = self._pick_start(level, level_tuple) if self._rng.random() < self.prob else None
        restored = None
        if start is not None:
            state, joypad, demo_frame = start
            restored = self._restore(state, joypad[0])
            if restored is None:
                obs, info = self.env.reset(options=options)
            else:
                obs, info = restored
                self._forced.extend(joypad[1:])
                if demo_frame is not None:
                    self._episode_demo = {
                        "level": level,
                        "frame": demo_frame,
                        "x": self._demo_states[level][demo_frame][2],
                        "goal_x": self._demo_routes[level]["waypoints"][-1]["x"],
                    }

        self._level = (info.get("world"), info.get("stage"))
        self._episode_max_x = info.get("x_pos", 0)
        self._episode_steps = 0
        info["frontier_start"] = restored is not None
        info["demo_start"] = self._episode_demo is not None
        if outcome is not None:
            info["demo_outcome"] = outcome
        return obs, info

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        if self._forced:
            obs, reward, terminated, truncated, info = self.env.unwrapped.step(
                self._forced.popleft()
            )
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        level = (info.get("world"), info.get("stage"))
        x_pos = info.get("x_pos", 0)
        self._episode_steps += 1
        if level == self._level:
            self._episode_max_x = max(self._episode_max_x, x_pos)
        nes = self.env.unwrapped
        if (
            not (terminated or truncated)
            and not info.get("flag_get")
            and nes.ram[SMB1_FLOAT_STATE] == 0
            and not nes._is_dying
        ):
            self._snapshots.setdefault(level, {})[x_pos // self.bucket] = nes.dump_state()
        return obs, reward, terminated, truncated, info
