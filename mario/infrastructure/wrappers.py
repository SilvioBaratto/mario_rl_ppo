"""Gymnasium wrappers that turn the raw NES environment into the training environment."""

from __future__ import annotations

from typing import Any, SupportsFloat

import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper

from mario.domain.config import FRAME_SKIP, HUD_ROWS, NOOP_MAX, OBS_SIZE, STALL_STEPS

StepReturn = tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]


NOOP_ACTION = SIMPLE_MOVEMENT.index(["NOOP"])


class SkipFrame(Wrapper):
    """Repeat each action for ``skip`` frames, ending the step early when the level changes.

    In the whole game a level starts mid-step, and the rest of the step would
    replay an action chosen while the previous level was on screen: entering
    1-3 that way made Mario jump at x=40 and meet the tree-top Koopa at a bad
    moment in every greedy run, although single-stage 1-3 (where the first
    input is chosen from the level's own first frame) was cleared every time.
    Single-stage episodes end at the flag, so they never hit the early exit.

    From the frame the flag is touched until the next level starts, every
    frame gets no buttons. The game ignores input during the end sequence
    except on the frame the next level begins, which already moves Mario:
    the agent's last 1-1 action (run + jump) used to start 1-2 with Mario
    at speed 2 and A held (SMB1 only jumps on a fresh press), a state no
    demonstration or single-stage reset has. The step that ends early reports
    ``info['level_start']``, which ``LevelStartFrameStack`` uses to clear the
    frame history.
    """

    def __init__(self, env: gym.Env, skip: int = FRAME_SKIP):
        super().__init__(env)
        self._skip = skip
        self._level = None
        self._ending = False

    def reset(self, **kwargs):
        """Reset the env and remember which level the episode starts on."""
        obs, info = self.env.reset(**kwargs)
        self._level = (info.get("world"), info.get("stage"))
        self._ending = False
        return obs, info

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        total_reward = 0.0
        obs: Any = None
        terminated = truncated = False
        info: dict[str, Any] = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(
                NOOP_ACTION if self._ending else action
            )
            total_reward += float(reward)
            self._ending = self._ending or bool(info.get("flag_get"))
            level = (info.get("world"), info.get("stage"))
            changed = level != self._level
            self._level = level
            if changed:
                self._ending = False
                if not (terminated or truncated):
                    info["level_start"] = True
            if terminated or truncated or changed:
                break
        return obs, total_reward, terminated, truncated, info


class StallTruncation(Wrapper):
    """Truncate the episode once Mario goes ``max_steps`` agent steps without a new furthest x.

    A greedy policy stuck against a pipe otherwise runs to the 2000-step
    TimeLimit; late in a run those episodes made in-training evaluation cost
    more than the training itself. Truncation rather than termination keeps
    PPO bootstrapping the value of the cut-off state.
    """

    def __init__(self, env: gym.Env, max_steps: int = STALL_STEPS):
        super().__init__(env)
        self.max_steps = max_steps
        self._segment = None
        self._best_x = 0
        self._since_progress = 0

    def reset(self, **kwargs):
        """Reset the env and the progress tracker.

        Args:
            **kwargs: Forwarded to the wrapped env's reset.

        Returns:
            The wrapped env's (obs, info) tuple.
        """
        obs, info = self.env.reset(**kwargs)
        self._start_segment(info)
        return obs, info

    def _start_segment(self, info):
        # x_pos restarts on a new level or a respawn, so progress is measured
        # per (world, stage, life) segment.
        self._segment = (info.get("world"), info.get("stage"), info.get("life"))
        self._best_x = info.get("x_pos", 0)
        self._since_progress = 0

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get("x_pos", 0)
        if (info.get("world"), info.get("stage"), info.get("life")) != self._segment:
            self._start_segment(info)
        elif x_pos > self._best_x:
            self._best_x = x_pos
            self._since_progress = 0
        else:
            self._since_progress += 1
        if self._since_progress >= self.max_steps and not terminated:
            truncated = True
            info["stalled"] = True
        return obs, reward, terminated, truncated, info


class NoopReset(Wrapper):
    """Run a random number (0..noop_max) of NOOP frames after every reset.

    The NES emulator is deterministic, so without this every episode starts
    from the identical state and a deterministic policy replays the same
    trajectory, which makes multi-episode evaluation meaningless.
    """

    def __init__(self, env: gym.Env, noop_max: int = NOOP_MAX):
        super().__init__(env)
        self.noop_max = noop_max
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        """Reset the env, then idle for a random number of frames.

        Args:
            seed: Reseeds both the env and the NOOP-count RNG when given.
            options: Forwarded to the wrapped env's reset.

        Returns:
            The (obs, info) tuple after the NOOP frames.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(int(self._rng.integers(0, self.noop_max + 1))):
            obs, _, terminated, truncated, info = self.env.step(NOOP_ACTION)
            if terminated or truncated:
                obs, info = self.env.reset(options=options)
        return obs, info


class GrayResizeObservation(ObservationWrapper):
    """Convert RGB frames to a square grayscale uint8 image with the status bar blanked.

    cv2 does conversion and resize in ~0.3 ms; gymnasium's GrayscaleObservation
    alone takes ~0.56 ms because it computes the weighted sum in float64 NumPy.

    The top ``HUD_ROWS`` raw rows (score, coins, world, time) are set to black.
    Policies trained with the status bar visible keyed on its digits (freezing
    just the score and coin display cut greedy 1-1 clears from 17 to 6 of 31
    NOOP offsets), and in the whole game the score and coins carried over from
    earlier levels never occur on the single-stage levels they train on.
    """

    def __init__(self, env: gym.Env, size: int = OBS_SIZE):
        super().__init__(env)
        self._size = size
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return the frame as a (size, size, 1) grayscale uint8 array.

        Args:
            observation: RGB frame of shape (H, W, 3).

        Returns:
            The converted frame.
        """
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        gray[:HUD_ROWS] = 0
        return cv2.resize(gray, (self._size, self._size), interpolation=cv2.INTER_AREA)[:, :, None]


class CustomReward(Wrapper):
    """Custom reward shaping for Super Mario Bros.

    Improvements over default rewards:
    - Adds score-based reward bonus (game score divided by 40)
    - +50 bonus for reaching the flag
    - -50 penalty for losing a life
    - All rewards scaled by 0.1 for more stable training

    Flag and death events come from ``info`` rather than from episode
    termination, so they fire on every occurrence in ``SuperMarioBros-v0``
    too, where one episode spans several lives and levels.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.curr_score = 0
        self.current_x = 40
        self._life = None
        self._level = None
        self._flag = False

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        reward = float(raw_reward)

        # Add score-based reward
        if "score" in info:
            reward += (info["score"] - self.curr_score) / 40.0
            self.curr_score = info["score"]

        flag = bool(info.get("flag_get", False))
        life = info.get("life")
        level = (info.get("world"), info.get("stage"))
        if flag and not self._flag:
            reward += 50
        # The final death wraps life 0 -> 255 instead of decreasing, but it also
        # terminates the episode, so the terminated check covers it.
        lost_life = self._life is not None and life is not None and life < self._life
        if lost_life or (terminated and not flag):
            reward -= 50
        if lost_life or level != self._level:
            # Mario respawns or a new level starts far to the left; the 7-4
            # backtrack check below would otherwise read that jump as a wrong path.
            self.current_x = info.get("x_pos", self.current_x)
        self._flag, self._life, self._level = flag, life, level

        # Special handling for maze levels (7-4 and 4-4) to penalize wrong paths
        world, stage = level
        if world == 7 and stage == 4:
            x_pos = info.get("x_pos", 0)
            y_pos = info.get("y_pos", 0)
            if (
                (506 <= x_pos <= 832 and y_pos > 127)
                or (832 < x_pos <= 1064 and y_pos < 80)
                or (1113 < x_pos <= 1464 and y_pos < 191)
                or (1579 < x_pos <= 1943 and y_pos < 191)
                or (1946 < x_pos <= 1964 and y_pos >= 191)
                or (1984 < x_pos <= 2060 and (y_pos >= 191 or y_pos < 127))
                or (2114 < x_pos < 2440 and y_pos < 191)
                or x_pos < self.current_x - 500
            ):
                reward -= 50
                terminated = True

        if world == 4 and stage == 4:
            x_pos = info.get("x_pos", 0)
            y_pos = info.get("y_pos", 0)
            if (x_pos <= 1500 and y_pos < 127) or (1588 <= x_pos < 2380 and y_pos >= 127):
                reward -= 50
                terminated = True

        self.current_x = info.get("x_pos", self.current_x)

        # Scale all rewards by 0.1 for stability
        return obs, reward / 10.0, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the env and the event trackers that shape the reward.

        Args:
            **kwargs: Forwarded to the wrapped env's reset.

        Returns:
            The wrapped env's (obs, info) tuple.
        """
        obs, info = self.env.reset(**kwargs)
        self.curr_score = info.get("score", 0)
        self.current_x = info.get("x_pos", 40)
        self._life = info.get("life")
        self._level = (info.get("world"), info.get("stage"))
        self._flag = bool(info.get("flag_get", False))
        return obs, info
