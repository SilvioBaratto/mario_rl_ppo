"""
Environment factory for creating Super Mario Bros environments
"""

import json
import multiprocessing
import re
import signal
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper, ObservationWrapper
from gymnasium.wrappers import TimeLimit
import numpy as np
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

# Gymnasium step return: (obs, reward, terminated, truncated, info)
StepReturn = Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]

DEFAULT_ENV_ID = 'SuperMarioBros-1-1-v0'
NOOP_ACTION = SIMPLE_MOVEMENT.index(['NOOP'])

EnvIds = Union[str, Sequence[str]]

DEMO_ROUTES_DIR = Path(__file__).with_name('demo_routes')


def load_demo_route(env_id: str) -> Optional[Dict[str, Any]]:
    """Return the recorded demonstration route for a single-stage env id, if one exists.

    Routes live in ``demo_routes/<world>-<stage>.json``: frame-level joypad
    bytes from a fresh, NOOP-free reset plus waypoints (frame index and the
    x position the replay must reach there).

    Args:
        env_id: e.g. 'SuperMarioBros-1-3-v0'.

    Returns:
        The parsed route, or None when the id is not single-stage or has no route.
    """
    match = re.fullmatch(r'SuperMarioBros-(\d)-(\d)-v\d+', env_id)
    if not match:
        return None
    path = DEMO_ROUTES_DIR / f'{match.group(1)}-{match.group(2)}.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def parse_levels(spec: str) -> List[str]:
    """Convert a level list such as '1-1,1-2,1-3' into single-stage env ids.

    Args:
        spec: Comma-separated WORLD-STAGE pairs, world 1-8 and stage 1-4.

    Returns:
        Env ids in the given order, e.g. ['SuperMarioBros-1-1-v0', ...].

    Raises:
        ValueError: If the list is empty or an entry is malformed or out of range.
    """
    env_ids = []
    for item in (part.strip() for part in spec.split(',')):
        match = re.fullmatch(r'([1-8])-([1-4])', item)
        if not match:
            raise ValueError(f"Invalid level {item!r}; expected WORLD-STAGE like 1-2 (world 1-8, stage 1-4)")
        env_ids.append(f'SuperMarioBros-{match.group(1)}-{match.group(2)}-v0')
    if not env_ids:
        raise ValueError("Level list is empty")
    return env_ids


def assign_levels(env_ids: EnvIds, num_envs: int) -> List[str]:
    """Return the env id each of ``num_envs`` workers plays, round-robin.

    Args:
        env_ids: One env id, or several to train on together.
        num_envs: Number of parallel workers.

    Returns:
        A list of length ``num_envs``; worker i plays ``env_ids[i % len(env_ids)]``.
        When ``num_envs`` is not a multiple of the level count, the first
        levels get one extra worker.
    """
    ids = [env_ids] if isinstance(env_ids, str) else list(env_ids)
    return [ids[rank % len(ids)] for rank in range(num_envs)]


class SkipFrame(Wrapper):
    """Skip frames - execute action every skip frames (Kaggle implementation)"""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def episode_end_reason(info: Dict[str, Any]) -> str:
    """Describe why an episode that did not reach the flag ended.

    Args:
        info: The final step's info dict, as returned by the VecEnv on ``done``.

    Returns:
        'Stalled (no progress)' for the stall cutoff, 'Time limit' for the
        TimeLimit wrapper, otherwise 'Died'.
    """
    if info.get('stalled'):
        return "Stalled (no progress)"
    if info.get('TimeLimit.truncated'):
        return "Time limit"
    return "Died"


class StallTruncation(Wrapper):
    """Truncate the episode once Mario goes ``max_steps`` agent steps without a new furthest x.

    A greedy policy stuck against a pipe otherwise runs to the 2000-step
    TimeLimit; late in a run those episodes made in-training evaluation cost
    more than the training itself. Truncation rather than termination keeps
    PPO bootstrapping the value of the cut-off state.
    """
    def __init__(self, env: gym.Env, max_steps: int = 150):
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
        self._segment = (info.get('world'), info.get('stage'), info.get('life'))
        self._best_x = info.get('x_pos', 0)
        self._since_progress = 0

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get('x_pos', 0)
        if (info.get('world'), info.get('stage'), info.get('life')) != self._segment:
            self._start_segment(info)
        elif x_pos > self._best_x:
            self._best_x = x_pos
            self._since_progress = 0
        else:
            self._since_progress += 1
        if self._since_progress >= self.max_steps and not terminated:
            truncated = True
            info['stalled'] = True
        return obs, reward, terminated, truncated, info


class NoopReset(Wrapper):
    """Run a random number (0..noop_max) of NOOP frames after every reset.

    The NES emulator is deterministic, so without this every episode starts
    from the identical state and a deterministic policy replays the same
    trajectory, which makes multi-episode evaluation meaningless.
    """
    def __init__(self, env: gym.Env, noop_max: int = 30):
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

    Optional demonstration routes add hand-verified states beyond an obstacle
    the policy cannot explore past on its own (start-state curriculum from a
    single demonstration). Each route is replayed once in-process (native
    snapshots cannot be pickled across workers) and its waypoints are offered
    for half of the restarts while they lie ahead of the frontier; once the
    agent's own episodes reach them they are no longer used. Episodes started
    from a demonstration state are left out of the frontier statistics.
    """
    # SMB1 RAM: player "float state", 0 while standing on solid ground. Only
    # grounded states are saved so a restart never begins in a doomed fall.
    _FLOAT_STATE_ADDR = 0x001D

    def __init__(self, env: gym.Env, prob: float = 0.5, back_min: int = 50, back_max: int = 400,
                 bucket: int = 32, window: int = 20, min_episodes: int = 5,
                 demo_routes: Optional[Sequence[Dict[str, Any]]] = None):
        super().__init__(env)
        self.prob = prob
        self.back_min = back_min
        self.back_max = back_max
        self.bucket = bucket
        self.min_episodes = min_episodes
        self._window = window
        self._snapshots: Dict[Tuple[Any, Any], Dict[int, Any]] = {}
        self._episode_ends: Dict[Tuple[Any, Any], deque] = {}
        self._rng = np.random.default_rng()
        self._level = None
        self._episode_max_x = 0
        self._episode_steps = 0
        self._episode_from_demo = False
        self._demo_routes: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for route in demo_routes or []:
            match = re.fullmatch(r'SuperMarioBros-(\d)-(\d)-v\d+', route['env_id'])
            self._demo_routes[(int(match.group(1)), int(match.group(2)))] = route
        self._demo_states: Dict[Tuple[Any, Any], List[Tuple[int, Any]]] = {}

    def _demo_states_for(self, level) -> List[Tuple[int, Any]]:
        if level not in self._demo_states:
            route = self._demo_routes.get(level)
            self._demo_states[level] = self._replay(route) if route else []
        return self._demo_states[level]

    def _replay(self, route) -> List[Tuple[int, Any]]:
        nes = self.env.unwrapped
        nes.reset()
        expected = {w['frame']: w['x'] for w in route['waypoints']}
        states = []
        for frame, byte in enumerate(route['frames'], start=1):
            nes.step(byte)
            if frame in expected:
                x_pos = int(nes._x_position)
                if x_pos != expected[frame]:
                    warnings.warn(f"demo route {route['env_id']} diverged at frame {frame} "
                                  f"(x={x_pos}, expected {expected[frame]}); ignoring it")
                    return []
                states.append((x_pos, nes.dump_state()))
        return states

    def frontier(self, level) -> Optional[float]:
        """Return the median furthest x of recent episodes on ``level``, or None while warming up."""
        ends = self._episode_ends.get(level)
        if not ends or len(ends) < self.min_episodes:
            return None
        return float(np.median(ends))

    def _pick_snapshot(self, level):
        """Return (snapshot or None, whether it came from a demonstration route)."""
        if self._rng.random() >= self.prob:
            return None, False
        frontier = self.frontier(level)
        ahead = [snap for x_pos, snap in self._demo_states_for(level)
                 if frontier is None or x_pos > frontier - self.back_min]
        if ahead and self._rng.random() < 0.5:
            return ahead[int(self._rng.integers(len(ahead)))], True
        if frontier is None:
            return None, False
        low, high = frontier - self.back_max, frontier - self.back_min
        candidates = [snap for key, snap in self._snapshots.get(level, {}).items()
                      if low <= key * self.bucket <= high]
        if not candidates:
            return None, False
        return candidates[int(self._rng.integers(len(candidates)))], False

    def reset(self, *, seed=None, options=None):
        """Reset normally, then maybe jump to a snapshot behind the frontier.

        Args:
            seed: Reseeds both the env and the restart RNG when given.
            options: Forwarded to the wrapped env's reset.

        Returns:
            The (obs, info) tuple of the state the episode actually starts
            from; ``info['frontier_start']`` tells whether it was restored.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Skip resets with no steps since the last one (the construction-time
        # seeding reset, a VecEnv's first reset): they'd record the start x as
        # an episode end and drag the frontier back toward the level start.
        if self._level is not None and self._episode_steps > 0 and not self._episode_from_demo:
            self._episode_ends.setdefault(self._level, deque(maxlen=self._window)).append(self._episode_max_x)

        obs, info = self.env.reset(seed=seed, options=options)
        level = (info.get('world'), info.get('stage'))
        if level in self._demo_routes and level not in self._demo_states:
            # The replay drives the emulator from its own reset, so redo the
            # episode reset afterwards.
            self._demo_states_for(level)
            obs, info = self.env.reset(options=options)
        snapshot, from_demo = self._pick_snapshot(level)
        restored = False
        if snapshot is not None:
            nes = self.env.unwrapped
            nes.load_state(snapshot)
            # gym-super-mario-bros keeps reward trackers (last time, max x,
            # score, coins) in Python; resync them with the restored RAM or the
            # first step pays e.g. a huge time penalty.
            nes._did_reset()
            # One idle frame redraws the screen buffer so obs matches the state.
            obs, _, terminated, truncated, info = self.env.step(NOOP_ACTION)
            if terminated or truncated:
                obs, info = self.env.reset(options=options)
            else:
                restored = True

        self._level = (info.get('world'), info.get('stage'))
        self._episode_max_x = info.get('x_pos', 0)
        self._episode_steps = 0
        self._episode_from_demo = restored and from_demo
        info['frontier_start'] = restored
        info['demo_start'] = self._episode_from_demo
        return obs, info

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        level = (info.get('world'), info.get('stage'))
        x_pos = info.get('x_pos', 0)
        self._episode_steps += 1
        if level == self._level:
            self._episode_max_x = max(self._episode_max_x, x_pos)
        nes = self.env.unwrapped
        if (not (terminated or truncated) and not info.get('flag_get')
                and nes.ram[self._FLOAT_STATE_ADDR] == 0 and not nes._is_dying):
            self._snapshots.setdefault(level, {})[x_pos // self.bucket] = nes.dump_state()
        return obs, reward, terminated, truncated, info


class GrayResizeObservation(ObservationWrapper):
    """Convert RGB frames to a square grayscale uint8 image.

    cv2 does conversion and resize in ~0.3 ms; gymnasium's GrayscaleObservation
    alone takes ~0.56 ms because it computes the weighted sum in float64 NumPy.
    """
    def __init__(self, env: gym.Env, size: int = 84):
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
        if 'score' in info:
            reward += (info['score'] - self.curr_score) / 40.0
            self.curr_score = info['score']

        flag = bool(info.get('flag_get', False))
        life = info.get('life')
        level = (info.get('world'), info.get('stage'))
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
            self.current_x = info.get('x_pos', self.current_x)
        self._flag, self._life, self._level = flag, life, level

        # Special handling for maze levels (7-4 and 4-4) to penalize wrong paths
        world, stage = level
        if world == 7 and stage == 4:
            x_pos = info.get('x_pos', 0)
            y_pos = info.get('y_pos', 0)
            if ((506 <= x_pos <= 832 and y_pos > 127) or
                (832 < x_pos <= 1064 and y_pos < 80) or
                (1113 < x_pos <= 1464 and y_pos < 191) or
                (1579 < x_pos <= 1943 and y_pos < 191) or
                (1946 < x_pos <= 1964 and y_pos >= 191) or
                (1984 < x_pos <= 2060 and (y_pos >= 191 or y_pos < 127)) or
                (2114 < x_pos < 2440 and y_pos < 191) or
                x_pos < self.current_x - 500):
                reward -= 50
                terminated = True

        if world == 4 and stage == 4:
            x_pos = info.get('x_pos', 0)
            y_pos = info.get('y_pos', 0)
            if ((x_pos <= 1500 and y_pos < 127) or
                (1588 <= x_pos < 2380 and y_pos >= 127)):
                reward -= 50
                terminated = True

        self.current_x = info.get('x_pos', self.current_x)

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
        self.curr_score = info.get('score', 0)
        self.current_x = info.get('x_pos', 40)
        self._life = info.get('life')
        self._level = (info.get('world'), info.get('stage'))
        self._flag = bool(info.get('flag_get', False))
        return obs, info


def create_mario_env(
    env_id=DEFAULT_ENV_ID,
    seed=None,
    return_base_env=False,
    max_steps=2000,
    use_custom_reward=True,
    render_mode: Optional[str] = None,
    noop_max: int = 30,
    stall_steps: int = 150,
    frontier_prob: float = 0.0,
    demo_starts: bool = False
):
    """Create a Super Mario Bros environment with preprocessing wrappers.

    Args:
        env_id: Registered gym-super-mario-bros id. Single-stage ids such as
            'SuperMarioBros-1-1-v0' end the episode on death or flag;
            'SuperMarioBros-v0' plays the whole game across 3 lives.
        seed: Seed for the first reset (and the NOOP-start RNG), or None to
            leave both unseeded.
        return_base_env: Also return the raw NES env (before any wrapper).
        max_steps: Agent steps (after frame skipping) before truncation.
        use_custom_reward: Apply CustomReward shaping instead of the raw
            env reward.
        render_mode: Gymnasium render mode ('human' or 'rgb_array'), or None
            for headless training.
        noop_max: Upper bound of random NOOP frames after each reset; 0
            disables random starts.
        stall_steps: Agent steps without a new furthest x before the episode
            is truncated; 0 disables the cutoff.
        frontier_prob: Fraction of resets that restart just behind where
            episodes usually end (see FrontierStart); 0 disables. Training only.
        demo_starts: Also restart from the waypoints of this level's
            demonstration route in ``demo_routes/``, if it has one. Only takes
            effect when frontier_prob > 0.

    Returns:
        The wrapped env with (84, 84, 1) uint8 observations, or a (wrapped env,
        raw NES env) tuple when return_base_env is True.
    """
    nes_env = gym_super_mario_bros.make(env_id, render_mode=render_mode)
    env = JoypadSpace(nes_env, SIMPLE_MOVEMENT)

    if noop_max > 0:
        env = NoopReset(env, noop_max=noop_max)

    if frontier_prob > 0:
        route = load_demo_route(env_id) if demo_starts else None
        env = FrontierStart(env, prob=frontier_prob, demo_routes=[route] if route else None)

    if use_custom_reward:
        env = CustomReward(env)

    env = SkipFrame(env, skip=4)
    if stall_steps > 0:
        env = StallTruncation(env, max_steps=stall_steps)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = GrayResizeObservation(env, size=84)

    if seed is not None:
        env.reset(seed=seed)

    if return_base_env:
        return env, nes_env
    return env


def create_vec_env(
    num_envs=8,
    env_id: EnvIds = DEFAULT_ENV_ID,
    frame_stack=4,
    seed=None,
    use_subproc=True,
    use_custom_reward=True,
    render_mode: Optional[str] = None,
    noop_max: int = 30,
    max_episode_steps: int = 2000,
    stall_steps: int = 150,
    frontier_prob: float = 0.0,
    demo_starts: bool = False
):
    """Create vectorized Super Mario Bros environments for parallel training.

    Args:
        num_envs: Number of parallel envs; SubprocVecEnv is only used when
            this is above 1 and use_subproc is True.
        env_id: Registered gym-super-mario-bros id, or a list of ids for
            multi-level training; workers are assigned levels round-robin
            (see ``assign_levels``), so every rollout mixes all levels.
        frame_stack: Frames stacked channels-last; 1 disables stacking.
        seed: Base seed; env i is seeded with seed + i.
        use_subproc: Run envs in worker processes instead of in-process.
        use_custom_reward: Apply CustomReward shaping.
        render_mode: Gymnasium render mode forwarded to every env.
        noop_max: Upper bound of random NOOP frames after each reset.
        max_episode_steps: Agent steps before an episode is truncated.
        stall_steps: Agent steps without forward progress before an episode
            is truncated; 0 disables the cutoff.
        frontier_prob: Fraction of resets that restart just behind each
            worker's frontier; 0 disables. Episode stats from VecMonitor then
            mix full and partial episodes, so judge progress by evaluation.
        demo_starts: Add demonstration-route waypoints to the restart points
            of workers whose level has a route.

    Returns:
        A VecMonitor-wrapped VecEnv whose observations are
        (84, 84, frame_stack) uint8. SB3 transposes them to channels-first
        when the env is passed to PPO.
    """

    worker_env_ids = assign_levels(env_id, num_envs)

    def make_env(rank):
        """Return a picklable factory that builds env ``rank`` inside its worker."""
        def _init():
            if multiprocessing.parent_process() is not None:
                # Ctrl+C reaches the whole process group; workers ignore it so the
                # main process can save the model and shut them down cleanly.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            env_seed = None if seed is None else seed + rank
            return create_mario_env(
                env_id=worker_env_ids[rank],
                seed=env_seed,
                max_steps=max_episode_steps,
                use_custom_reward=use_custom_reward,
                render_mode=render_mode,
                noop_max=noop_max,
                stall_steps=stall_steps,
                frontier_prob=frontier_prob,
                demo_starts=demo_starts
            )
        return _init

    if use_subproc and num_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore
    else:
        vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore

    vec_env = VecMonitor(vec_env)

    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order='last')

    return vec_env


def create_eval_env(
    env_id: EnvIds = DEFAULT_ENV_ID,
    frame_stack=4,
    seed=None,
    use_custom_reward=True,
    noop_max: int = 30,
    stall_steps: int = 150
):
    """Create an evaluation VecEnv laid out like the training env.

    Args:
        env_id: Registered gym-super-mario-bros id, or a list of ids to get
            one in-process env per level. SB3's evaluate_policy splits
            ``n_eval_episodes`` evenly across those envs.
        frame_stack: Frames stacked per observation.
        seed: Seed for the env and its NOOP-start RNG.
        use_custom_reward: Apply CustomReward shaping.
        noop_max: Upper bound of random NOOP frames after each reset; keeps
            deterministic-policy episodes distinct.
        stall_steps: Agent steps without forward progress before an episode
            is truncated; keeps stuck greedy episodes from running 2000 steps.

    Returns:
        A channels-first VecEnv, matching the VecTransposeImage wrapper SB3
        adds to the training env, so EvalCallback sees the same env type.
    """
    num_envs = 1 if isinstance(env_id, str) else len(env_id)

    return VecTransposeImage(create_vec_env(
        num_envs=num_envs,
        env_id=env_id,
        frame_stack=frame_stack,
        seed=seed,
        use_subproc=False,  # Use DummyVecEnv for single eval env
        use_custom_reward=use_custom_reward,
        noop_max=noop_max,
        stall_steps=stall_steps
    ))
