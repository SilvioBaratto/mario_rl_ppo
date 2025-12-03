"""
Environment factory for creating Super Mario Bros environments
"""

import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', category=UserWarning, module='gym')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
warnings.filterwarnings('ignore', message='.*step API.*')
warnings.filterwarnings('ignore', message='.*np.bool8.*')

from typing import Any, Dict, Tuple

import gym
from gym import spaces
from gym.core import Wrapper, ObservationWrapper
import numpy as np
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from gym.wrappers.gray_scale_observation import GrayScaleObservation

# Type alias for old Gym step return (obs, reward, done, info)
StepReturn = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class SkipFrame(Wrapper):
    """Skip frames - execute action every skip frames (Kaggle implementation)"""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        total_reward = 0.0
        result = self.env.step(action)
        obs: np.ndarray = result[0]
        total_reward += float(result[1])
        done: bool = result[2]  # type: ignore[assignment]
        info: Dict[str, Any] = result[-1]  # type: ignore[assignment]
        for _ in range(self._skip - 1):
            if done:
                break
            result = self.env.step(action)
            obs = result[0]
            total_reward += float(result[1])
            done = result[2]  # type: ignore[assignment]
            info = result[-1]  # type: ignore[assignment]
        return obs, total_reward, done, info


class ResizeEnv(ObservationWrapper):
    """Resize environment to 84x84"""
    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        old_shape = env.observation_space.shape
        oldc = old_shape[2] if old_shape is not None and len(old_shape) > 2 else 1
        self._size = size
        self._channels = oldc
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, oldc), dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        frame = cv2.resize(observation, (self._size, self._size), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame


class CustomReward(Wrapper):
    """Custom reward shaping for Super Mario Bros.

    Improvements over default rewards:
    - Adds score-based reward bonus (game score divided by 40)
    - +50 bonus for reaching the flag
    - -50 penalty for dying
    - All rewards scaled by 0.1 for more stable training
    """
    def __init__(self, env: gym.Env, world: int = 1, stage: int = 1):
        super().__init__(env)
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        result = self.env.step(action)
        obs: np.ndarray = result[0]
        reward: float = float(result[1])
        done: bool = result[2]  # type: ignore[assignment]
        info: Dict[str, Any] = result[-1]  # type: ignore[assignment]

        # Add score-based reward
        if 'score' in info:
            reward += (info['score'] - self.curr_score) / 40.0
            self.curr_score = info['score']

        # Flag/death bonuses
        if done:
            if info.get('flag_get', False):
                reward += 50
            else:
                reward -= 50

        # Special handling for maze levels (7-4 and 4-4) to penalize wrong paths
        if self.world == 7 and self.stage == 4:
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
                done = True

        if self.world == 4 and self.stage == 4:
            x_pos = info.get('x_pos', 0)
            y_pos = info.get('y_pos', 0)
            if ((x_pos <= 1500 and y_pos < 127) or
                (1588 <= x_pos < 2380 and y_pos >= 127)):
                reward = -50
                done = True

        self.current_x = info.get('x_pos', self.current_x)

        # Scale all rewards by 0.1 for stability
        return obs, reward / 10.0, done, info

    def reset(self, **kwargs):
        self.curr_score = 0
        self.current_x = 40
        return self.env.reset(**kwargs)


class NormalizeObservation(ObservationWrapper):
    """Normalize observations to [0, 1] range by dividing by 255."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_space.shape,
            dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32) / 255.0


class TimeLimitWrapper(Wrapper):
    """Limit the maximum number of steps per episode to prevent getting stuck."""
    def __init__(self, env: gym.Env, max_steps: int = 2000):
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps = 0

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        result = self.env.step(action)
        obs: np.ndarray = result[0]
        reward: float = float(result[1])
        done: bool = result[2]  # type: ignore[assignment]
        info: Dict[str, Any] = result[-1]  # type: ignore[assignment]

        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def parse_world_stage(env_id: str) -> Tuple[int, int]:
    """Parse world and stage from environment ID.

    Examples:
        'SuperMarioBros-v0' -> (1, 1)
        'SuperMarioBros-1-2-v0' -> (1, 2)
        'SuperMarioBros-7-4-v0' -> (7, 4)
    """
    import re
    match = re.match(r'SuperMarioBros-(\d+)-(\d+)-v\d+', env_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 1  # Default to World 1-1


def create_mario_env(
    env_id='SuperMarioBros-v0',
    seed=None,
    return_base_env=False,
    max_steps=2000,
    use_custom_reward=True
):
    """Create a Super Mario Bros environment with preprocessing wrappers."""
    world, stage = parse_world_stage(env_id)

    nes_env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(nes_env, SIMPLE_MOVEMENT)

    if use_custom_reward:
        env = CustomReward(env, world=world, stage=stage)

    env = SkipFrame(env, skip=4)
    env = TimeLimitWrapper(env, max_steps=max_steps)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = NormalizeObservation(env)
    env = GymV21CompatibilityV0(env=env)

    if seed is not None:
        env.reset(seed=seed)

    if return_base_env:
        return env, nes_env
    return env


def create_vec_env(
    num_envs=8,
    env_id='SuperMarioBros-v0',
    frame_stack=4,
    seed=None,
    use_subproc=True,
    use_custom_reward=True
):
    """Create vectorized Super Mario Bros environments for parallel training."""

    def make_env(rank):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_mario_env(
                env_id=env_id,
                seed=env_seed,
                use_custom_reward=use_custom_reward
            )
        return _init
    
    if use_subproc and num_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore
    else:
        vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore

    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order='last')

    return vec_env


def create_eval_env(
    env_id='SuperMarioBros-v0',
    frame_stack=4,
    seed=None,
    use_custom_reward=True
):
    """Create an evaluation environment."""

    return create_vec_env(
        num_envs=1,
        env_id=env_id,
        frame_stack=frame_stack,
        seed=seed,
        use_subproc=False,  # Use DummyVecEnv for single eval env
        use_custom_reward=use_custom_reward
    )
