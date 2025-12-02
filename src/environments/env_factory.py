"""
Environment factory for creating Super Mario Bros environments
Following Kaggle notebook implementation
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
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
    """Resize environment to 84x84 (Kaggle implementation)"""
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


class NormalizeRewardEnv(Wrapper):
    """Normalize rewards from the environment.

    The default gym-super-mario-bros reward function is already well-designed:
    - x_reward: Horizontal progress (positive for moving right)
    - time_penalty: Small penalty for each frame (encourages speed)
    - death_penalty: -25 for dying

    Reward range is clipped to (-15, 15) by the environment.

    This wrapper optionally scales rewards for more stable training.
    Reference: https://github.com/Kautenja/gym-super-mario-bros
    """
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action: int) -> StepReturn:  # type: ignore[override]
        result = self.env.step(action)
        obs: np.ndarray = result[0]
        reward: float = float(result[1]) * self.scale
        done: bool = result[2]  # type: ignore[assignment]
        info: Dict[str, Any] = result[-1]  # type: ignore[assignment]
        return obs, reward, done, info


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


def create_mario_env(
    env_id='SuperMarioBros-v0',
    seed=None,
    return_base_env=False,
    reward_scale=1.0,
    max_steps=2000
):
    """Create a Super Mario Bros environment with standard preprocessing.

    Uses the default gym-super-mario-bros reward function which is well-designed:
    - x_reward: Horizontal movement progress (positive for right, negative for left)
    - time_penalty: Small penalty per frame to encourage speed
    - death_penalty: -25 for dying

    Preprocessing pipeline:
    1. JoypadSpace with SIMPLE_MOVEMENT (7 discrete actions)
    2. NormalizeRewardEnv (optional reward scaling)
    3. SkipFrame (skip=4 for temporal abstraction)
    4. TimeLimitWrapper (max_steps to prevent getting stuck)
    5. GrayScaleObservation (keep_dim=True)
    6. ResizeEnv (84x84)
    7. GymV21CompatibilityV0 (for Stable Baselines3 compatibility)

    Args:
        env_id: Environment ID (default: 'SuperMarioBros-v0')
        seed: Random seed for reproducibility
        return_base_env: If True, return both wrapped and base NES environment
        reward_scale: Scale factor for rewards (default: 1.0, no scaling)
        max_steps: Maximum steps per episode before reset (default: 2000)
    """

    nes_env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(nes_env, SIMPLE_MOVEMENT)
    env = NormalizeRewardEnv(env, scale=reward_scale)
    env = SkipFrame(env, skip=4)
    env = TimeLimitWrapper(env, max_steps=max_steps)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
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
    reward_scale=1.0
):
    """Create vectorized Super Mario Bros environments for parallel training.

    Args:
        num_envs: Number of parallel environments
        env_id: Environment ID
        frame_stack: Number of frames to stack (default: 4)
        seed: Random seed for reproducibility
        reward_scale: Scale factor for rewards (default: 1.0)
    """

    def make_env(rank):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_mario_env(
                env_id=env_id,
                seed=env_seed,
                reward_scale=reward_scale
            )
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore

    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order='last')

    return vec_env


def create_eval_env(
    env_id='SuperMarioBros-v0',
    frame_stack=4,
    seed=None,
    reward_scale=1.0
):
    """Create an evaluation environment.

    Args:
        env_id: Environment ID
        frame_stack: Number of frames to stack (default: 4)
        seed: Random seed for reproducibility
        reward_scale: Scale factor for rewards (default: 1.0)
    """

    return create_vec_env(
        num_envs=1,
        env_id=env_id,
        frame_stack=frame_stack,
        seed=seed,
        reward_scale=reward_scale
    )
