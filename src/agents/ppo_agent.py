"""
PPO Agent implementation with custom CNN architecture for Super Mario Bros
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
import numpy as np


class MarioCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for Super Mario Bros.

    Accepts stacked frames channels-first or channels-last. Channels-first is
    the fast path: SB3 produces it for uint8 image observations, whereas
    permuting channels-last input inside forward() is ~3x slower on MPS.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        if observation_space.shape is None:
            raise ValueError("observation_space.shape is None")

        obs_shape = observation_space.shape
        # Determine input channels based on shape
        if obs_shape[0] <= 4:
            n_input_channels = obs_shape[0]
            self.is_channels_first = True
        else:
            n_input_channels = obs_shape[-1]
            self.is_channels_first = False

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_obs = observation_space.sample()[None]

            if self.is_channels_first:
                sample_obs_chw = sample_obs
            else:
                sample_obs_chw = np.transpose(sample_obs, (0, 3, 1, 2))

            n_flatten = self.cnn(torch.as_tensor(sample_obs_chw).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        """Encode a batch of stacked frames.

        Args:
            observations: Float batch already scaled to [0, 1] by SB3, laid out
                like the observation space this extractor was built for.

        Returns:
            Features of shape (batch, features_dim).
        """
        if not self.is_channels_first:
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


def mario_policy_kwargs(features_dim: int = 512) -> Dict[str, Any]:
    """Return the policy_kwargs shared by training and every model loader.

    Args:
        features_dim: Width of MarioCNN's output layer.

    Returns:
        A dict for ``PPO(policy_kwargs=...)`` or for the ``policy_kwargs``
        entry of ``PPO.load(custom_objects=...)``.
    """
    return dict(
        features_extractor_class=MarioCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        # Observations stay uint8 through the env and rollout buffer; SB3 scales
        # them to [0, 1] on the training device.
        normalize_images=True,
    )


def resolve_model_path(path: Union[str, Path]) -> Path:
    """Return an existing checkpoint path, accepting it with or without ``.zip``.

    Args:
        path: Checkpoint path as given on the command line.

    Returns:
        The checkpoint file that exists on disk.

    Raises:
        FileNotFoundError: If neither ``path`` nor ``path`` + ``.zip`` exists.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    with_zip = candidate.with_name(candidate.name + ".zip")
    if with_zip.is_file():
        return with_zip
    raise FileNotFoundError(f"Checkpoint not found: {path} (also tried {with_zip})")


def load_mario_model(
    path: Union[str, Path],
    env: Optional[VecEnv] = None,
    device: str = "auto",
    custom_objects: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> PPO:
    """Load a PPO checkpoint that uses MarioCNN.

    The saved policy_kwargs are replaced with ``mario_policy_kwargs()`` so the
    checkpoint loads regardless of the import path MarioCNN was pickled under.

    Args:
        path: Checkpoint path, with or without ``.zip``.
        env: Env to attach; its observation space must match the checkpoint's.
            Checkpoints trained on float32 observations (before uint8 frames)
            are rejected by SB3 with an observation-space mismatch.
        device: Torch device for the loaded policy.
        custom_objects: Extra saved attributes to override, e.g. learning rate
            or batch size when resuming training.
        **kwargs: Forwarded to ``PPO.load`` (e.g. ``tensorboard_log``).

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    overrides = {"policy_kwargs": mario_policy_kwargs(), **(custom_objects or {})}
    return PPO.load(
        resolve_model_path(path), env=env, device=device, custom_objects=overrides, **kwargs
    )
