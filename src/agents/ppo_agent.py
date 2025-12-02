"""
PPO Agent implementation with custom CNN architecture for Super Mario Bros
"""

import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class MarioCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for Super Mario Bros (Kaggle implementation)"""

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

        # Kaggle notebook architecture: 4 Conv2d layers with 32 filters each
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
            nn.Dropout(0.2)
        )

    def forward(self, observations):
        if not self.is_channels_first:
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


class TrainingCallback(BaseCallback):
    """Callback for logging and checkpointing during training"""

    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        import logging
        self.file_logger = logging.getLogger('mario_rl')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])

                self.logger.record("train/mean_episode_reward", mean_reward)
                self.logger.record("train/mean_episode_length", mean_length)

                self.file_logger.info(f"Checkpoint at {self.n_calls} steps - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_path = os.path.join(self.save_path, 'best_model')
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"New best model saved with mean reward: {mean_reward:.2f}")
                    self.file_logger.info(f"New best model saved with mean reward: {mean_reward:.2f}")

            # Always save checkpoint at check_freq intervals, regardless of ep_info_buffer
            model_path = os.path.join(self.save_path, f'checkpoint_{self.n_calls}')
            self.model.save(model_path)
            self.file_logger.info(f"Checkpoint saved: {model_path}")

        return True
