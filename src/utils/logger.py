"""
Logging utilities for training and evaluation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np


def setup_logger(log_dir, name='mario_rl'):
    """Setup logger for training"""

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f'training_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MetricsLogger:
    """Logger for training metrics"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'timesteps': [],
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'success_rate': [],
            'learning_rate': [],
            'loss': []
        }

        self.episode_count = 0
        self.timestep_count = 0

    def log_episode(self, reward, length, success, info=None):
        self.episode_count += 1
        self.metrics['episodes'].append(self.episode_count)
        self.metrics['rewards'].append(reward)
        self.metrics['lengths'].append(length)

        if len(self.metrics['rewards']) >= 100:
            recent_successes = sum(self.metrics['success_rate'][-100:])
            success_rate = recent_successes / 100
        else:
            success_rate = success

        self.metrics['success_rate'].append(success_rate)

    def log_training(self, timesteps, loss, learning_rate):
        self.timestep_count = timesteps
        self.metrics['timesteps'].append(timesteps)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(learning_rate)

    def save(self, filename='metrics.json'):
        filepath = self.log_dir / filename

        save_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                if isinstance(values[0], (np.integer, np.floating)):
                    save_metrics[key] = [float(v) for v in values]
                else:
                    save_metrics[key] = values

        with open(filepath, 'w') as f:
            json.dump(save_metrics, f, indent=2)

    def get_summary(self):
        if len(self.metrics['rewards']) == 0:
            return {}

        summary = {
            'total_episodes': self.episode_count,
            'total_timesteps': self.timestep_count,
            'avg_reward': np.mean(self.metrics['rewards']),
            'std_reward': np.std(self.metrics['rewards']),
            'max_reward': np.max(self.metrics['rewards']),
            'min_reward': np.min(self.metrics['rewards']),
            'avg_length': np.mean(self.metrics['lengths']),
            'recent_success_rate': self.metrics['success_rate'][-1] if self.metrics['success_rate'] else 0
        }

        return summary


class TensorBoardLogger:
    """TensorBoard logging wrapper"""

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step=None):
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag, image, step=None):
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step)

    def log_video(self, tag, video, step=None, fps=30):
        if step is None:
            step = self.step
        self.writer.add_video(tag, video, step, fps=fps)

    def update_step(self, step):
        self.step = step

    def close(self):
        self.writer.close()
