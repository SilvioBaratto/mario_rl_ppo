"""Simplified training script for Super Mario Bros RL agent using PPO."""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import argparse
import logging
import os
import sys
import signal
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure SDL drivers before environment creation
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from environments.env_factory import create_vec_env, create_eval_env
from utils.logger import setup_logger


_agent = None
_dirs = None
_env = None


class CheckpointCallback(BaseCallback):
    """Simple callback for saving model checkpoints."""

    def __init__(self, save_freq, save_path, num_envs=1, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Calculate actual timesteps (n_calls * num_envs)
            timesteps = self.n_calls * self.num_envs
            checkpoint_path = self.save_path / f'checkpoint_{timesteps}'
            self.model.save(str(checkpoint_path))
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_path.name}")
        return True


def signal_handler(signum, _):
    global _agent, _dirs, _env

    print(f"\n\nReceived {signal.Signals(signum).name} signal")

    if _agent and _dirs:
        save_path = _dirs['checkpoints'] / f'interrupted_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        try:
            _agent.save(str(save_path))
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    if _env:
        _env.close()

    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mario RL Agent with PPO (Simplified)')

    parser.add_argument('--env-id', type=str, default='SuperMarioBros-v0')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--total-timesteps', type=int, default=10_000_000)
    parser.add_argument('--eval-freq', type=int, default=50_000)
    parser.add_argument('--save-freq', type=int, default=100_000)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'])
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    return parser.parse_args()


def setup_directories(exp_name):
    exp_dir = Path(f"runs/{exp_name}")

    dirs = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs',
        'videos': exp_dir / 'videos'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def select_device(device_arg):
    if device_arg != 'auto':
        return device_arg

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def create_callbacks(args, dirs):
    """Create training callbacks for checkpointing and evaluation."""
    callbacks = []

    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=str(dirs['checkpoints']),
        num_envs=args.num_envs,
        verbose=1
    ))

    # Evaluation callback
    if args.eval_freq > 0:
        eval_env = create_eval_env(
            env_id=args.env_id,
            frame_stack=args.frame_stack,
            seed=args.seed + 1000
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(dirs['checkpoints']),
            log_path=str(dirs['root']),
            eval_freq=args.eval_freq // args.num_envs,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

    return callbacks


def main():
    args = parse_args()

    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"ppo_{args.env_id.replace('SuperMarioBros-', '')}_{timestamp}"

    # Setup directories
    dirs = setup_directories(args.exp_name)

    # Setup logger
    logger = setup_logger(dirs['logs'])
    logger.info("=" * 60)
    logger.info("Starting Super Mario Bros RL Training (Simplified)")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Total timesteps: {args.total_timesteps:,}")
    logger.info(f"Parallel environments: {args.num_envs}")
    logger.info(f"Frame stack: {args.frame_stack}")
    logger.info("Action space: SIMPLE_MOVEMENT (7 actions)")
    logger.info("Hyperparameters (Optimized for fast training):")
    logger.info("  learning_rate=lin_2.5e-4, n_steps=128, batch_size=256")
    logger.info("  n_epochs=4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Select device
    device = select_device(args.device)
    logger.info(f"Device: {device}")

    # Create environment
    logger.info("Creating vectorized environment...")
    env = create_vec_env(
        num_envs=args.num_envs,
        env_id=args.env_id,
        frame_stack=args.frame_stack,
        seed=args.seed
    )

    # Store globals for signal handler
    global _agent, _dirs, _env
    _dirs = dirs
    _env = env

    # Create or load PPO agent
    from agents.ppo_agent import MarioCNN

    policy_kwargs = dict(
        features_extractor_class=MarioCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    if args.resume:
        # Resume from checkpoint
        logger.info(f"Resuming training from {args.resume}")
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        custom_objects = {
            'learning_rate': linear_schedule(2.5e-4),
            'clip_range': linear_schedule(0.1),
            'policy_kwargs': policy_kwargs
        }

        agent = PPO.load(
            args.resume,
            env=env,
            device=device,
            custom_objects=custom_objects,
            tensorboard_log=str(dirs['logs'])
        )
        logger.info("Model loaded successfully, resuming training...")
    else:
        # Create new PPO agent with optimized hyperparameters
        logger.info("Creating PPO agent with custom MarioCNN...")

        agent = PPO(
            'CnnPolicy',
            env,
            learning_rate=linear_schedule(2.5e-4),
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.1),
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(dirs['logs']),
            device=device
        )
    _agent = agent

    # Create callbacks
    callbacks = create_callbacks(args, dirs)

    try:
        logger.info(f"Starting training for {args.total_timesteps:,} timesteps")
        print(f"\nTraining: {args.exp_name}")
        print(f"Device: {device}")
        print(f"Timesteps: {args.total_timesteps:,}")
        print("-" * 50)

        agent.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        final_path = dirs['checkpoints'] / 'final_model'
        agent.save(str(final_path))
        logger.info(f"Training complete. Final model saved to {final_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        save_path = dirs['checkpoints'] / f'interrupted_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        agent.save(str(save_path))
        logger.info(f"Model saved to {save_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()
