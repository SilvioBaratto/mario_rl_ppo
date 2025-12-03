"""
Simple script to play one game with a trained model and record video
"""

import argparse
import os
import sys

# Configure SDL for video recording
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import torch
import numpy as np
import imageio
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environments.env_factory import create_vec_env
from agents.ppo_agent import MarioCNN
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser(description='Play one game with trained model and record video')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: auto-generated)')
    parser.add_argument('--env-id', type=str, default='SuperMarioBros-v0',
                       help='Environment ID')
    parser.add_argument('--frame-stack', type=int, default=4,
                       help='Frame stack')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')

    return parser.parse_args()


def record_game(agent, env, max_steps=5000, deterministic=True):
    """Play one game and record frames"""

    print("Starting game...")
    frames = []
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    max_x_pos = 0
    info = {}

    while not done and step_count < max_steps:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        # Handle vectorized environment outputs
        if isinstance(done, np.ndarray):
            done = done[0]
        if isinstance(reward, np.ndarray):
            reward = reward[0]
        if isinstance(info, list):
            info = info[0]

        total_reward += reward
        step_count += 1

        if 'x_pos' in info:
            max_x_pos = max(max_x_pos, info['x_pos'])

        # Capture frame
        try:
            base_env = env.envs[0]
            unwrapped = base_env
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env

            frame = None

            # Try to get frame from screen buffer
            if hasattr(unwrapped, 'screen') and unwrapped.screen is not None:
                screen = unwrapped.screen
                if isinstance(screen, np.ndarray):
                    frame = screen
                else:
                    # pygame surface
                    import pygame
                    frame = pygame.surfarray.array3d(screen)
                    frame = np.transpose(frame, (1, 0, 2))

            # Fallback to render method
            if frame is None and hasattr(unwrapped, 'render'):
                try:
                    frame = unwrapped.render(mode='rgb_array')
                except:
                    pass

            if frame is not None and frame.size > 0:
                frames.append(frame.copy())
        except Exception as e:
            if step_count == 1:
                print(f"Warning: Frame capture error: {e}")

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}: x_pos={max_x_pos}, reward={total_reward:.1f}")

    # Game summary
    flag_get = info.get('flag_get', False)
    completion_status = "FLAG GET!" if flag_get else f"Died at x={max_x_pos}"

    print(f"\nGame finished!")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Max distance: {max_x_pos}")
    print(f"Status: {completion_status}")
    print(f"Frames captured: {len(frames)}")

    return frames, {
        'steps': step_count,
        'reward': total_reward,
        'max_x_pos': max_x_pos,
        'flag_get': flag_get
    }


def main():
    args = parse_args()

    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    # Select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    print("Creating environment...")
    env = create_vec_env(
        num_envs=1,
        env_id=args.env_id,
        frame_stack=args.frame_stack,
        seed=args.seed
    )

    # Load agent with custom CNN
    print(f"Loading model from {args.model_path}...")

    # Set up custom policy kwargs
    policy_kwargs = dict(
        features_extractor_class=MarioCNN,
        features_extractor_kwargs=dict(features_dim=512),
        normalize_images=False,  # Already normalized to [0,1] in wrapper
    )

    custom_objects = {
        'learning_rate': 1e-4,
        'lr_schedule': lambda _: 1e-4,
        'policy_kwargs': policy_kwargs
    }

    agent = PPO.load(args.model_path, env=env, device=device, custom_objects=custom_objects)

    # Play and record
    frames, game_info = record_game(
        agent=agent,
        env=env,
        max_steps=args.max_steps,
        deterministic=args.deterministic
    )

    # Save video
    if frames:
        if args.output:
            video_path = args.output
        else:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "complete" if game_info['flag_get'] else f"dist_{game_info['max_x_pos']}"
            video_path = f"mario_gameplay_{timestamp}_{status}.mp4"

        print(f"\nSaving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved successfully! ({len(frames)} frames, {len(frames)/30:.1f} seconds)")
    else:
        print("Warning: No frames captured, video not saved")

    env.close()


if __name__ == "__main__":
    main()
