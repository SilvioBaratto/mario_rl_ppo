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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments.env_factory import DEFAULT_ENV_ID, GameProgress, create_vec_env, episode_end_reason, parse_levels
from agents.ppo_agent import load_mario_model


def parse_args():
    """Parse the recording command line."""
    parser = argparse.ArgumentParser(description='Play one game with trained model and record video')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.zip optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: auto-generated)')
    parser.add_argument('--env-id', type=str, default=DEFAULT_ENV_ID,
                       help='Environment ID')
    parser.add_argument('--frame-stack', type=int, default=4,
                       help='Frame stack')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum agent steps in the episode')
    parser.add_argument('--noop-max', type=int, default=30,
                       help='Max random NOOP frames before the episode starts')
    parser.add_argument('--stall-steps', type=int, default=150,
                       help='Stop after this many steps without a new furthest x (0 disables)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--stochastic', action='store_true',
                       help='Sample actions instead of using the deterministic policy')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--stop-at-level', type=str, default=None,
                       help='With a whole-game env id (SuperMarioBros-v0), end the recording when this '
                            'level starts, e.g. 2-1 to show 1-1 through 1-4')

    return parser.parse_args()


def record_game(agent, env, max_steps=5000, deterministic=True, stop_at=None):
    """Play one game and record frames.

    Args:
        stop_at: (world, stage) whose start ends the recording, or None to
            play until the episode ends.
    """

    print("Starting game...")
    frames = []
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    max_x_pos = 0
    info = {}
    progress = GameProgress()
    stopped = False

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
        progress.update(info, bool(done))
        if stop_at is not None and (info.get('world'), info.get('stage')) == stop_at:
            print(f"Reached {stop_at[0]}-{stop_at[1]}; stopping the recording")
            stopped = True
            break

        if 'x_pos' in info:
            max_x_pos = max(max_x_pos, info['x_pos'])

        # On done the VecEnv has already reset, so the screen shows the next
        # episode's first frame; skip it. screen is a zero-copy view, so copy it.
        if not done:
            frames.append(env.envs[0].unwrapped.screen.copy())

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}: x_pos={max_x_pos}, reward={total_reward:.1f}")

    # Game summary
    flag_get = info.get('flag_get', False)
    if stopped:
        completion_status = f"Reached {stop_at[0]}-{stop_at[1]} (recording stopped)"
    elif flag_get:
        completion_status = "FLAG GET!"
    else:
        completion_status = f"{episode_end_reason(info)} at x={max_x_pos}"

    print("\nGame finished!")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Max distance: {max_x_pos}")
    print(f"Status: {completion_status}")
    print(f"Game: {progress.summary()}")
    print(f"Frames captured: {len(frames)}")

    return frames, {
        'steps': step_count,
        'reward': total_reward,
        'max_x_pos': max_x_pos,
        'flag_get': flag_get
    }


def main():
    """Play one episode with a trained model and save it as an MP4."""
    args = parse_args()

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
        seed=args.seed,
        noop_max=args.noop_max,
        max_episode_steps=args.max_steps,
        stall_steps=args.stall_steps
    )

    print(f"Loading model from {args.model_path}...")
    try:
        agent = load_mario_model(args.model_path, env=env, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        env.close()
        return
    # PPO.load re-seeds the env with the training run's seed; restore --seed so
    # different seeds give different NOOP starts.
    agent.set_random_seed(args.seed)

    # Play and record
    frames, game_info = record_game(
        agent=agent,
        env=env,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        stop_at=tuple(int(n) for n in parse_levels(args.stop_at_level)[0].split('-')[1:3])
        if args.stop_at_level else None
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
