"""The ``record`` command: play one game with a trained agent and save it as an MP4."""

from __future__ import annotations

import argparse
from datetime import datetime

from mario.cli.common import configure_sdl, select_device
from mario.domain.config import (
    DEFAULT_ENV_ID,
    FRAME_STACK,
    NOOP_MAX,
    RECORD_MAX_STEPS,
    SEED,
    STALL_STEPS,
)

DESCRIPTION = "Play one game with trained model and record video"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's flags to ``parser``."""
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model (.zip optional)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output video path (default: auto-generated)"
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID, help="Environment ID")
    parser.add_argument("--frame-stack", type=int, default=FRAME_STACK, help="Frame stack")
    parser.add_argument(
        "--max-steps", type=int, default=RECORD_MAX_STEPS, help="Maximum agent steps in the episode"
    )
    parser.add_argument(
        "--noop-max",
        type=int,
        default=NOOP_MAX,
        help="Max random NOOP frames before the episode starts",
    )
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=STALL_STEPS,
        help="Stop after this many steps without a new furthest x (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic policy",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--stop-at-level",
        type=str,
        default=None,
        help="With a whole-game env id (SuperMarioBros-v0), end the recording when this "
        "level starts, e.g. 2-1 to show 1-1 through 1-4",
    )


def run(args: argparse.Namespace) -> int:
    """Play one episode with a trained model and save it as an MP4."""
    configure_sdl(video=True)
    import imageio
    import numpy as np
    import torch

    from mario.application.recording import record_game
    from mario.domain.levels import parse_levels
    from mario.infrastructure.factory import create_vec_env
    from mario.infrastructure.model import load_mario_model

    # Select device
    device = select_device(args.device)

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
        stall_steps=args.stall_steps,
    )

    print(f"Loading model from {args.model_path}...")
    try:
        agent = load_mario_model(args.model_path, env=env, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        env.close()
        return 0
    # PPO.load re-seeds the env with the training run's seed; restore --seed so
    # different seeds give different NOOP starts.
    agent.set_random_seed(args.seed)

    # Play and record
    frames, game_info = record_game(
        agent=agent,
        env=env,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        stop_at=tuple(int(n) for n in parse_levels(args.stop_at_level)[0].split("-")[1:3])
        if args.stop_at_level
        else None,
    )

    # Save video
    if frames:
        if args.output:
            video_path = args.output
        else:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "complete" if game_info["flag_get"] else f"dist_{game_info['max_x_pos']}"
            video_path = f"mario_gameplay_{timestamp}_{status}.mp4"

        print(f"\nSaving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved successfully! ({len(frames)} frames, {len(frames) / 30:.1f} seconds)")
    else:
        print("Warning: No frames captured, video not saved")

    env.close()
    return 0
