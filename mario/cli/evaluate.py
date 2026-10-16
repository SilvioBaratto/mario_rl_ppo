"""The ``evaluate`` command: run a trained agent for several episodes and report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

from mario.cli.common import configure_sdl, select_device
from mario.domain.config import DEFAULT_ENV_ID

DESCRIPTION = "Evaluate trained Mario PPO agent"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's flags to ``parser``."""
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model (.zip optional)"
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID, help="Environment ID")
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=150,
        help="End an episode after this many steps without a new furthest x (0 disables)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Evaluate on several levels, e.g. 1-1,1-2,1-3,1-4 (overrides --env-id); "
        "--n-episodes runs on each level",
    )
    parser.add_argument(
        "--noop-max",
        type=int,
        default=30,
        help="Max random NOOP frames after each reset; 0 makes every "
        "deterministic episode identical",
    )
    parser.add_argument("--frame-stack", type=int, default=4, help="Frame stack")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos")
    parser.add_argument(
        "--video-dir", type=str, default="eval_videos", help="Directory to save videos"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")


def print_results(results):
    """Print evaluation results"""

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes evaluated: {len(results['episode_rewards'])}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Average reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Average episode length: {results['avg_length']:.1f} +/- {results['std_length']:.1f}")
    print(f"Max reward: {results['max_reward']:.2f}")
    print(f"Min reward: {results['min_reward']:.2f}")
    print(f"Max distance reached: {results['max_distance']}")
    if results.get("whole_game"):
        clean = [info["clean_levels"] for info in results["episode_info"]]
        print(f"Levels cleared before the first life lost: {clean}")
        for n in range(1, max(clean) + 1):
            print(
                f"  >= {n} level(s) without losing a life: "
                f"{sum(c >= n for c in clean)}/{len(clean)}"
            )
    print("=" * 50)

    print("\nPer-episode details:")
    for i, (reward, length, info) in enumerate(
        zip(results["episode_rewards"], results["episode_lengths"], results["episode_info"])
    ):
        status = "COMPLETED" if info["flag_get"] else f"Distance: {info['max_x_pos']}"
        print(f"  Episode {i + 1}: Reward={reward:.2f}, Length={length}, {status}")


def print_level_summary(results_by_env):
    """Print one line per level: success rate, mean reward and furthest distance."""

    print("\n" + "=" * 70)
    print(f"{'LEVEL':28s} {'SUCCESS':>8s} {'AVG REWARD':>12s} {'AVG MAX X':>10s} {'BEST X':>8s}")
    print("=" * 70)
    for env_id, results in results_by_env.items():
        avg_max_x = np.mean([info["max_x_pos"] for info in results["episode_info"]])
        print(
            f"{env_id:28s} {results['success_rate']:7.1f}% {results['avg_reward']:12.2f} "
            f"{avg_max_x:10.1f} {results['max_distance']:8d}"
        )
    print("=" * 70)


def save_results(results_by_env, save_path):
    """Save evaluation results to JSON, keyed by env id under ``levels``."""

    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    json_results = {"timestamp": datetime.now().isoformat(), "levels": {}}
    for env_id, results in results_by_env.items():
        json_results["levels"][env_id] = {
            "episodes": len(results["episode_rewards"]),
            "success_rate": float(results["success_rate"]),
            "avg_reward": float(results["avg_reward"]),
            "std_reward": float(results["std_reward"]),
            "avg_length": float(results["avg_length"]),
            "std_length": float(results["std_length"]),
            "max_reward": float(results["max_reward"]),
            "min_reward": float(results["min_reward"]),
            "max_distance": int(results["max_distance"]),
            "episode_rewards": [float(r) for r in results["episode_rewards"]],
            "episode_lengths": [int(length) for length in results["episode_lengths"]],
            "episode_info": convert_to_native(results["episode_info"]),
        }

    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {save_path}")


def run(args: argparse.Namespace) -> int:
    """Evaluate a trained agent on one or several levels and save a JSON report."""
    configure_sdl(video=args.record_video or args.render)
    import torch

    from mario.application.evaluation import evaluate_agent
    from mario.domain.levels import is_single_stage, parse_levels
    from mario.infrastructure.factory import create_vec_env
    from mario.infrastructure.model import load_mario_model

    if args.record_video or args.render:
        print("Using rendering mode for video capture")
    else:
        print("Using headless mode (no video/audio)")

    device = select_device(args.device)

    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_ids = parse_levels(args.levels) if args.levels else [args.env_id]
    deterministic = args.deterministic and not args.stochastic
    print(f"Policy mode: {'deterministic' if deterministic else 'stochastic'}")

    results_by_env = {}
    for env_id in env_ids:
        print(f"\nCreating environment {env_id}...")
        env = create_vec_env(
            num_envs=1,
            env_id=env_id,
            frame_stack=args.frame_stack,
            seed=args.seed,
            render_mode="human" if args.render else None,
            noop_max=args.noop_max,
            stall_steps=args.stall_steps,
        )

        print(f"Loading PPO agent from {args.model_path}...")
        agent = load_mario_model(args.model_path, env=env, device=device)
        # PPO.load re-seeds the env with the training run's seed; restore --seed.
        agent.set_random_seed(args.seed)

        print(f"Evaluating agent for {args.n_episodes} episodes on {env_id}...")
        video_dir = os.path.join(args.video_dir, env_id) if len(env_ids) > 1 else args.video_dir
        results = evaluate_agent(
            agent=agent,
            env=env,
            n_episodes=args.n_episodes,
            render=args.render,
            record_video=args.record_video,
            video_dir=video_dir,
            deterministic=deterministic,
            whole_game=not is_single_stage(env_id),
        )
        results["whole_game"] = not is_single_stage(env_id)
        print_results(results)
        results_by_env[env_id] = results
        env.close()

    if len(env_ids) > 1:
        print_level_summary(results_by_env)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"eval_results_ppo_{timestamp}.json"
    save_results(results_by_env, results_path)
    return 0
