"""
Evaluation script for trained Mario RL agents
"""

import argparse
import os

temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--record-video", action="store_true")
temp_parser.add_argument("--render", action="store_true")
temp_args, _ = temp_parser.parse_known_args()

if not temp_args.record_video and not temp_args.render:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
else:
    os.environ["SDL_AUDIODRIVER"] = "dummy"

import json
from datetime import datetime

import imageio
import numpy as np
import torch
from tqdm import tqdm

from mario.agents.ppo_agent import load_mario_model
from mario.environments.env_factory import (
    DEFAULT_ENV_ID,
    GameProgress,
    create_vec_env,
    episode_end_reason,
    is_single_stage,
    parse_levels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Mario PPO agent")

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

    return parser.parse_args()


def evaluate_agent(
    agent,
    env,
    n_episodes,
    render=False,
    record_video=False,
    video_dir=None,
    deterministic=True,
    whole_game=False,
):
    """Evaluate agent performance over multiple episodes.

    With ``whole_game`` (an id like 'SuperMarioBros-v0'), each episode also
    reports the levels it reached and cleared and the lives it lost.
    """

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_info": [],
        "success_rate": 0,
        "avg_reward": 0,
        "avg_length": 0,
        "max_reward": -float("inf"),
        "min_reward": float("inf"),
        "max_distance": 0,
    }

    if record_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        frames = []
        max_x_pos = 0
        action_counts = {}
        progress = GameProgress()

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            action_value = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            action_counts[action_value] = action_counts.get(action_value, 0) + 1

            obs, reward, done, info = env.step(action)

            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(info, list):
                info = info[0]

            total_reward += reward
            episode_length += 1
            progress.update(info, bool(done))

            if "x_pos" in info:
                max_x_pos = max(max_x_pos, info["x_pos"])

            # On done the VecEnv has already reset, so the screen shows the next
            # episode's first frame; skip it. screen is a zero-copy view, so copy it.
            if record_video and not done:
                frames.append(env.envs[0].unwrapped.screen.copy())

            if render:
                env.render()

        flag_get = info.get("flag_get", False)

        if record_video and frames:
            status = "COMPLETE" if flag_get else f"dist_{max_x_pos}"
            video_path = os.path.join(video_dir, f"episode_{episode}_{status}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Video saved to {video_path}")

        results["episode_rewards"].append(total_reward)
        results["episode_lengths"].append(episode_length)
        results["episode_info"].append(
            {
                "max_x_pos": max_x_pos,
                "flag_get": flag_get,
                "time": info.get("time", 0),
                "coins": info.get("coins", 0),
                "score": info.get("score", 0),
                "levels_cleared": progress.levels_cleared,
                "lives_lost": progress.lives_lost,
                "clean_levels": progress.levels_cleared
                if progress.clean_levels is None
                else progress.clean_levels,
            }
        )

        if flag_get:
            results["success_rate"] += 1

        results["max_reward"] = max(results["max_reward"], total_reward)
        results["min_reward"] = min(results["min_reward"], total_reward)
        results["max_distance"] = max(results["max_distance"], max_x_pos)

        completion_status = (
            "FLAG GET!" if flag_get else f"{episode_end_reason(info)} at x={max_x_pos}"
        )
        most_common_action = (
            max(action_counts.items(), key=lambda x: x[1]) if action_counts else (0, 0)
        )
        print(
            f"Episode {episode + 1}: Reward={total_reward:.2f}, "
            f"Length={episode_length}, {completion_status}, "
            f"Most common action: {most_common_action[0]} "
            f"({most_common_action[1]}/{episode_length} times)"
        )
        print(f"  Action distribution: {dict(sorted(action_counts.items()))}")
        if whole_game:
            print(f"  Game: {progress.summary()}")

    results["success_rate"] = results["success_rate"] / n_episodes * 100
    results["avg_reward"] = np.mean(results["episode_rewards"])
    results["std_reward"] = np.std(results["episode_rewards"])
    results["avg_length"] = np.mean(results["episode_lengths"])
    results["std_length"] = np.std(results["episode_lengths"])

    return results


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


def main():
    args = parse_args()

    if args.record_video or args.render:
        print("Using rendering mode for video capture")
    else:
        print("Using headless mode (no video/audio)")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

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


if __name__ == "__main__":
    main()
