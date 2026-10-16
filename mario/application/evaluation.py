"""Evaluating a trained agent over several episodes."""

from __future__ import annotations

import os

import imageio
import numpy as np
from tqdm import tqdm

from mario.domain.progress import GameProgress, episode_end_reason


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
