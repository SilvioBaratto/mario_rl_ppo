"""Playing one game with a trained agent and collecting the frames."""

from __future__ import annotations

import numpy as np

from mario.domain.config import RECORD_MAX_STEPS
from mario.domain.progress import GameProgress, episode_end_reason


def record_game(agent, env, max_steps=RECORD_MAX_STEPS, deterministic=True, stop_at=None):
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
        if stop_at is not None and (info.get("world"), info.get("stage")) == stop_at:
            print(f"Reached {stop_at[0]}-{stop_at[1]}; stopping the recording")
            stopped = True
            break

        if "x_pos" in info:
            max_x_pos = max(max_x_pos, info["x_pos"])

        # On done the VecEnv has already reset, so the screen shows the next
        # episode's first frame; skip it. screen is a zero-copy view, so copy it.
        if not done:
            frames.append(env.envs[0].unwrapped.screen.copy())

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}: x_pos={max_x_pos}, reward={total_reward:.1f}")

    # Game summary
    flag_get = info.get("flag_get", False)
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
        "steps": step_count,
        "reward": total_reward,
        "max_x_pos": max_x_pos,
        "flag_get": flag_get,
    }
