"""Building the training, evaluation and recording environments from the wrappers."""

from __future__ import annotations

import multiprocessing
import signal

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

from mario.domain.config import (
    DEFAULT_ENV_ID,
    FRAME_SKIP,
    FRAME_STACK,
    MAX_EPISODE_STEPS,
    NOOP_MAX,
    OBS_SIZE,
    STALL_STEPS,
)
from mario.domain.levels import EnvIds, assign_levels
from mario.domain.routes import demo_warmup_frames
from mario.infrastructure.frontier import FrontierStart
from mario.infrastructure.route_files import load_demo_route
from mario.infrastructure.wrappers import (
    CustomReward,
    GrayResizeObservation,
    NoopReset,
    SkipFrame,
    StallTruncation,
)


class LevelStartFrameStack(VecFrameStack):
    """Channels-last ``VecFrameStack`` that clears an env's history when a level starts mid-episode.

    A single-stage episode begins with a zero-filled stack, so the policy
    learned every level's first moves from that. In the whole game the stack
    would still hold the previous level's last screens and change the first
    decisions on the new level (at 1-2's start: run instead of jump). Steps
    flagged ``info['level_start']`` by ``SkipFrame`` get the same zero fill.
    """

    def step_wait(self):
        """Step the envs, stack frames, and zero the older frames of envs that just entered a level.

        Returns:
            The usual (observations, rewards, dones, infos) of ``VecFrameStack``.
        """
        observations, rewards, dones, infos = super().step_wait()
        frame = self.venv.observation_space.shape[-1]
        for idx, info in enumerate(infos):
            if info.get("level_start") and not dones[idx]:
                # update() returns its own buffer, so this clears the kept history too.
                self.stacked_obs.stacked_obs[idx, ..., :-frame] = 0
                observations[idx, ..., :-frame] = 0
        return observations, rewards, dones, infos


def create_mario_env(
    env_id=DEFAULT_ENV_ID,
    seed=None,
    return_base_env=False,
    max_steps=MAX_EPISODE_STEPS,
    use_custom_reward=True,
    render_mode: str | None = None,
    noop_max: int = NOOP_MAX,
    stall_steps: int = STALL_STEPS,
    frontier_prob: float = 0.0,
    demo_starts: bool = False,
    demo_warmup: int = 0,
):
    """Create a Super Mario Bros environment with preprocessing wrappers.

    Args:
        env_id: Registered gym-super-mario-bros id. Single-stage ids such as
            'SuperMarioBros-1-1-v0' end the episode on death or flag;
            'SuperMarioBros-v0' plays the whole game across 3 lives.
        seed: Seed for the first reset (and the NOOP-start RNG), or None to
            leave both unseeded.
        return_base_env: Also return the raw NES env (before any wrapper).
        max_steps: Agent steps (after frame skipping) before truncation.
        use_custom_reward: Apply CustomReward shaping instead of the raw
            env reward.
        render_mode: Gymnasium render mode ('human' or 'rgb_array'), or None
            for headless training.
        noop_max: Upper bound of random NOOP frames after each reset; 0
            disables random starts.
        stall_steps: Agent steps without a new furthest x before the episode
            is truncated; 0 disables the cutoff.
        frontier_prob: Fraction of resets that restart just behind where
            episodes usually end (see FrontierStart); 0 disables. Training only.
        demo_starts: Load this level's demonstration route from
            ``demo_routes/``, if it has one, so frontier restarts can run a
            reverse curriculum along it. Only takes effect when frontier_prob
            > 0 and a ``ReverseCurriculum`` sets the start frame.
        demo_warmup: Route frames replayed before the agent takes over after a
            demo restart; ``create_vec_env`` sets ``demo_warmup_frames(frame_stack)``.

    Returns:
        The wrapped env with (84, 84, 1) uint8 observations, or a (wrapped env,
        raw NES env) tuple when return_base_env is True.
    """
    nes_env = gym_super_mario_bros.make(env_id, render_mode=render_mode)
    env = JoypadSpace(nes_env, SIMPLE_MOVEMENT)

    if noop_max > 0:
        env = NoopReset(env, noop_max=noop_max)

    if frontier_prob > 0:
        route = load_demo_route(env_id) if demo_starts else None
        env = FrontierStart(
            env,
            prob=frontier_prob,
            demo_routes=[route] if route else None,
            warmup_frames=demo_warmup,
        )

    if use_custom_reward:
        env = CustomReward(env)

    env = SkipFrame(env, skip=FRAME_SKIP)
    if stall_steps > 0:
        env = StallTruncation(env, max_steps=stall_steps)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = GrayResizeObservation(env, size=OBS_SIZE)

    if seed is not None:
        env.reset(seed=seed)

    if return_base_env:
        return env, nes_env
    return env


def create_vec_env(
    num_envs=8,
    env_id: EnvIds = DEFAULT_ENV_ID,
    frame_stack=FRAME_STACK,
    seed=None,
    use_subproc=True,
    use_custom_reward=True,
    render_mode: str | None = None,
    noop_max: int = NOOP_MAX,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    stall_steps: int = STALL_STEPS,
    frontier_prob: float = 0.0,
    demo_starts: bool = False,
):
    """Create vectorized Super Mario Bros environments for parallel training.

    Args:
        num_envs: Number of parallel envs; SubprocVecEnv is only used when
            this is above 1 and use_subproc is True.
        env_id: Registered gym-super-mario-bros id, or a list of ids for
            multi-level training; workers are assigned levels round-robin
            (see ``assign_levels``), so every rollout mixes all levels.
        frame_stack: Frames stacked channels-last; 1 disables stacking.
        seed: Base seed; env i is seeded with seed + i.
        use_subproc: Run envs in worker processes instead of in-process.
        use_custom_reward: Apply CustomReward shaping.
        render_mode: Gymnasium render mode forwarded to every env.
        noop_max: Upper bound of random NOOP frames after each reset.
        max_episode_steps: Agent steps before an episode is truncated.
        stall_steps: Agent steps without forward progress before an episode
            is truncated; 0 disables the cutoff.
        frontier_prob: Fraction of resets that restart just behind each
            worker's frontier; 0 disables. Episode stats from VecMonitor then
            mix full and partial episodes, so judge progress by evaluation.
        demo_starts: Load demonstration routes for workers whose level has
            one; drive the curriculum with ``ReverseCurriculum`` and
            ``env_method('set_demo_cursor', ...)``.

    Returns:
        A VecMonitor-wrapped VecEnv whose observations are
        (84, 84, frame_stack) uint8. SB3 transposes them to channels-first
        when the env is passed to PPO.
    """

    worker_env_ids = assign_levels(env_id, num_envs)

    def make_env(rank):
        """Return a picklable factory that builds env ``rank`` inside its worker."""

        def _init():
            if multiprocessing.parent_process() is not None:
                # Ctrl+C reaches the whole process group; workers ignore it so the
                # main process can save the model and shut them down cleanly.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            env_seed = None if seed is None else seed + rank
            return create_mario_env(
                env_id=worker_env_ids[rank],
                seed=env_seed,
                max_steps=max_episode_steps,
                use_custom_reward=use_custom_reward,
                render_mode=render_mode,
                noop_max=noop_max,
                stall_steps=stall_steps,
                frontier_prob=frontier_prob,
                demo_starts=demo_starts,
                demo_warmup=demo_warmup_frames(frame_stack),
            )

        return _init

    if use_subproc and num_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore
    else:
        vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])  # type: ignore

    vec_env = VecMonitor(vec_env)

    if frame_stack > 1:
        vec_env = LevelStartFrameStack(vec_env, n_stack=frame_stack, channels_order="last")

    return vec_env


def create_eval_env(
    env_id: EnvIds = DEFAULT_ENV_ID,
    frame_stack=FRAME_STACK,
    seed=None,
    use_custom_reward=True,
    noop_max: int = NOOP_MAX,
    stall_steps: int = STALL_STEPS,
):
    """Create an evaluation VecEnv laid out like the training env.

    Args:
        env_id: Registered gym-super-mario-bros id, or a list of ids to get
            one in-process env per level. SB3's evaluate_policy splits
            ``n_eval_episodes`` evenly across those envs.
        frame_stack: Frames stacked per observation.
        seed: Seed for the env and its NOOP-start RNG.
        use_custom_reward: Apply CustomReward shaping.
        noop_max: Upper bound of random NOOP frames after each reset; keeps
            deterministic-policy episodes distinct.
        stall_steps: Agent steps without forward progress before an episode
            is truncated; keeps stuck greedy episodes from running 2000 steps.

    Returns:
        A channels-first VecEnv, matching the VecTransposeImage wrapper SB3
        adds to the training env, so EvalCallback sees the same env type.
    """
    num_envs = 1 if isinstance(env_id, str) else len(env_id)

    return VecTransposeImage(
        create_vec_env(
            num_envs=num_envs,
            env_id=env_id,
            frame_stack=frame_stack,
            seed=seed,
            use_subproc=False,  # Use DummyVecEnv for single eval env
            use_custom_reward=use_custom_reward,
            noop_max=noop_max,
            stall_steps=stall_steps,
        )
    )
