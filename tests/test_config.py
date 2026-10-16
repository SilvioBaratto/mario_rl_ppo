"""Each default is defined once, in mario.domain.config, and every layer reads it from there."""

from __future__ import annotations

import inspect

from mario.domain import config
from tests import adapter as a


def _default(function, parameter: str):
    return inspect.signature(function).parameters[parameter].default


def test_cli_defaults_come_from_the_shared_constants() -> None:
    """The observation pipeline and episode limits are the same on every command."""
    for command in ("train", "evaluate", "record"):
        defaults = a.cli_defaults(command)
        assert defaults["frame_stack"] == config.FRAME_STACK
        assert defaults["noop_max"] == config.NOOP_MAX
        assert defaults["stall_steps"] == config.STALL_STEPS
        assert defaults["seed"] == config.SEED
    assert a.cli_defaults("record")["max_steps"] == config.RECORD_MAX_STEPS


def test_environment_builders_default_to_the_shared_constants() -> None:
    """A library caller that passes nothing gets the same env as a command-line user."""
    for builder in (a.create_vec_env, a.create_eval_env):
        assert _default(builder, "frame_stack") == config.FRAME_STACK
        assert _default(builder, "noop_max") == config.NOOP_MAX
        assert _default(builder, "stall_steps") == config.STALL_STEPS
    assert _default(a.create_mario_env, "max_steps") == config.MAX_EPISODE_STEPS
    assert _default(a.create_vec_env, "max_episode_steps") == config.MAX_EPISODE_STEPS


def test_wrappers_and_cloning_default_to_the_shared_constants() -> None:
    """The cloning data is shaped like what the policy sees in training."""
    assert _default(a.SkipFrame, "skip") == config.FRAME_SKIP
    assert _default(a.NoopReset, "noop_max") == config.NOOP_MAX
    assert _default(a.StallTruncation, "max_steps") == config.STALL_STEPS
    assert _default(a.GrayResizeObservation, "size") == config.OBS_SIZE
    assert _default(a.build_demo_dataset, "frame_stack") == config.FRAME_STACK
    assert _default(a.build_demo_dataset, "max_noop") == config.NOOP_MAX


def test_ppo_hyperparameters_keep_the_values_the_checkpoints_were_trained_with() -> None:
    """These are the values of the reference run; changing one changes what training learns."""
    assert (config.LEARNING_RATE, config.CLIP_RANGE) == (1e-4, 0.2)
    assert (config.N_STEPS, config.GAMMA, config.GAE_LAMBDA) == (512, 0.9, 1.0)
    assert (config.ENT_COEF, config.VF_COEF, config.MAX_GRAD_NORM) == (0.01, 0.5, 0.5)
