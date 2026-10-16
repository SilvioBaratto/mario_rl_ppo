"""The wrappers that turn the raw emulator into the training environment."""

from __future__ import annotations

import numpy as np
import pytest

from tests import adapter as a
from tests.doubles import ScriptedEnv, level_info

NOOP = a.NOOP_ACTION


def _rows(count: int, reward: float = 1.0, **fields) -> list:
    return [(reward, False, False, level_info(**fields)) for _ in range(count)]


def test_skip_frame_repeats_the_action_and_sums_the_rewards() -> None:
    """One agent step is FRAME_SKIP emulator frames with the same buttons."""
    env = ScriptedEnv(level_info(), _rows(4))
    wrapped = a.SkipFrame(env, skip=4)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(3)
    assert env.actions == [3, 3, 3, 3]
    assert reward == pytest.approx(4.0)
    assert not (terminated or truncated)


def test_skip_frame_stops_early_when_the_episode_ends() -> None:
    """A terminal frame ends the agent step without playing the rest."""
    rows = [(1.0, False, False, level_info()), (1.0, True, False, level_info())] + _rows(2)
    env = ScriptedEnv(level_info(), rows)
    wrapped = a.SkipFrame(env, skip=4)
    wrapped.reset()
    _, reward, terminated, _, _ = wrapped.step(3)
    assert env.actions == [3, 3]
    assert terminated
    assert reward == pytest.approx(2.0)


def test_skip_frame_sends_no_buttons_after_the_flag_is_touched() -> None:
    """The game ignores input in the end sequence, and the next level must not inherit a held A."""
    rows = [(0.0, False, False, level_info()), (0.0, False, False, level_info(flag_get=True))] + _rows(2)
    env = ScriptedEnv(level_info(), rows)
    wrapped = a.SkipFrame(env, skip=4)
    wrapped.reset()
    wrapped.step(5)
    assert env.actions == [5, 5, NOOP, NOOP]


def test_skip_frame_ends_the_step_when_a_new_level_starts() -> None:
    """The remaining frames would replay an action chosen on the previous level's screen."""
    rows = [(1.0, False, False, level_info(1, 1)), (1.0, False, False, level_info(1, 2))] + _rows(2, stage=2)
    env = ScriptedEnv(level_info(1, 1), rows)
    wrapped = a.SkipFrame(env, skip=4)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(5)
    assert env.actions == [5, 5]
    assert info["level_start"] is True


def test_stall_truncation_cuts_the_episode_after_too_many_steps_without_progress() -> None:
    """Truncation, not termination, so PPO still bootstraps the value of the cut-off state."""
    env = ScriptedEnv(level_info(x_pos=100), _rows(10, x_pos=100))
    wrapped = a.StallTruncation(env, max_steps=3)
    wrapped.reset()
    outcomes = [wrapped.step(0) for _ in range(3)]
    assert [o[3] for o in outcomes] == [False, False, True]
    assert outcomes[-1][4]["stalled"] is True
    assert not outcomes[-1][2]


def test_stall_truncation_restarts_the_count_on_a_new_furthest_x() -> None:
    """Only steps without a new furthest x count as stalling."""
    rows = [(0.0, False, False, level_info(x_pos=x)) for x in (41, 41, 42, 42, 42, 43)]
    env = ScriptedEnv(level_info(x_pos=40), rows)
    wrapped = a.StallTruncation(env, max_steps=2)
    wrapped.reset()
    truncated = [wrapped.step(0)[3] for _ in range(6)]
    assert truncated == [False, False, False, False, True, False]


def test_stall_truncation_measures_progress_per_life() -> None:
    """x_pos restarts at a respawn, so a new life begins a new segment."""
    rows = [(0.0, False, False, level_info(x_pos=500)), (0.0, False, False, level_info(x_pos=40, life=1))]
    env = ScriptedEnv(level_info(x_pos=40), rows * 3)
    wrapped = a.StallTruncation(env, max_steps=1)
    wrapped.reset()
    assert wrapped.step(0)[3] is False
    assert wrapped.step(0)[3] is False


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_noop_reset_idles_for_a_seeded_number_of_frames(seed: int) -> None:
    """The count is the first draw of default_rng(seed) over 0..noop_max, played as NOOPs."""
    env = ScriptedEnv(level_info(), _rows(40))
    wrapped = a.NoopReset(env, noop_max=30)
    wrapped.reset(seed=seed)
    assert env.actions == [NOOP] * int(np.random.default_rng(seed).integers(0, 31))
    assert env.reset_seeds == [seed]


def test_noop_reset_never_exceeds_the_maximum() -> None:
    """Across many seeds the idle count stays within 0..noop_max."""
    counts = []
    for seed in range(40):
        env = ScriptedEnv(level_info(), _rows(40))
        a.NoopReset(env, noop_max=5).reset(seed=seed)
        counts.append(len(env.actions))
    assert min(counts) == 0
    assert max(counts) == 5


def test_noop_reset_without_a_seed_keeps_the_running_generator() -> None:
    """Only an explicit seed reseeds, so successive resets draw successive counts."""
    env = ScriptedEnv(level_info(), _rows(40))
    wrapped = a.NoopReset(env, noop_max=30)
    wrapped.reset(seed=3)
    first = len(env.actions)
    wrapped.reset()
    second = len(env.actions) - first
    rng = np.random.default_rng(3)
    assert (first, second) == (int(rng.integers(0, 31)), int(rng.integers(0, 31)))


def test_custom_reward_adds_the_score_gain_and_scales_by_a_tenth() -> None:
    """Reward = (raw + score gain / 40) / 10."""
    env = ScriptedEnv(level_info(), [(1.0, False, False, level_info(score=400))])
    wrapped = a.CustomReward(env)
    wrapped.reset()
    assert wrapped.step(0)[1] == pytest.approx((1.0 + 400 / 40) / 10)


def test_custom_reward_pays_the_flag_bonus_once() -> None:
    """The bonus fires on the rising edge of flag_get, not on every flagged frame."""
    flagged = (1.0, False, False, level_info(flag_get=True))
    wrapped = a.CustomReward(ScriptedEnv(level_info(), [flagged, flagged]))
    wrapped.reset()
    assert wrapped.step(0)[1] == pytest.approx((1.0 + 50) / 10)
    assert wrapped.step(0)[1] == pytest.approx(1.0 / 10)


def test_custom_reward_charges_a_lost_life() -> None:
    """A drop in ``life`` costs 50 before scaling."""
    wrapped = a.CustomReward(ScriptedEnv(level_info(), [(1.0, False, False, level_info(life=1))]))
    wrapped.reset()
    assert wrapped.step(0)[1] == pytest.approx((1.0 - 50) / 10)


def test_custom_reward_charges_a_death_that_ends_the_episode() -> None:
    """The final death wraps life to 255, so termination without the flag is the signal."""
    wrapped = a.CustomReward(ScriptedEnv(level_info(life=0), [(1.0, True, False, level_info(life=255))]))
    wrapped.reset()
    assert wrapped.step(0)[1] == pytest.approx((1.0 - 50) / 10)


def test_custom_reward_ends_the_episode_on_a_wrong_path_in_the_4_4_maze() -> None:
    """Castle 4-4 loops unless Mario takes the right route, so the wrong one is fatal."""
    wrong = level_info(4, 4, x_pos=1000, y_pos=100)
    wrapped = a.CustomReward(ScriptedEnv(level_info(4, 4), [(0.0, False, False, wrong)]))
    wrapped.reset()
    _, reward, terminated, _, _ = wrapped.step(0)
    assert terminated
    assert reward == pytest.approx(-50 / 10)
