"""Reading an episode's story from its step infos."""

from __future__ import annotations

from tests import adapter as a
from tests.doubles import level_info


def test_end_reason_prefers_the_stall_cutoff_then_the_time_limit() -> None:
    """Anything that is neither a stall nor a time limit counts as a death."""
    assert (
        a.episode_end_reason({"stalled": True, "TimeLimit.truncated": True})
        == "Stalled (no progress)"
    )
    assert a.episode_end_reason({"TimeLimit.truncated": True}) == "Time limit"
    assert a.episode_end_reason({}) == "Died"


def test_progress_counts_a_cleared_level_and_a_final_flag() -> None:
    """Each level change counts as a clear, and so does ending on the flag."""
    progress = a.GameProgress()
    progress.update(level_info(1, 1), False)
    progress.update(level_info(1, 2), False)
    progress.update(level_info(1, 2, flag_get=True), True)
    assert progress.levels == [(1, 1), (1, 2)]
    assert progress.levels_cleared == 2
    assert progress.lives_lost == 0
    assert progress.clean_levels is None


def test_progress_counts_a_lost_life_when_life_drops() -> None:
    """A drop in ``life`` is a death and fixes the clean-level count."""
    progress = a.GameProgress()
    progress.update(level_info(1, 1, life=2), False)
    progress.update(level_info(1, 1, life=1), False)
    assert progress.lives_lost == 1
    assert progress.clean_levels == 0


def test_progress_treats_the_final_death_wrap_as_a_death() -> None:
    """The last death wraps life 0 to 255 but ends the episode, which still counts."""
    progress = a.GameProgress()
    progress.update(level_info(1, 1, life=0), False)
    progress.update(level_info(1, 1, life=255), True)
    assert progress.lives_lost == 1


def test_progress_does_not_count_a_truncation_as_a_death() -> None:
    """Stall cutoffs and time limits end an episode without costing a life."""
    progress = a.GameProgress()
    progress.update(level_info(1, 1), False)
    progress.update(level_info(1, 1, stalled=True), True)
    assert progress.lives_lost == 0


def test_progress_summary_names_the_level_reached() -> None:
    """The summary is a single human-readable line."""
    progress = a.GameProgress()
    progress.update(level_info(1, 1, life=2), False)
    progress.update(level_info(1, 1, life=1), False)
    assert progress.summary() == (
        "reached 1-1, cleared 0 level(s), 1 life lost, 0 level(s) cleared before the first life lost"
    )
