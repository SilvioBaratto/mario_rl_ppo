"""The reverse curriculum that moves a level's start point backward along its demonstration."""

from __future__ import annotations

import pytest

from tests import adapter as a
from tests.doubles import route


def _curriculum(**kwargs):
    return a.ReverseCurriculum([route([0] * 40, 80)], warmup_frames=0, window=4, **kwargs)


def _outcome(frame: int, success: bool = True) -> dict:
    return {"level": "1-1", "frame": frame, "success": success}


def test_warmup_covers_one_less_frame_stack_of_agent_steps() -> None:
    """The warm-up replays frame_stack - 1 agent steps of FRAME_SKIP frames each."""
    assert a.demo_warmup_frames(4) == 12
    assert a.demo_warmup_frames(1) == 0


def test_start_frames_are_strided_and_leave_room_for_the_warmup() -> None:
    """Every stride-th frame qualifies once it is past the warm-up."""
    short = route([0] * 40, 80)
    assert a.demo_start_frames(short, 8, 0) == [8, 16, 24, 32, 40]
    assert a.demo_start_frames(short, 8, 12) == [16, 24, 32, 40]


def test_curriculum_starts_at_the_end_of_the_route() -> None:
    """The first start is the last strided frame, past the obstacle."""
    assert _curriculum().cursors() == {"1-1": 40}


def test_curriculum_steps_back_once_enough_recent_episodes_succeed() -> None:
    """With threshold 0.5 over 4 episodes, two successes move the cursor one start earlier."""
    curriculum = _curriculum()
    assert curriculum.required == 2
    assert curriculum.record(_outcome(40)) is None
    assert curriculum.record(_outcome(40)) == (2, 2)
    assert curriculum.cursors() == {"1-1": 32}


def test_curriculum_counts_failures_toward_the_window_but_not_the_threshold() -> None:
    """A failure is an episode played but not a success."""
    curriculum = _curriculum()
    assert curriculum.record(_outcome(40, False)) is None
    assert curriculum.record(_outcome(40)) is None
    assert curriculum.record(_outcome(40)) == (2, 3)


def test_curriculum_ignores_episodes_from_other_start_frames() -> None:
    """Rehearsal episodes from a later state say nothing about the current start."""
    curriculum = _curriculum()
    assert curriculum.record(_outcome(32)) is None
    assert curriculum.record(_outcome(32)) is None
    assert curriculum.cursors() == {"1-1": 40}


def test_curriculum_finishes_with_no_cursor_after_the_first_start() -> None:
    """Past the earliest start frame the level's curriculum is done."""
    curriculum = _curriculum()
    for frame in (40, 32, 24, 16, 8):
        curriculum.record(_outcome(frame))
        curriculum.record(_outcome(frame))
    assert curriculum.cursors() == {"1-1": None}


def test_curriculum_resumes_from_a_logged_start_frame_rounded_down() -> None:
    """A frame between start frames rounds down to the earlier one."""
    assert _curriculum(start_frames={"1-1": 30}).cursors() == {"1-1": 24}
    assert _curriculum(start_frames={"1-1": 3}).cursors() == {"1-1": None}


def test_curriculum_rejects_a_resume_position_for_a_level_without_a_route() -> None:
    """Resuming a level that has no route is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="no demo route for level 1-2"):
        _curriculum(start_frames={"1-2": 8})
