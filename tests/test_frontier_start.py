"""Demonstration replay and frontier restarts, driven by a one-dimensional fake emulator."""

from __future__ import annotations

import pytest

from tests import adapter as a
from tests.doubles import FakeNes, route

# 40 frames of one pixel each from x=40 reach x=80; the 4 frames before a start are replayed as warm-up.
FRAMES = [1] * 40
GOAL_X = 80


def test_replay_collects_a_state_for_every_start_frame() -> None:
    """Each state comes with the bytes that lead into the frame and the x reached there."""
    nes = FakeNes()
    states = a.replay_demo_route(nes, route(FRAMES, GOAL_X), stride=8, warmup_frames=4)
    assert sorted(states) == [8, 16, 24, 32, 40]
    state, joypad, x_pos = states[24]
    assert x_pos == 64
    assert joypad == [1] * 5
    assert state == 59


def test_replay_gives_up_with_a_warning_when_a_waypoint_is_missed() -> None:
    """A route that no longer reaches its waypoint x would teach nonsense, so it is dropped."""
    with pytest.warns(UserWarning, match="diverged at frame 40"):
        states = a.replay_demo_route(FakeNes(start_x=41), route(FRAMES, GOAL_X), stride=8)
    assert states == {}


def _frontier(prob: float = 1.0):
    nes = FakeNes()
    env = a.FrontierStart(nes, prob=prob, demo_routes=[route(FRAMES, GOAL_X)], demo_exact=1.0, warmup_frames=4)
    return nes, env


def test_frontier_start_without_a_cursor_starts_normally() -> None:
    """Demo restarts only happen once the curriculum has set a start frame."""
    nes, env = _frontier()
    _, info = env.reset(seed=0)
    assert info["demo_start"] is False
    assert nes.restored == []


def test_demo_restart_restores_the_state_and_forces_the_warmup_bytes() -> None:
    """The first decision after a restore sees a full stack of real frames, so the route plays on."""
    nes, env = _frontier()
    env.set_demo_cursor("1-1", 24)
    _, info = env.reset(seed=0)
    assert info["frontier_start"] is True
    assert info["demo_start"] is True
    assert nes.restored == [59]
    assert nes.synced == 1
    assert info["x_pos"] == 60

    nes.received.clear()
    for _ in range(4):
        env.step(99)
    assert nes.received == [1, 1, 1, 1]
    assert nes.x == 64

    env.step(2)
    assert nes.received[-1] == 2


def test_demo_outcome_reports_whether_the_route_end_was_reached() -> None:
    """The next reset carries the result of the demo episode, for the curriculum to read."""
    nes, env = _frontier()
    env.set_demo_cursor("1-1", 24)
    env.reset(seed=0)
    for _ in range(4):
        env.step(99)
    for _ in range(8):
        env.step(2)
    assert nes.x == GOAL_X
    _, info = env.reset()
    assert info["demo_outcome"] == {"level": "1-1", "frame": 24, "x": 64, "success": True}
