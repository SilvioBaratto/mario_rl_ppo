"""The contract of the shipped demonstration routes."""

from __future__ import annotations

import pytest
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from tests import adapter as a


def _joypad_byte(action: int) -> int:
    return sum(JoypadSpace._button_map[button] for button in SIMPLE_MOVEMENT[action])


@pytest.fixture(params=a.LEVELS)
def level_route(request: pytest.FixtureRequest) -> tuple[str, dict]:
    """Return a level label and its shipped route."""
    level = request.param
    return level, a.load_demo_route(f"SuperMarioBros-{level[0]}-{level[2]}-v0")


def test_route_is_recorded_on_the_level_it_is_named_after(level_route) -> None:
    """The file name and the env id inside it must agree."""
    level, route = level_route
    assert a.route_level(route) == level
    assert (a.DEMO_ROUTES_DIR / f"{level}.json").is_file()


def test_route_frames_are_the_agent_actions_expanded_to_joypad_bytes(level_route) -> None:
    """Both forms are the same trajectory: each action holds its buttons for FRAME_SKIP frames."""
    _, route = level_route
    expanded = [_joypad_byte(action) for action in route["agent_actions"] for _ in range(a.FRAME_SKIP)]
    assert route["frames"] == expanded[: len(route["frames"])]


def test_route_is_cut_at_the_flag_frame_within_its_last_agent_step(level_route) -> None:
    """The last agent step may be cut short, never a whole step more."""
    _, route = level_route
    steps = len(route["agent_actions"])
    assert a.FRAME_SKIP * (steps - 1) < len(route["frames"]) <= a.FRAME_SKIP * steps


def test_route_waypoints_advance_and_the_last_is_the_final_frame(level_route) -> None:
    """The curriculum goal is the last waypoint, reached on the route's final frame."""
    _, route = level_route
    frames = [w["frame"] for w in route["waypoints"]]
    xs = [w["x"] for w in route["waypoints"]]
    assert frames == sorted(set(frames))
    assert xs == sorted(set(xs))
    assert frames[-1] == len(route["frames"])


def test_route_actions_are_valid_simple_movement_indices(level_route) -> None:
    """Behaviour cloning feeds these straight to the policy's action head."""
    _, route = level_route
    assert set(route["agent_actions"]) <= set(range(len(SIMPLE_MOVEMENT)))
    assert all(0 <= byte <= 255 for byte in route["frames"])
