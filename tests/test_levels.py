"""Level lists, worker assignment and route lookup."""

from __future__ import annotations

import pytest

from tests import adapter as a


def test_parse_levels_builds_single_stage_ids_in_order() -> None:
    """Each WORLD-STAGE pair becomes its gym-super-mario-bros id."""
    assert a.parse_levels("1-3,1-1") == ["SuperMarioBros-1-3-v0", "SuperMarioBros-1-1-v0"]


def test_parse_levels_maps_game_to_the_whole_game_id() -> None:
    """The word 'game' selects the one id that spans every level and life."""
    assert a.parse_levels("game") == ["SuperMarioBros-v0"]


def test_parse_levels_tolerates_spaces_around_entries() -> None:
    """Spaces next to commas are ignored."""
    assert a.parse_levels(" 1-1 , game ") == ["SuperMarioBros-1-1-v0", "SuperMarioBros-v0"]


@pytest.mark.parametrize("bad", ["", "9-1", "0-1", "1-5", "1_1", "1-1,", "x"])
def test_parse_levels_rejects_malformed_or_out_of_range_entries(bad: str) -> None:
    """World must be 1-8, stage 1-4, and no entry may be empty."""
    with pytest.raises(ValueError, match="Invalid level"):
        a.parse_levels(bad)


def test_assign_levels_round_robins_and_weights_repeated_ids() -> None:
    """Worker i plays ids[i % len(ids)], so a repeated id gets proportionally more workers."""
    assert a.assign_levels(["a", "b", "c"], 8) == ["a", "b", "c", "a", "b", "c", "a", "b"]
    assert a.assign_levels(["a", "a", "b"], 5) == ["a", "a", "b", "a", "a"]


def test_assign_levels_accepts_a_single_id() -> None:
    """A plain string means every worker plays that level."""
    assert a.assign_levels("x", 3) == ["x", "x", "x"]


def test_is_single_stage_tells_levels_from_the_whole_game() -> None:
    """Only ids with a world and stage count as single-stage."""
    assert a.is_single_stage("SuperMarioBros-1-1-v0")
    assert not a.is_single_stage("SuperMarioBros-v0")


def test_level_key_joins_world_and_stage() -> None:
    """The key is the 'W-S' label used across processes."""
    assert a.level_key(1, 3) == "1-3"


def test_load_demo_route_returns_none_without_a_route() -> None:
    """The whole game and levels without a route have nothing to load."""
    assert a.load_demo_route("SuperMarioBros-v0") is None
    assert a.load_demo_route("SuperMarioBros-1-5-v0") is None
    assert a.load_demo_route("not-an-env-id") is None


def test_load_demo_route_reads_the_shipped_route_for_a_level() -> None:
    """A level with a route yields its parsed JSON and the label of its level."""
    route = a.load_demo_route("SuperMarioBros-1-3-v0")
    assert route is not None
    assert a.route_level(route) == "1-3"


def test_route_level_rejects_the_whole_game_id() -> None:
    """A route must be recorded on one level."""
    with pytest.raises(ValueError, match="not a single-stage id"):
        a.route_level({"env_id": "SuperMarioBros-v0"})
