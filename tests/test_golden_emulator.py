"""Short emulator checks: raw routes, the wrapped training env, the whole game and cloning data."""

from __future__ import annotations

import pytest

from tests import golden_io, probes

pytestmark = pytest.mark.emulator

TRAJECTORY_CASES = [
    (level, offset) for level, offsets in golden_io.CI_OFFSETS.items() for offset in offsets
]


def test_routes_replay_on_the_bare_emulator_as_recorded() -> None:
    """Waypoint x, RAM hashes every 60 frames and the replay start states are unchanged."""
    golden_io.assert_matches("raw_routes", probes.raw_route_replay())


@pytest.mark.parametrize(("level", "offset"), TRAJECTORY_CASES)
def test_route_actions_through_the_training_env_match_the_golden(level: str, offset: int) -> None:
    """Rewards, infos and observations per agent step, after a NOOP offset."""
    golden_io.assert_matches(
        "trajectories", probes.wrapped_trajectory(level, offset), f"{level}/{offset}"
    )


@pytest.mark.parametrize("offset", golden_io.GAME_CI_OFFSETS)
def test_whole_game_chain_matches_the_golden(offset: int) -> None:
    """Level transitions, idling through level ends, lives and observations of the chain."""
    golden_io.assert_matches("game_chains", probes.game_chain(offset), str(offset))


def test_behaviour_cloning_datasets_match_the_golden() -> None:
    """Pair counts, action bytes, skipped-offset warnings and the observation hash."""
    golden_io.assert_matches("bc_datasets", probes.bc_datasets())
