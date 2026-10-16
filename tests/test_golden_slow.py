"""Full sweeps: every NOOP offset, worker processes and frontier restarts."""

from __future__ import annotations

import pytest

from tests import adapter as a
from tests import golden_io, probes


@pytest.mark.slow
@pytest.mark.emulator
@pytest.mark.parametrize("level", a.LEVELS)
def test_every_noop_offset_matches_the_golden(level: str) -> None:
    """All 31 offsets of one level, through the training env."""
    for offset in golden_io.ALL_OFFSETS:
        golden_io.assert_matches("trajectories", probes.wrapped_trajectory(level, offset), f"{level}/{offset}")


@pytest.mark.slow
@pytest.mark.emulator
def test_every_whole_game_offset_matches_the_golden() -> None:
    """All 31 offsets of the chained whole-game run."""
    for offset in golden_io.ALL_OFFSETS:
        golden_io.assert_matches("game_chains", probes.game_chain(offset), str(offset))


@pytest.mark.slow
@pytest.mark.emulator
def test_vec_env_rollout_matches_the_golden_in_process() -> None:
    """Four workers on mixed levels, scripted actions: seeds, wrapper chain and stacked frames."""
    golden_io.assert_matches("vec_rollout", probes.vec_rollout(use_subproc=False))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.emulator
def test_worker_processes_return_exactly_what_the_in_process_envs_do() -> None:
    """SubprocVecEnv under forkserver re-imports the entry module; the result must not depend on it."""
    in_process = probes.vec_rollout(use_subproc=False)
    subproc = probes.vec_rollout(use_subproc=True)
    assert probes.same_rollout(in_process, subproc)


@pytest.mark.slow
@pytest.mark.emulator
def test_frontier_and_demo_restarts_match_the_golden() -> None:
    """Restore points, forced warm-up bytes and demo outcomes with a fixed curriculum cursor."""
    golden_io.assert_matches("frontier_demo", probes.frontier_demo_restarts())
