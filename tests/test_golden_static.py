"""Surfaces and tables that must not move during the restructure (no emulator needed)."""

from __future__ import annotations

from tests import golden_io, probes

# Offsets (raw idle frames after the reset) whose routes are documented not to reach the flag.
DOCUMENTED_MISSES = {
    "1-1": set(),
    "1-2": {15, 19, 23, 25, 29},
    "1-3": set(range(1, 31, 2)),
    "1-4": {7, 9, 15},
}


def test_public_api_names_signatures_and_constants_are_unchanged() -> None:
    """Parameter names and defaults of every public symbol, and the module constants."""
    golden_io.assert_matches("api", probes.api_surface())


def test_command_line_defaults_are_unchanged() -> None:
    """Every flag and default of train, evaluate, record and find-route."""
    golden_io.assert_matches("cli", probes.cli_surface())


def test_demonstration_routes_are_byte_identical() -> None:
    """A moved or reformatted route file would change what the curriculum replays."""
    golden_io.assert_matches("route_files", probes.route_files())


def test_pure_logic_tables_are_unchanged() -> None:
    """Level parsing, worker assignment, start frames, curriculum walks and episode summaries."""
    golden_io.assert_matches("pure_tables", probes.pure_tables())


def test_captured_noop_offset_outcomes_match_the_documented_ones() -> None:
    """The golden itself reproduces the success sets recorded in the project notes."""
    trajectories = golden_io.load("trajectories")["exact"]
    for level, misses in DOCUMENTED_MISSES.items():
        reached = {
            offset
            for offset in golden_io.ALL_OFFSETS
            if trajectories[f"{level}/{offset}"]["reached_goal"]
        }
        assert set(golden_io.ALL_OFFSETS) - reached == misses, level
