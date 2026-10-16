"""Capture the golden traces from the code as it stands.

Run from the repository root with the pinned environment::

    python -m tests.capture_golden             # every golden
    python -m tests.capture_golden api cli     # only the named ones

Each golden is produced twice and written only when both runs agree, so a golden is never
a coin flip. Never run this inside a refactor commit: a golden that has to change is a
behaviour change and gets its own commit with a written reason.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

from tests import adapter as a
from tests import golden_io, probes

Producer = Callable[[], dict[str, Any]]


def _keyed(keys: list[str], one: Callable[[str], dict[str, Any]]) -> Producer:
    def run() -> dict[str, Any]:
        """Run the probe for every key and group the results by key."""
        outs = {key: one(key) for key in keys}
        return {
            "exact": {k: v["exact"] for k, v in outs.items()},
            "platform_bound": {k: v["platform_bound"] for k, v in outs.items()},
        }

    return run


def _trajectory(key: str) -> dict[str, Any]:
    level, offset = key.split("/")
    return probes.wrapped_trajectory(level, int(offset))


def _vec_rollout() -> dict[str, Any]:
    in_process = probes.vec_rollout(use_subproc=False)
    subproc = probes.vec_rollout(use_subproc=True)
    if not probes.same_rollout(in_process, subproc):
        raise SystemExit("vec_rollout: SubprocVecEnv differs from DummyVecEnv")
    return in_process


PRODUCERS: dict[str, Producer] = {
    "api": probes.api_surface,
    "cli": probes.cli_surface,
    "route_files": probes.route_files,
    "pure_tables": probes.pure_tables,
    "raw_routes": probes.raw_route_replay,
    "trajectories": _keyed(
        [f"{lv}/{off}" for lv in a.LEVELS for off in golden_io.ALL_OFFSETS], _trajectory
    ),
    "game_chains": _keyed(
        [str(off) for off in golden_io.ALL_OFFSETS], lambda k: probes.game_chain(int(k))
    ),
    "vec_rollout": _vec_rollout,
    "frontier_demo": probes.frontier_demo_restarts,
    "bc_datasets": probes.bc_datasets,
    "checkpoint": probes.checkpoint_contract,
    "policy": probes.policy_behaviour,
    "train_smoke": probes.train_smoke,
    "evaluate_cli": probes.evaluate_cli,
}


def main(argv: list[str]) -> int:
    """Capture the goldens named in ``argv`` (all when empty); return the exit code."""
    names = argv or list(PRODUCERS)
    unknown = [n for n in names if n not in PRODUCERS]
    if unknown:
        print(f"unknown goldens: {unknown}; choose from {list(PRODUCERS)}", file=sys.stderr)
        return 2
    for name in names:
        started = time.monotonic()
        first = PRODUCERS[name]()
        second = PRODUCERS[name]()
        elapsed = time.monotonic() - started
        if first["exact"] != second["exact"]:
            print(f"{name}: NONDETERMINISTIC exact values (A/A mismatch), not written", file=sys.stderr)
            return 1
        bound_stable = first["platform_bound"] == second["platform_bound"]
        if not bound_stable:
            print(f"{name}: platform-bound values differ between runs, dropped", file=sys.stderr)
            first["platform_bound"] = {}
        golden_io.save(name, {"platform": probes.platform_tag(), **first})
        print(f"{name}: written ({elapsed:.0f}s for two runs, platform-bound "
              f"{'stable' if bound_stable else 'DROPPED'})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
