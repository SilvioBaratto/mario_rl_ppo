"""The ``find-route`` command: search the emulator for a demonstration route."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os

from mario.cli.common import configure_sdl

DESCRIPTION = "Find a demonstration route through a level by emulator search"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's flags to ``parser``."""
    parser.add_argument("--level", type=str, required=True, help="Level to search, e.g. 1-4")
    parser.add_argument(
        "--searches",
        type=int,
        default=20,
        help="Independent searches (seeds); the most NOOP-robust flag route is kept",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=60_000,
        help="Agent steps per search; each spends all of them and returns its shortest flag route",
    )
    parser.add_argument("--workers", type=int, default=4, help="Searches run in parallel")
    parser.add_argument(
        "--max-noop",
        type=int,
        default=30,
        help="Robustness is checked for NOOP offsets 0..this (match --noop-max)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Route file (default: mario/environments/demo_routes/<W>-<S>.json)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing route file")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first search")


def run(args: argparse.Namespace) -> int:
    """Run the searches, keep the most NOOP-robust flag route and write it."""
    configure_sdl()
    from mario.domain.levels import parse_levels
    from mario.infrastructure.route_files import DEMO_ROUTES_DIR
    from mario.infrastructure.route_search import build_route, flag_offsets, search_worker

    env_id = parse_levels(args.level)[0]
    world, stage = args.level.strip().split("-")
    output = args.output or str(DEMO_ROUTES_DIR / f"{world}-{stage}.json")
    if os.path.exists(output) and not args.force:
        raise SystemExit(
            f"{output} exists; pass --force to replace it or --output to write elsewhere"
        )

    jobs = [(env_id, args.budget, args.seed + i) for i in range(args.searches)]
    found = []
    with multiprocessing.get_context("forkserver").Pool(args.workers) as pool:
        for seed, actions, seconds in pool.imap_unordered(search_worker, jobs):
            status = f"flag in {len(actions)} steps" if actions else "no flag"
            print(f"search seed {seed}: {status} ({seconds:.0f}s)", flush=True)
            if actions:
                found.append((seed, actions))
        if not found:
            raise SystemExit("No search reached the flag; raise --budget or --searches")
        # Seed order keeps the choice between equally good routes independent of finishing order.
        found.sort()
        print(f"Checking NOOP offsets 0-{args.max_noop} for {len(found)} routes...", flush=True)
        offsets = pool.starmap(
            flag_offsets, [(env_id, actions, args.max_noop) for _, actions in found]
        )
    ranked = sorted(
        zip(offsets, (actions for _, actions in found)),
        key=lambda item: (-len(item[0]), len(item[1])),
    )
    ok_offsets, actions = ranked[0]
    route = build_route(env_id, actions, ok_offsets, args.max_noop, args.searches)
    with open(output, "w") as f:
        json.dump(route, f, indent=1)
    print(
        f"Saved {output}: {len(actions)} agent steps, {len(route['frames'])} frames, "
        f"flag at x={route['waypoints'][-1]['x']}, reaches the flag after "
        f"{len(ok_offsets)}/{args.max_noop + 1} NOOP offsets"
    )
    return 0
