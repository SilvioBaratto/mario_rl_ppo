"""
Search the emulator for a demonstration route through a level and save it to demo_routes/
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from environments.env_factory import DEMO_ROUTES_DIR, FRAME_SKIP, SMB1_FLOAT_STATE, parse_levels

# SMB1 RAM: area type (0 water, 1 ground, 2 underground, 3 castle) and the
# pointer of the area being played; together they identify an area of a level.
SMB1_AREA_TYPE = 0x074E
SMB1_AREA_POINTER = 0x0750

# Exploration favours moving right and jumping; NOOP, A-only and LEFT stay
# possible for waiting out fire bars or backing off a ledge.
ACTION_WEIGHTS = np.array([0.04, 0.12, 0.22, 0.2, 0.3, 0.07, 0.05])
STICKY = 0.85
CELL_PX = 16
WAYPOINT_EVERY = 240


def parse_args():
    """Parse the route-search command line."""
    parser = argparse.ArgumentParser(
        description="Find a demonstration route through a level by emulator search"
    )
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
        help="Route file (default: src/environments/demo_routes/<W>-<S>.json)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing route file")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first search")
    return parser.parse_args()


def _make_raw_env(env_id):
    """Return (raw gym-super-mario-bros env, SIMPLE_MOVEMENT index -> joypad byte)."""
    nes = gym_super_mario_bros.make(env_id).unwrapped
    return nes, dict(JoypadSpace(nes, SIMPLE_MOVEMENT)._action_map)


def _agent_step(nes, byte):
    for _ in range(FRAME_SKIP):
        _, _, terminated, truncated, info = nes.step(byte)
        if terminated or truncated:
            return info, True
    return info, False


def go_explore(env_id, budget, seed):
    """Search for the shortest flag-reaching action sequence from a NOOP-free reset.

    Go-Explore style: an archive keeps, per ``CELL_PX`` x ``CELL_PX`` cell of
    Mario's position within an area, the shortest known action sequence
    reaching it and the emulator state there. Each iteration restores a cell
    (rarely tried cells and the most advanced 10% are favoured) and plays
    random sticky actions. Leaving the level other than by its flag (a warp
    zone) counts as a dead end.

    The search spends the whole budget instead of stopping at the first flag:
    the first route carries the detours of early exploration (1-2's spent 74
    steps wandering among the block columns at its start, and a policy cloned
    from it looped there), while later flags extend archive prefixes that
    have since been shortened.

    Args:
        env_id: Single-stage env id.
        budget: Agent steps to spend.
        seed: Seed of the exploration RNG.

    Returns:
        The shortest found SIMPLE_MOVEMENT indices ending on the frame that
        reports ``flag_get`` (world-over in castles), or None when no flag was
        reached within the budget.
    """
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    nes, action_map = _make_raw_env(env_id)
    nes.reset()
    info = nes._get_info()
    level = (info["world"], info["stage"])
    # x restarts in every area (e.g. 1-2's outdoor exit after the underground
    # part), so cells and progress are per area; areas found later rank higher.
    area_rank = {}

    def area_of():
        """Return the (area type, area pointer) Mario is in."""
        return int(nes.ram[SMB1_AREA_TYPE]), int(nes.ram[SMB1_AREA_POINTER])

    def progress(info):
        """Return how far along the level ``info`` is: area discovery rank first, then x."""
        return area_rank.setdefault(area_of(), len(area_rank)) * 100_000 + info["x_pos"]

    def cell_of(info):
        """Return the archive cell of Mario's position in ``info``."""
        return area_of() + (info["x_pos"] // CELL_PX, info["y_pos"] // CELL_PX)

    archive = {
        cell_of(info): {"state": nes.dump_state(), "actions": [], "x": progress(info), "visits": 0}
    }
    steps = 0
    best = None
    while steps < budget:
        cells = list(archive)
        xs = np.array([archive[c]["x"] for c in cells], dtype=float)
        weights = 1.0 / np.sqrt(np.array([archive[c]["visits"] for c in cells], dtype=float) + 1.0)
        if rng.random() < 0.5:
            weights = weights * (xs >= np.quantile(xs, 0.9))
        entry = archive[cells[int(rng.choice(len(cells), p=weights / weights.sum()))]]
        entry["visits"] += 1
        nes.load_state(entry["state"])
        actions = list(entry["actions"])
        action = int(rng.choice(len(ACTION_WEIGHTS), p=ACTION_WEIGHTS))
        for _ in range(int(rng.integers(8, 48))):
            if rng.random() > STICKY:
                action = int(rng.choice(len(ACTION_WEIGHTS), p=ACTION_WEIGHTS))
            info, done = _agent_step(nes, action_map[action])
            steps += 1
            actions.append(action)
            if info.get("flag_get"):
                if best is None or len(actions) < len(best):
                    best = list(actions)
                break
            if done or (info["world"], info["stage"]) != level:
                break
            cell = cell_of(info)
            old = archive.get(cell)
            if old is None or len(actions) < len(old["actions"]):
                archive[cell] = {
                    "state": nes.dump_state(),
                    "actions": list(actions),
                    "x": progress(info),
                    "visits": old["visits"] if old else 0,
                }
    return best


def _search_worker(job):
    env_id, budget, seed = job
    start = time.time()
    return seed, go_explore(env_id, budget, seed), time.time() - start


def flag_offsets(env_id, actions, max_noop):
    """Return the NOOP offsets after which ``actions`` still reach the flag.

    Training resets idle a random number of frames (``NoopReset``), and
    behaviour cloning replays a route after every offset, keeping only those
    where it still reaches the flag.

    Args:
        env_id: Single-stage env id.
        actions: SIMPLE_MOVEMENT indices from a reset.
        max_noop: Largest offset checked.

    Returns:
        The offsets in 0..``max_noop`` whose replay ends on ``flag_get``.
    """
    nes, action_map = _make_raw_env(env_id)
    ok = []
    for noop in range(max_noop + 1):
        nes.reset()
        for _ in range(noop):
            nes.step(0)
        for action in actions:
            info, done = _agent_step(nes, action_map[action])
            if done:
                break
        if info.get("flag_get"):
            ok.append(noop)
    return ok


def build_route(env_id, actions, ok_offsets, max_noop, searches):
    """Expand agent actions into a route dict in the ``load_demo_route`` format.

    ``frames`` is every action repeated ``FRAME_SKIP`` times and cut at the
    flag frame, so the frame-level and agent-level forms are one trajectory.
    Waypoints are the first grounded frame after every ``WAYPOINT_EVERY``
    frames, plus the flag frame, whose x is the reverse curriculum's goal.

    Args:
        env_id: Single-stage env id.
        actions: Flag-reaching SIMPLE_MOVEMENT indices from a NOOP-free reset.
        ok_offsets: Result of ``flag_offsets``, recorded in the description.
        max_noop: Largest offset that was checked.
        searches: Number of searches the route was selected from.

    Returns:
        The route dict, ready to be written as JSON.

    Raises:
        RuntimeError: If replaying ``actions`` from a fresh reset does not reach the flag.
    """
    nes, action_map = _make_raw_env(env_id)
    nes.reset()
    frames, xs, grounded = [], [], []
    info = {}
    for action in actions:
        for _ in range(FRAME_SKIP):
            _, _, terminated, truncated, info = nes.step(action_map[action])
            frames.append(action_map[action])
            xs.append(int(nes._x_position))
            grounded.append(nes.ram[SMB1_FLOAT_STATE] == 0 and not nes._is_dying)
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    if not info.get("flag_get"):
        raise RuntimeError(
            f"{env_id}: route does not reach the flag on replay (x={info.get('x_pos')})"
        )

    waypoints = []
    for target in range(WAYPOINT_EVERY, len(frames) - WAYPOINT_EVERY // 2, WAYPOINT_EVERY):
        frame = next((f for f in range(target, len(frames)) if grounded[f - 1]), None)
        if frame is not None:
            waypoints.append({"frame": frame, "x": xs[frame - 1]})
    waypoints.append({"frame": len(frames), "x": xs[-1], "note": "flag_get"})
    failing = [k for k in range(max_noop + 1) if k not in ok_offsets]
    return {
        "env_id": env_id,
        "description": (
            f"Found by find_route.py: Go-Explore style emulator search (archive of states per {CELL_PX}x{CELL_PX} "
            f"px cell, random sticky actions, shortest path kept per cell, shortest flag route within the budget) "
            f"from a fresh reset with no NOOPs; the most offset-robust flag route of {searches} searches. "
            f"frames is agent_actions expanded to {FRAME_SKIP} joypad bytes per step, cut at the flag frame. "
            f"agent_actions reaches the flag after {len(ok_offsets)} of the {max_noop + 1} NOOP offsets 0-{max_noop}"
            + (f" (not {', '.join(map(str, failing))})." if failing else ".")
        ),
        "frames": frames,
        "waypoints": waypoints,
        "agent_actions": [int(a) for a in actions],
    }


def main():
    """Run the searches, keep the most NOOP-robust flag route and write it."""
    args = parse_args()
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
        for seed, actions, seconds in pool.imap_unordered(_search_worker, jobs):
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


if __name__ == "__main__":
    main()
