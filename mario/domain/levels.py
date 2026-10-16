"""Level lists, and which worker plays which level."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

EnvIds = str | Sequence[str]


def parse_levels(spec: str) -> list[str]:
    """Convert a level list such as '1-1,1-2,1-3' into env ids.

    Args:
        spec: Comma-separated WORLD-STAGE pairs (world 1-8, stage 1-4), or
            'game' for the whole game from 1-1 ('SuperMarioBros-v0').

    Returns:
        Env ids in the given order, e.g. ['SuperMarioBros-1-1-v0', ...].

    Raises:
        ValueError: If the list is empty or an entry is malformed or out of range.
    """
    env_ids = []
    for item in (part.strip() for part in spec.split(",")):
        if item == "game":
            env_ids.append("SuperMarioBros-v0")
            continue
        match = re.fullmatch(r"([1-8])-([1-4])", item)
        if not match:
            raise ValueError(
                f"Invalid level {item!r}; expected WORLD-STAGE like 1-2 (world 1-8, stage 1-4) "
                f"or 'game'"
            )
        env_ids.append(f"SuperMarioBros-{match.group(1)}-{match.group(2)}-v0")
    if not env_ids:
        raise ValueError("Level list is empty")
    return env_ids


def assign_levels(env_ids: EnvIds, num_envs: int) -> list[str]:
    """Return the env id each of ``num_envs`` workers plays, round-robin.

    Args:
        env_ids: One env id, or several to train on together.
        num_envs: Number of parallel workers.

    Returns:
        A list of length ``num_envs``; worker i plays ``env_ids[i % len(env_ids)]``.
        When ``num_envs`` is not a multiple of the level count, the first
        levels get one extra worker.
    """
    ids = [env_ids] if isinstance(env_ids, str) else list(env_ids)
    return [ids[rank % len(ids)] for rank in range(num_envs)]


def level_key(world: Any, stage: Any) -> str:
    """Return the 'W-S' label used to address a level across processes, e.g. '1-3'."""
    return f"{world}-{stage}"


def is_single_stage(env_id: str) -> bool:
    """Tell whether ``env_id`` plays one stage (``SuperMarioBros-W-S-vN``), not the whole game."""
    return re.fullmatch(r"SuperMarioBros-\d-\d-v\d+", env_id) is not None
