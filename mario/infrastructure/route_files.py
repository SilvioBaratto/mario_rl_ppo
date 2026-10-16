"""Reading the recorded demonstration routes that ship inside the package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEMO_ROUTES_DIR = Path(__file__).with_name("demo_routes")


def load_demo_route(env_id: str) -> dict[str, Any] | None:
    """Return the recorded demonstration route for a single-stage env id, if one exists.

    Routes live in ``demo_routes/<world>-<stage>.json``: frame-level joypad
    bytes from a fresh, NOOP-free reset plus waypoints (frame index and the
    x position the replay must reach there). An optional ``agent_actions``
    list gives the same route as SIMPLE_MOVEMENT indices per agent step,
    for behaviour cloning.

    Args:
        env_id: e.g. 'SuperMarioBros-1-3-v0'.

    Returns:
        The parsed route, or None when the id is not single-stage or has no route.
    """
    match = re.fullmatch(r"SuperMarioBros-(\d)-(\d)-v\d+", env_id)
    if not match:
        return None
    path = DEMO_ROUTES_DIR / f"{match.group(1)}-{match.group(2)}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
