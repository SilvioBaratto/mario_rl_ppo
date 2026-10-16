"""Reading, writing and comparing the golden traces in ``tests/golden``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.probes import platform_tag

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Every NOOP offset the demonstration routes are documented against.
ALL_OFFSETS = tuple(range(31))
# The offsets the pull-request CI replays: 0 and 2 for every level, plus odd offset 1 for 1-3,
# whose route is documented to die near x=1724 after an odd offset.
CI_OFFSETS: dict[str, tuple[int, ...]] = {
    "1-1": (0, 2),
    "1-2": (0, 2),
    "1-3": (0, 1, 2),
    "1-4": (0, 2),
}
GAME_CI_OFFSETS = (0, 2)


def load(name: str) -> dict[str, Any]:
    """Return the golden called ``name``."""
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


def save(name: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` as the golden called ``name``."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    (GOLDEN_DIR / f"{name}.json").write_text(text)


def comparable_here(golden: dict[str, Any]) -> bool:
    """Tell whether the platform-bound values of ``golden`` were captured on this platform."""
    return golden["platform"] == platform_tag()


def assert_matches(name: str, got: dict[str, Any], key: str | None = None) -> None:
    """Assert that a probe result equals the golden ``name`` (or its entry ``key``).

    Exact values must always match. Platform-bound values are compared only when this machine
    is the one that captured them and the capture kept them.
    """
    golden = load(name)
    exact, bound = golden["exact"], golden["platform_bound"]
    if key is not None:
        exact, bound = exact[key], bound.get(key, {})
    assert got["exact"] == exact
    if bound and comparable_here(golden):
        assert got["platform_bound"] == bound
