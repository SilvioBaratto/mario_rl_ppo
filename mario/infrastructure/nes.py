"""The part of the nes-py emulator that this project touches, described for the type checker."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class NesEmulator(Protocol):
    """gym-super-mario-bros' raw env, as returned by ``env.unwrapped``.

    ``_x_position``, ``_is_dying``, ``_get_info`` and ``_did_reset`` are private in the library.
    Frontier snapshots, route replay and the route search depend on them, so this protocol is the
    list of what an upgrade of the emulator must keep.
    """

    ram: np.ndarray
    screen: np.ndarray
    _is_dying: bool

    @property
    def _x_position(self) -> int:
        """Mario's x position in the level."""
        ...

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Start the level again."""
        ...

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance one frame with the joypad byte ``action``."""
        ...

    def dump_state(self) -> Any:
        """Snapshot the whole emulator."""
        ...

    def load_state(self, state: Any) -> None:
        """Restore a snapshot taken by ``dump_state``."""
        ...

    def close(self) -> None:
        """Release the emulator."""
        ...

    def _did_reset(self) -> None:
        """Resync the Python-side reward trackers with the emulator's RAM."""
        ...

    def _get_info(self) -> dict[str, Any]:
        """Return the info dict for the current frame."""
        ...
