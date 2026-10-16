"""Reading an episode's story from its step infos: why it ended, levels reached, lives lost."""

from __future__ import annotations

from typing import Any


def episode_end_reason(info: dict[str, Any]) -> str:
    """Describe why an episode that did not reach the flag ended.

    Args:
        info: The final step's info dict, as returned by the VecEnv on ``done``.

    Returns:
        'Stalled (no progress)' for the stall cutoff, 'Time limit' for the
        TimeLimit wrapper, otherwise 'Died'.
    """
    if info.get("stalled"):
        return "Stalled (no progress)"
    if info.get("TimeLimit.truncated"):
        return "Time limit"
    return "Died"


class GameProgress:
    """Follow one episode through its step infos: levels reached and lives lost.

    Meant for whole-game ids such as 'SuperMarioBros-v0', where one episode
    spans levels and lives; on a single-stage id it sees one level and at
    most one lost life. A life counts as lost when ``life`` drops or the
    episode ends in a death (the last one wraps ``life`` 0 -> 255);
    truncations (stall cutoff, time limit) are not deaths.

    Attributes:
        levels: (world, stage) pairs in the order they were played.
        lives_lost: Lives lost so far.
        clean_levels: Levels cleared before the first life was lost, or None
            while no life has been lost.
    """

    def __init__(self):
        self.levels: list[tuple[Any, Any]] = []
        self.lives_lost = 0
        self.clean_levels: int | None = None
        self._life = None
        self._ended_on_flag = False

    def update(self, info: dict[str, Any], done: bool) -> None:
        """Account for one step.

        Args:
            info: The step's info dict (for the final step, the VecEnv's terminal info).
            done: Whether the episode ended on this step.
        """
        level = (info.get("world"), info.get("stage"))
        if not self.levels or level != self.levels[-1]:
            self.levels.append(level)
        life = info.get("life")
        flag = bool(info.get("flag_get"))
        lost = self._life is not None and life is not None and life < self._life
        if done and not flag and episode_end_reason(info) == "Died":
            lost = True
        if lost:
            self.lives_lost += 1
            if self.clean_levels is None:
                self.clean_levels = len(self.levels) - 1
        self._life = life
        self._ended_on_flag = done and flag

    @property
    def levels_cleared(self) -> int:
        """Return the levels finished: every level change, plus a final flag."""
        return max(len(self.levels) - 1, 0) + int(self._ended_on_flag)

    def summary(self) -> str:
        """Describe the episode, e.g. 'reached 2-1, cleared 4 levels, 0 lives lost'."""
        world, stage = self.levels[-1] if self.levels else (None, None)
        clean = self.levels_cleared if self.clean_levels is None else self.clean_levels
        return (
            f"reached {world}-{stage}, cleared {self.levels_cleared} level(s), "
            f"{self.lives_lost} li{'fe' if self.lives_lost == 1 else 'ves'} lost, "
            f"{clean} level(s) cleared before the first life lost"
        )
