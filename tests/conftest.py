"""Test configuration: headless SDL, and skipping what needs the untracked checkpoint."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``local_golden`` tests when the reference checkpoint is not on disk."""
    from tests import adapter

    if adapter.CHECKPOINT.exists():
        return
    skip = pytest.mark.skip(reason=f"reference checkpoint not found: {adapter.CHECKPOINT}")
    for item in items:
        if "local_golden" in item.keywords:
            item.add_marker(skip)
