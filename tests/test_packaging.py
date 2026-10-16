"""The package metadata: one set of pins, the version, and the data files a wheel must carry."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import mario

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _name(requirement: str) -> str:
    """Return the normalised distribution name of a requirement line."""
    return re.split(r"[\[=<>!~ ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _pin(requirement: str) -> str | None:
    """Return the exact version a requirement line pins, or None."""
    match = re.search(r"==\s*([\w.+]+)", requirement)
    return match.group(1) if match else None


def _requirements() -> dict[str, str | None]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return {_name(line): _pin(line) for line in lines if line.strip() and not line.startswith("#")}


def test_every_runtime_dependency_is_pinned_exactly() -> None:
    """Checkpoints and goldens depend on exact library versions, so no floats are allowed."""
    for requirement in PYPROJECT["project"]["dependencies"]:
        assert _pin(requirement), f"{requirement} is not pinned with =="


def test_requirements_txt_carries_the_same_pins_as_pyproject() -> None:
    """The Docker image and CI install from requirements.txt, so it must agree with pyproject."""
    pinned = _requirements()
    for requirement in PYPROJECT["project"]["dependencies"]:
        assert pinned.get(_name(requirement)) == _pin(requirement), requirement


def test_version_is_the_installed_one_or_marked_unknown() -> None:
    """A checkout that was never installed reports 0+unknown instead of failing."""
    assert mario.__version__ in {PYPROJECT["project"]["version"], "0+unknown"}


def test_package_data_glob_matches_the_shipped_routes() -> None:
    """If the glob matched nothing, a wheel would ship without routes and training would degrade."""
    pattern = PYPROJECT["tool"]["setuptools"]["package-data"]["mario"][0]
    shipped = sorted(p.name for p in (ROOT / "mario").glob(pattern))
    assert shipped == ["1-1.json", "1-2.json", "1-3.json", "1-4.json"]


@pytest.mark.integration
def test_built_wheel_contains_the_demonstration_routes(tmp_path: Path) -> None:
    """The routes are package data; a wheel without them silently trains without demonstrations."""
    pytest.importorskip("pip")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("mario_rl_ppo-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    for level in ("1-1", "1-2", "1-3", "1-4"):
        assert f"mario/infrastructure/demo_routes/{level}.json" in names
