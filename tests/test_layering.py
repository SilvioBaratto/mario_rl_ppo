"""The dependency rule, enforced: inner layers never import outer ones, and no two layers
import each other.

Reads every module's imports with ``ast`` instead of trusting the README diagram, so a stray
import fails the build rather than slowly eroding the architecture.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "mario"

TOP_LEVEL = {"__init__": set(), "__main__": {"cli"}}

# Which mario layers each layer may import (itself always included). The adapters in
# infrastructure satisfy what the application asks of them structurally, so neither needs the other.
ALLOWED = {
    "domain": set(),
    "application": {"domain"},
    "infrastructure": {"domain"},
    "cli": {"domain", "application", "infrastructure"},
}

# The domain is plain Python plus numpy: no emulator, framework or file access.
DOMAIN_THIRD_PARTY = {"numpy"}


def _package_imports(path: Path, package: Path = PACKAGE) -> set[str]:
    """Return the top-level layer names a module imports (``mario.x.y`` -> ``x``).

    Covers absolute imports, ``from mario import x`` and relative imports, which are resolved
    against the module's own position inside ``package``.
    """
    tree = ast.parse(path.read_text())
    here = list(path.relative_to(package.parent).with_suffix("").parts)
    root = package.name
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dotted = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - node.level]
                module = base + (node.module.split(".") if node.module else [])
            else:
                module = node.module.split(".") if node.module else []
            # `from mario import cli` names the layer in the alias, not in the module.
            dotted = (
                [module + [alias.name] for alias in node.names] if module == [root] else [module]
            )
        else:
            continue
        for parts in dotted:
            # `from mario import __version__` is a module attribute, not a layer.
            if len(parts) > 1 and parts[0] == root and not parts[1].startswith("_"):
                names.add(parts[1])
    return names


def _external_imports(path: Path) -> set[str]:
    """Return the top-level names of the non-mario, non-stdlib modules a file imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n != PACKAGE.name and n not in sys.stdlib_module_names}


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_what_it_may(layer: str) -> None:
    """Each module in ``layer`` imports only itself and the layers it is allowed to see."""
    for module in sorted((PACKAGE / layer).rglob("*.py")):
        forbidden = _package_imports(module) - ALLOWED[layer] - {layer}
        assert not forbidden, f"{module.relative_to(PACKAGE)} imports {sorted(forbidden)}"


def test_no_two_layers_may_import_each_other() -> None:
    """The allowed graph itself has no cycle, so passing the rule above means none exists."""
    for layer, allowed in ALLOWED.items():
        for other in allowed & set(ALLOWED):
            assert layer not in ALLOWED[other], f"{layer} and {other} may import each other"


def test_domain_depends_on_nothing_but_numpy() -> None:
    """Pure rules must stay importable and testable without the emulator or any framework."""
    for module in sorted((PACKAGE / "domain").rglob("*.py")):
        extra = _external_imports(module) - DOMAIN_THIRD_PARTY
        assert not extra, f"{module.relative_to(PACKAGE)} imports {sorted(extra)}"


def test_top_level_modules_import_only_what_they_may() -> None:
    """``__main__`` only reaches the CLI, and no top-level module hides outside the rule."""
    modules = {p.stem for p in PACKAGE.glob("*.py")}
    assert modules == set(TOP_LEVEL)
    for name, allowed in TOP_LEVEL.items():
        forbidden = _package_imports(PACKAGE / f"{name}.py") - allowed
        assert not forbidden, f"{name}.py imports {sorted(forbidden)}"


def test_every_layer_is_covered() -> None:
    """A new top-level package cannot dodge the rule by not being listed."""
    packages = {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    assert packages == set(ALLOWED)


@pytest.mark.parametrize(
    "source",
    [
        "from mario.infrastructure import wrappers",
        "import mario.infrastructure.wrappers",
        "from mario import infrastructure",
        "from ..infrastructure import wrappers",
        "from .. import infrastructure",
    ],
)
def test_every_import_style_is_seen(tmp_path: Path, source: str) -> None:
    """Absolute, package-alias and relative imports of a layer are all detected."""
    module = tmp_path / "mario" / "domain" / "rogue.py"
    module.parent.mkdir(parents=True)
    module.write_text(source + "\n")
    assert "infrastructure" in _package_imports(module, package=tmp_path / "mario")
