"""The import rules from CLAUDE.md, enforced rather than described.

`analysis/` measures a finished world.  If a stage could import it, a stage could read its
own report card and start tuning itself against a metric, and the pipeline would stop
being a straight line from seed to world.  The rule was stated in three places and checked
in none, which is the state a rule is in just before someone breaks it.

`render/` is here for the same reason and is the older rule: a stage that could draw would
make the debug viewer load-bearing.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent / "worldgen"

# package -> the sibling packages it may not import, and why.
_FORBIDDEN = {
    "stages": {
        "analysis": "a stage must not read its own report card",
        "render": "a stage must not draw",
    },
    "core": {
        "analysis": "core is data types and the pipeline, and measures nothing",
        "render": "core must not draw",
        "export": "core does no file I/O",
    },
    "analysis": {
        "render": "analysis returns numbers, not pictures",
        "export": "analysis does no file I/O",
        "stages": "analysis reads a finished world, it does not make one",
    },
}


def _imported_packages(source: str, package: str) -> set[str]:
    """Every `worldgen.<package>` the source pulls in, however it spells the import.

    Takes text and the name of the package the file lives in, not a path, so the guard
    below can be shown to fail without a probe file being written into the source tree.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "worldgen" and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("worldgen."):
                    found.add(node.module.split(".")[1])
            elif node.level == 1:
                # `from .x import y` — a sibling inside this package, not a layer hop.
                continue
            else:
                # `from ..x import y` — up to worldgen, then back down into a package.
                found.add(node.module.split(".")[0] if node.module else package)
    return found


def _offences(package: str, sources: dict[str, str]) -> list[str]:
    """Which of *sources* — name to text — break *package*'s import rules."""
    banned = _FORBIDDEN[package]
    return [
        f"{name} imports {imported} — {banned[imported]}"
        for name, source in sorted(sources.items())
        for imported in sorted(_imported_packages(source, package) & banned.keys())
    ]


def _sources_in(package: str) -> dict[str, str]:
    return {
        str(path.relative_to(_ROOT.parent)): path.read_text(encoding="utf-8")
        for path in sorted((_ROOT / package).rglob("*.py"))
    }


@pytest.mark.parametrize("package", sorted(_FORBIDDEN))
def test_a_layer_does_not_import_the_layers_above_it(package):
    offences = _offences(package, _sources_in(package))
    assert not offences, "\n".join(offences)


@pytest.mark.parametrize(
    "spelling",
    [
        "from ..analysis import drainage_metrics",
        "from worldgen.analysis import drainage_metrics",
        "import worldgen.analysis",
        "def run(self):\n    from ..analysis.drainage import build_network",
    ],
)
def test_the_check_would_catch_a_violation(spelling):
    """A guard that cannot fail is not a guard.

    It has to see every spelling, the deferred import inside a method included — which is
    how one would really arrive, since that is the shape the codebase already uses to dodge
    an import cycle.
    """
    assert _offences("stages", {"stages/probe.py": spelling}) == [
        "stages/probe.py imports analysis — a stage must not read its own report card"
    ]


def test_a_sibling_import_is_not_a_layer_hop():
    """`from .hex import HexCoord` inside core is core using core."""
    assert _offences("core", {"core/probe.py": "from .hex import HexCoord"}) == []


def test_render_may_import_analysis():
    """The rule is one-directional: the drainage plate depends on being allowed to measure."""
    source = (_ROOT / "render" / "debug_viewer.py").read_text(encoding="utf-8")
    assert "analysis" in _imported_packages(source, "render")
