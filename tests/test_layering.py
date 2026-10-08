"""The import rules from CLAUDE.md, enforced rather than described.

`analysis/` measures a finished world.  If a stage could import it, a stage could read its
own report card and start tuning itself against a metric, and the pipeline would stop
being a straight line from seed to world.  The rule was stated in three places and checked
in none, which is the state a rule is in just before someone breaks it.

`render/` is here for the same reason and is the older rule: a stage that could draw would
make the debug viewer load-bearing.

`export/` is where all file I/O lives, so a stage reaching into it is a stage one import
away from writing a file.  Two stages do have to read one — a heightmap picture, the
culture-pack YAML — and those readers stay in `export/` with the rest of the I/O, so the
rule carries a short, named allowance for them rather than a hole the width of a package.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent / "worldgen"

# package -> the sibling packages it may not import, and why.
_WEB = {"web": "the web interface drives the generator; nothing it drives may reach back"}

_FORBIDDEN = {
    "stages": {
        "analysis": "a stage must not read its own report card",
        "render": "a stage must not draw",
        "export": (
            "a stage does no file I/O; it may read through export.culture_packs or "
            "export.heightmap_import and nothing else"
        ),
        **_WEB,
    },
    "core": {
        "analysis": "core is data types and the pipeline, and measures nothing",
        "render": "core must not draw",
        "export": "core does no file I/O",
        **_WEB,
    },
    "naming": {
        "stages": "naming is a vocabulary stages use, not a stage",
        "analysis": "naming must not read a report card either",
        "render": "naming must not draw",
        "export": "naming does no file I/O",
        **_WEB,
    },
    "analysis": {
        "render": "analysis returns numbers, not pictures",
        "export": "analysis does no file I/O",
        "stages": "analysis reads a finished world, it does not make one",
        **_WEB,
    },
    "export": _WEB,
    "render": _WEB,
}


# package -> the modules of a forbidden package it may import all the same, and why.  A
# module named here is allowed with everything inside it; its siblings are not.
_ALLOWED = {
    "stages": {
        "export.heightmap_import": "ImageElevationStage reads its picture; it writes nothing",
        "export.culture_packs": "NamingStage reads the pack YAML; it writes nothing",
    },
}


def _imported_modules(source: str, package: str) -> set[str]:
    """Every `worldgen.<module>` the source pulls in, dotted from below `worldgen`.

    `from ..export import svg_export` and `import worldgen.export.svg_export` both come out
    as `export.svg_export`.  A `from` import contributes each name it imports, since that
    name may be a module; a name that is only a class or function makes the path one
    segment too long, which no rule here minds.

    Takes text and the name of the package the file lives in, not a path, so the guard
    below can be shown to fail without a probe file being written into the source tree.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "worldgen" and len(parts) > 1:
                    found.add(".".join(parts[1:]))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not (node.module and node.module.startswith("worldgen.")):
                    continue
                base = node.module.removeprefix("worldgen.")
            elif node.level == 1:
                # `from .x import y` — a sibling inside this package, not a layer hop.
                continue
            else:
                # `from ..x import y` — up to worldgen, then back down into a package.
                # `from .. import x` names the package itself.
                base = node.module or ""
            for alias in node.names:
                found.add(f"{base}.{alias.name}" if base else alias.name)
    return found


def _imported_packages(source: str, package: str) -> set[str]:
    """Every `worldgen.<package>` the source pulls in, however it spells the import."""
    return {module.split(".")[0] for module in _imported_modules(source, package)}


def _is_allowed(package: str, module: str) -> bool:
    return any(
        module == allowed or module.startswith(allowed + ".")
        for allowed in _ALLOWED.get(package, {})
    )


def _offences(package: str, sources: dict[str, str]) -> list[str]:
    """Which of *sources* — name to text — break *package*'s import rules."""
    banned = _FORBIDDEN[package]
    offences = []
    for name, source in sorted(sources.items()):
        hops = {
            module.split(".")[0]
            for module in _imported_modules(source, package)
            if module.split(".")[0] in banned and not _is_allowed(package, module)
        }
        offences += [f"{name} imports {hop} — {banned[hop]}" for hop in sorted(hops)]
    return offences


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


def test_every_allowance_names_a_module_that_exists():
    """An allowance for a reader that has moved or been renamed is a hole nobody is using."""
    for allowances in _ALLOWED.values():
        for module in allowances:
            assert (_ROOT / (module.replace(".", "/") + ".py")).is_file(), module


@pytest.mark.parametrize(
    "spelling",
    [
        "from ..export.heightmap_import import load_luminance",
        "from ..export import heightmap_import",
        "from ..export.culture_packs import LoadedPack, load_packs",
        "from worldgen.export.culture_packs import load_packs",
        "import worldgen.export.heightmap_import",
        "def run(self):\n    from ..export.heightmap_import import load_luminance",
    ],
)
def test_a_stage_may_import_a_read_only_loader(spelling):
    assert _offences("stages", {"stages/probe.py": spelling}) == []


_NO_FILE_IO = (
    "stages/probe.py imports export — a stage does no file I/O; it may read through "
    "export.culture_packs or export.heightmap_import and nothing else"
)


@pytest.mark.parametrize(
    "spelling",
    [
        "from ..export import svg_export",
        "from ..export.json_export import save",
        "from worldgen.export import png_export",
        "import worldgen.export",
        "from .. import export",
        "from ..export import heightmap_import, svg_export",
        "def run(self):\n    from ..export.json_export import save",
    ],
)
def test_a_stage_may_not_import_the_rest_of_export(spelling):
    """The allowance is two modules wide.  The package itself, or a sibling imported in the
    same statement as an allowed reader, is still a stage reaching for the writers."""
    assert _offences("stages", {"stages/probe.py": spelling}) == [_NO_FILE_IO]


def test_the_allowance_is_for_stages_only():
    """core does no file I/O at all, reading included."""
    offences = _offences("core", {"core/probe.py": "from ..export import heightmap_import"})
    assert offences == ["core/probe.py imports export — core does no file I/O"]


def test_a_stage_that_reads_through_export_does_not_load_matplotlib():
    """`export/__init__` loads its writers lazily, so the allowance above stays narrow at
    run time too: reading a heightmap or a culture pack does not start the plotting stack.

    A fresh interpreter, since this test process has long since imported matplotlib.
    """
    probe = (
        "import sys\n"
        "import worldgen.export.heightmap_import, worldgen.export.culture_packs\n"
        "import worldgen.stages.image_elevation, worldgen.stages.naming\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT.parent,
    )
    assert result.stdout.strip() == "[]"


def test_the_writers_are_still_reachable_as_attributes():
    """Lazy loading must not break `from worldgen import export; export.svg_export`."""
    from worldgen import export

    for name in export.__all__:
        assert getattr(export, name).__name__ == f"worldgen.export.{name}"
    with pytest.raises(AttributeError):
        export.no_such_writer  # noqa: B018
