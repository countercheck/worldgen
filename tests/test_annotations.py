"""Every builtin container annotation says what it holds.

`owner: dict = {}` tells a reader nothing the `{}` did not.  Is it keyed by coordinate or by
seat, and does it hold a seat, a cost or a set of them?  In code where half the dicts map a
hex to another hex and the other half map a hex to a float, that is the question every
reader asks, and the annotation is the one place the answer could have been written down.

Ruff has no rule for it: the `ANN` family checks that an annotation is present, not that
it is complete, and `UP006` only rewrites `typing.Dict` to `dict`.  So this walks the
source instead, as `test_layering.py` does for imports.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent / "worldgen"

_GENERICS = frozenset({"dict", "list", "set", "tuple", "frozenset"})


def _annotations(tree: ast.AST) -> list[ast.expr]:
    """Every annotation in *tree*: arguments, returns, and annotated assignments, which
    covers dataclass fields and locals alike."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
            found += [a.annotation for a in every if a is not None and a.annotation is not None]
            if node.returns is not None:
                found.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
    return found


def _bare(annotation: ast.expr) -> list[str]:
    """The builtin generics in *annotation* written without their parameters.

    Anywhere inside it, so `list[dict]` and `dict | None` are caught along with `dict`.  A
    quoted annotation is parsed and read the same way.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    subscripted = {
        id(node.value) for node in ast.walk(annotation) if isinstance(node, ast.Subscript)
    }
    return [
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and node.id in _GENERICS and id(node) not in subscripted
    ]


def _offences(sources: dict[str, str]) -> list[str]:
    """Which of *sources* — name to text — annotate a builtin container bare, as file:line."""
    offences = []
    for name, source in sorted(sources.items()):
        for annotation in _annotations(ast.parse(source)):
            offences += [
                f"{name}:{annotation.lineno}: bare `{kind}` — say what it holds"
                for kind in _bare(annotation)
            ]
    return offences


def _sources() -> dict[str, str]:
    return {
        str(path.relative_to(_ROOT.parent)): path.read_text(encoding="utf-8")
        for path in sorted(_ROOT.rglob("*.py"))
    }


def test_no_container_annotation_is_bare():
    offences = _offences(_sources())
    assert not offences, "\n".join(offences)


@pytest.mark.parametrize(
    "source, kind",
    [
        ("owner: dict = {}", "dict"),
        ("def f(seen: set) -> None: ...", "set"),
        ("def f() -> list: ...", "list"),
        ("def f(*runs: tuple) -> None: ...", "tuple"),
        ("def f(**kw: frozenset) -> None: ...", "frozenset"),
        ("def f(cache: dict | None = None) -> None: ...", "dict"),
        ("def f() -> list[dict]: ...", "dict"),
        ("edges: dict[frozenset, float] = {}", "frozenset"),
        ("def f() -> 'dict': ...", "dict"),
        ("class C:\n    metadata: dict = field(default_factory=dict)", "dict"),
        ("async def f() -> set: ...", "set"),
    ],
)
def test_the_check_would_catch_a_bare_container(source, kind):
    """A guard that cannot fail is not a guard: every place an annotation can stand, and a
    bare container nested inside one that is otherwise complete."""
    assert _offences({"probe.py": source}) == [
        f"probe.py:{source.count(chr(10)) + 1}: bare `{kind}` — say what it holds"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "owner: dict[str, int] = {}",
        "def f(seen: set[tuple[int, int]]) -> list[frozenset[int]]: ...",
        "def f(cache: dict[str, int] | None = None) -> 'dict[str, int]': ...",
        "def f(xs: tuple[int, ...]) -> None: ...",
        # Not annotations: the names used as values are none of this check's business.
        "owner = dict()",
        "def f(x) -> bool:\n    return isinstance(x, dict | list)",
        "seen: set[int] = set()",
    ],
)
def test_a_parameterised_container_passes(source):
    assert _offences({"probe.py": source}) == []
