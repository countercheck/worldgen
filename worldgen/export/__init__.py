"""All file I/O: the writers (JSON, SVG, PNG) and the read-only loaders stages use.

The writers are loaded on first use rather than here.  SVG and PNG pull in matplotlib, and
a stage that imports `export.heightmap_import` or `export.culture_packs` would otherwise
load the whole plotting stack just to read a picture or a YAML file.
"""

import importlib

__all__ = ["json_export", "svg_export", "png_export"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
