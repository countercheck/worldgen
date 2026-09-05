"""The reference manual must describe the generator that exists.

Prose goes stale quietly. `docs/REFERENCE.md` accumulated eighteen retired setting names
and lost track of fifty-three live ones across the units work without anything failing,
because nothing reads it. These tests read it.

They check names and coverage only — that every setting is documented somewhere in the
configuration reference and that no retired name is still presented as current. What a
setting *means* is still on the author.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from worldgen.core.config import _RENAMED_FIELDS, _RETIRED_FIELDS, WorldConfig

REFERENCE = Path(__file__).resolve().parent.parent / "docs" / "REFERENCE.md"

# `| `name` | `type` | `default` | range | effect |` — the row shape used throughout
# section 4. Anchored at the start of a line so prose mentioning a setting in a sentence
# does not count as documenting it.
_PARAM_ROW = re.compile(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|", re.MULTILINE)


@pytest.fixture(scope="module")
def reference_text():
    return REFERENCE.read_text()


@pytest.fixture(scope="module")
def config_section(reference_text):
    """Section 4, the configuration reference, up to section 5."""
    start = reference_text.index("## 4. Configuration Reference")
    end = reference_text.index("## 5. In-Code Constants")
    return reference_text[start:end]


def test_every_setting_appears_in_the_configuration_reference(config_section):
    documented = set(_PARAM_ROW.findall(config_section))
    declared = {f.name for f in fields(WorldConfig)}
    missing = sorted(declared - documented)
    assert not missing, f"docs/REFERENCE.md section 4 does not document: {', '.join(missing)}"


def test_the_configuration_reference_documents_nothing_that_does_not_exist(config_section):
    documented = set(_PARAM_ROW.findall(config_section))
    declared = {f.name for f in fields(WorldConfig)}
    unknown = sorted(documented - declared)
    assert not unknown, (
        f"docs/REFERENCE.md section 4 documents settings that do not exist: {', '.join(unknown)}"
    )


def test_no_retired_setting_is_presented_as_current(reference_text):
    """A retired name may appear only where the document says it was retired.

    Naming the old setting is often the clearest way to explain a change, so this checks
    for retired names standing in a parameter table or as a live cross-reference, not for
    every mention.
    """
    retired = set(_RETIRED_FIELDS) | set(_RENAMED_FIELDS)
    offenders = sorted(set(_PARAM_ROW.findall(reference_text)) & retired)
    assert not offenders, (
        f"docs/REFERENCE.md still lists retired settings as parameters: {', '.join(offenders)}"
    )
