"""`docs/CAMPAIGN.md` must describe the campaign layer that exists.

The same reasoning as `test_docs.py`: prose goes stale quietly, and a document nothing
reads is a document nobody notices going wrong. These tests read the parts that can drift
mechanically — the commands it tells you to type, and the environment variables it says
the server understands. What any of it *means* is still on the author.

Deliberately not asserted: anything about the TypeScript source beyond those two surfaces.
That is what `campaign/`'s own suite is for, and duplicating it here in a language that
cannot import it would be a worse copy of a better test.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "CAMPAIGN.md"
PACKAGE = ROOT / "campaign" / "package.json"
DOCKERFILE = ROOT / "campaign" / "Dockerfile"
ENTRYPOINT = ROOT / "campaign" / "server" / "src" / "index.ts"


@pytest.fixture(scope="module")
def doc():
    assert DOC.is_file(), f"{DOC} is missing"
    return DOC.read_text()


def test_the_document_exists_and_says_what_it_is(doc):
    assert doc.startswith("# The campaign layer")


def test_every_npm_command_it_names_is_a_real_script(doc):
    """`npm run <script>`, checked against the workspace root's manifest.

    A document telling a referee to run something that no longer exists wastes the time of
    the one person who most needs the instructions to be right.
    """
    scripts = set(json.loads(PACKAGE.read_text())["scripts"])
    named = set(re.findall(r"npm run ([a-z][a-z-]*)", doc))

    missing = sorted(named - scripts)
    assert not missing, f"docs/CAMPAIGN.md runs scripts that do not exist: {missing}"


def test_every_worldgen_option_it_names_is_a_real_option(doc):
    """`python3 -m worldgen.cli <command> --flag`, checked against click itself.

    This is the first command a referee setting up a campaign types, and it was wrong: the
    document said `--output world.json` for a year after the option became `--output-dir`
    and started naming a directory. Nothing failed anywhere — the CLI's own tests passed,
    the campaign layer's passed, and the only person who found out was whoever followed
    the instructions.
    """
    from worldgen.cli import cli

    invocations = re.findall(r"python3? -m worldgen\.cli (\w+)((?:[^`]|\n)*?)(?:```|\n\n)", doc)
    assert invocations, "no worldgen.cli commands found — has the document changed shape?"

    wrong: list[str] = []
    for name, tail in invocations:
        command = cli.commands.get(name)
        if command is None:
            wrong.append(f"no such command: {name}")
            continue

        real = {opt for param in command.params for opt in param.opts}
        for flag in re.findall(r"(--[a-z][a-z0-9-]*)", tail):
            if flag not in real:
                wrong.append(f"{name} has no option {flag}")

    assert not wrong, "docs/CAMPAIGN.md types options that do not exist: " + "; ".join(wrong)


def test_it_documents_every_environment_variable_the_server_reads(doc):
    """The server's whole configuration surface is four `process.env` reads.

    Adding a fifth without a row in the table is exactly the drift this catches — the
    variable works, nothing fails, and nobody deploying it ever finds out.
    """
    read = set(re.findall(r"process\.env\.([A-Z_][A-Z0-9_]*)", ENTRYPOINT.read_text()))
    documented = set(re.findall(r"`([A-Z_][A-Z0-9_]*)`", doc))

    missing = sorted(read - documented)
    assert not missing, f"docs/CAMPAIGN.md does not document: {missing}"


def test_it_documents_nothing_the_server_does_not_read(doc):
    """The other direction: a table row for a variable that does nothing is worse than no
    row at all, because somebody will set it and expect an effect."""
    read = set(re.findall(r"process\.env\.([A-Z_][A-Z0-9_]*)", ENTRYPOINT.read_text()))

    # Only rows of the environment table, which is the part making a claim about what the
    # server honours. Prose elsewhere may legitimately mention other names.
    rows = re.findall(r"^\|\s*`([A-Z_][A-Z0-9_]*)`\s*\|", doc, re.MULTILINE)

    invented = sorted(set(rows) - read)
    assert not invented, f"docs/CAMPAIGN.md documents variables nothing reads: {invented}"


def test_the_container_sets_only_variables_that_are_documented(doc):
    """The image's `ENV` block and the document have to agree.

    They drifted apart the first time this was written: the Dockerfile grew
    `CAMPAIGN_CLIENT` before anything said what it was for.
    """
    env_block = re.findall(r"^\s+([A-Z_][A-Z0-9_]*)=", DOCKERFILE.read_text(), re.MULTILINE)
    documented = set(re.findall(r"`([A-Z_][A-Z0-9_]*)`", doc))

    # NODE_ENV is Node's own and is not part of the application's configuration surface.
    missing = sorted({v for v in env_block if v != "NODE_ENV"} - documented)
    assert not missing, f"the image sets undocumented variables: {missing}"


def test_the_dockerfile_builds_from_the_repository_root(doc):
    """The build context has to be the root, because the client imports a world fixture
    from `campaign/shared/test/fixtures` for its demonstration scenario.

    Documenting `docker build` from inside `campaign/` would fail in a way whose error
    message points at the wrong thing entirely.
    """
    assert "-f campaign/Dockerfile ." in doc
    assert "COPY campaign/package.json" in DOCKERFILE.read_text()
