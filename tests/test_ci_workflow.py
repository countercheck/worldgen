"""The CI workflow must be a workflow.

`.github/workflows/ci.yml` was invalid YAML from 2026-05-04 to 2026-09-06 — `name: Status
Check` was edited to `name: name: CIExpected`, which does not parse. GitHub cannot report
a parse error against anything, so every run failed in 0s before reaching a step, and lint
and pytest did not run in CI for four months. Nothing noticed, because a workflow that
never starts also never fails visibly, and no branch protection required it.

These tests read the file the way GitHub does. They are cheap and they close the specific
hole that let a dead pipeline look like a live one.

The repository now has two suites in two languages. The `campaign/` TypeScript workspace
carries the fog leakage tests, which are the most important tests in the project, and it
is exactly as capable of not being run as the Python side was. So the assertions below
cover both, and one of them checks the subtler version of the same failure: a job that is
waited for by the gate but never actually examined by it.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow():
    assert WORKFLOW.is_file(), f"{WORKFLOW} is missing"
    return yaml.safe_load(WORKFLOW.read_text())


def test_the_workflow_parses(workflow):
    """A parse error here means CI does not run at all — the failure mode that hid."""
    assert isinstance(workflow, dict)
    assert "jobs" in workflow


def test_every_job_declares_steps(workflow):
    for name, spec in workflow["jobs"].items():
        assert spec.get("steps"), f"job {name!r} has no steps"


def test_lint_and_tests_actually_run(workflow):
    runs = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
    joined = "\n".join(runs)
    assert "ruff check" in joined, "CI does not lint"
    assert "pytest" in joined, "CI does not run the tests"


def test_no_step_swallows_its_own_failure(workflow):
    """`|| true` on the coverage step made the floor unenforceable for months.

    A step that cannot fail is a step that is not checking anything. If one ever needs to
    be advisory, give it `continue-on-error` so it is visible as such in the run summary.
    """
    for job_name, job in workflow["jobs"].items():
        for step in job["steps"]:
            run = step.get("run", "")
            assert "|| true" not in run, (
                f"step {step.get('name', run[:40])!r} in job {job_name!r} swallows failure"
            )


def test_the_coverage_floor_is_enforced(workflow):
    runs = [s.get("run", "") for j in workflow["jobs"].values() for s in j["steps"]]
    cov = [r for r in runs if "--cov-fail-under" in r]
    assert cov, "no step enforces a coverage floor"


def test_the_gate_job_waits_for_lint_and_tests(workflow):
    """The gate is what branch protection points at; if it does not need the real jobs
    it reports success while they fail."""
    gate = workflow["jobs"]["required"]
    assert set(gate["needs"]) >= {"lint", "test"}


# ---- the TypeScript half ------------------------------------------------
#
# `campaign/` is a second suite in a second language, and a suite nobody runs is a suite
# that does not exist. These are the same assertions as above, aimed one language over.

JS_JOBS = ("js-lint", "js-test")


def test_the_typescript_suite_runs(workflow):
    runs = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
    joined = "\n".join(runs)
    assert "npm test" in joined, "CI does not run the TypeScript tests"
    assert "npm run lint" in joined, "CI does not lint the TypeScript"
    assert "npm run typecheck" in joined, "CI does not typecheck the TypeScript"


def test_the_typescript_install_is_reproducible(workflow):
    """`npm ci` installs the lock file; `npm install` may quietly resolve something else.

    Same class of mistake as the unpinned ruff that broke the first green build after the
    outage — a gate that moves on its own is not a gate.
    """
    for name in JS_JOBS:
        runs = [step.get("run", "") for step in workflow["jobs"][name]["steps"]]
        assert any("npm ci" in r for r in runs), f"job {name!r} does not use `npm ci`"
        assert not any("npm install" in r for r in runs), (
            f"job {name!r} uses `npm install`, which ignores the lock file"
        )


def test_the_gate_waits_for_every_suite(workflow):
    """Both languages, or the one that is not listed gates nothing."""
    gate = workflow["jobs"]["required"]
    assert set(gate["needs"]) >= {"lint", "test", *JS_JOBS}


def test_the_gate_examines_everything_it_waits_for(workflow):
    """The subtle version of the same hole.

    A job in `needs` but absent from the shell check is waited for and then ignored: the
    gate reports success while that suite is red. It looks correct in the job graph and in
    the run summary, and it is the natural mistake to make when adding a job — which is
    why it is asserted rather than trusted.
    """
    gate = workflow["jobs"]["required"]
    script = "\n".join(step.get("run", "") for step in gate["steps"])

    for job in gate["needs"]:
        # Either spelling. A job id containing a hyphen must be indexed rather than
        # dotted — see the comment on the gate step — so both forms are legitimate and
        # the next person to add a job should not have to guess which one this checks.
        dotted = f"needs.{job}.result"
        indexed = f"needs['{job}'].result"
        assert dotted in script or indexed in script, (
            f"the gate waits for {job!r} but never looks at its result"
        )


def test_hyphenated_jobs_are_indexed_not_dotted(workflow):
    """`needs.js-lint.result` is a subtraction, not a property.

    GitHub's expression language treats `-` as an operator, so the dotted form of a
    hyphenated job id evaluates to nothing rather than to a result. It fails silently in
    the direction of a gate that never passes — or, once somebody "fixes" it by deleting
    the clause, a gate that never checks.
    """
    for job_name, job in workflow["jobs"].items():
        script = "\n".join(step.get("run", "") for step in job["steps"])
        for needed in job.get("needs", []):
            if "-" in needed:
                assert f"needs.{needed}." not in script, (
                    f"job {job_name!r} dots a hyphenated job id: use needs['{needed}']"
                )


def test_every_gated_job_exists(workflow):
    """A `needs` entry naming a job that is not defined is a workflow that will not run."""
    defined = set(workflow["jobs"])
    for job in workflow["jobs"]["required"]["needs"]:
        assert job in defined, f"the gate needs {job!r}, which is not a job"


def test_the_typescript_tests_are_built_before_they_are_run(workflow):
    """The server and client suites import `@campaign/shared` by package name.

    That resolves to the workspace's built `dist/`, which a fresh checkout does not have,
    so without a build step two of the three suites fail to load a single test file. It
    passes on a laptop because a developer has built at some point and `tsc --build` then
    reports the project up to date from a stale `.tsbuildinfo` — which makes this the
    exact shape of bug that is green locally and red on a runner. It was red on the first
    attempt here.
    """
    steps = workflow["jobs"]["js-test"]["steps"]
    runs = [step.get("run", "") for step in steps]

    build = next((i for i, r in enumerate(runs) if "npm run build" in r), None)
    test = next((i for i, r in enumerate(runs) if "npm test" in r), None)

    assert build is not None, "the TypeScript suites are run without being built"
    assert test is not None, "the TypeScript job runs no tests"
    assert build < test, "the build must come before the tests, not after them"
