"""The CI workflow must be a workflow.

`.github/workflows/ci.yml` was invalid YAML from 2026-05-04 to 2026-09-06 — `name: Status
Check` was edited to `name: name: CIExpected`, which does not parse. GitHub cannot report
a parse error against anything, so every run failed in 0s before reaching a step, and lint
and pytest did not run in CI for four months. Nothing noticed, because a workflow that
never starts also never fails visibly, and no branch protection required it.

These tests read the file the way GitHub does. They are cheap and they close the specific
hole that let a dead pipeline look like a live one.
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
