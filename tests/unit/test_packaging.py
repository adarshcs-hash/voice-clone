"""Packaging invariants.

These exist because a broken `pyproject.toml` fails at *install* time, which no
amount of passing tests catches: a green suite on an already-installed package
says nothing about whether the package still installs.

Two failures have already shipped from this file:

* a direct-reference dependency without ``allow-direct-references``, which
  breaks metadata generation for **every** extra and for the plain
  ``pip install -e .``, since extras are validated whether or not they are
  requested; and
* dependency pins for the IndicF5 backend that are load-bearing rather than
  cautionary, and would be silently fatal if tidied away.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    """The parsed pyproject.toml."""
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _all_dependencies(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject["project"]
    extras = project.get("optional-dependencies", {})
    return [
        *project.get("dependencies", []),
        *(dep for group in extras.values() for dep in group),
    ]


class TestDirectReferences:
    def test_direct_references_are_permitted_when_used(self, pyproject: dict[str, Any]) -> None:
        """Hatchling rejects a direct reference at metadata generation unless
        this is set, which breaks the plain editable install as well."""
        uses_direct = any(" @ " in dep for dep in _all_dependencies(pyproject))
        allowed = (
            pyproject.get("tool", {})
            .get("hatch", {})
            .get("metadata", {})
            .get("allow-direct-references", False)
        )
        if uses_direct:
            assert allowed is True, (
                "a dependency uses a direct reference, so "
                "tool.hatch.metadata.allow-direct-references must be true"
            )

    def test_the_package_metadata_is_installed_and_readable(self) -> None:
        """If metadata generation had failed, this import would not resolve."""
        from importlib.metadata import metadata

        assert metadata("mlvoice")["Name"] == "mlvoice"


class TestIndicF5Pins:
    """Each of these was established by measurement, not preference. The
    comments in pyproject.toml carry the reasoning; these assertions stop the
    constraints being dropped by someone tidying up."""

    @pytest.fixture
    def extra(self, pyproject: dict[str, Any]) -> list[str]:
        return pyproject["project"]["optional-dependencies"]["indicf5"]

    def test_f5_tts_comes_from_the_ai4bharat_fork(self, extra: list[str]) -> None:
        """The PyPI package shares the import name but not the API."""
        entry = next((d for d in extra if d.startswith("f5_tts")), None)
        assert entry is not None
        assert "github.com/AI4Bharat/IndicF5" in entry

    def test_pypi_f5_tts_is_not_requested(self, extra: list[str]) -> None:
        assert not any(d.startswith("f5-tts") for d in extra)

    def test_transformers_is_bounded_below_4_50(self, extra: list[str]) -> None:
        """4.51 makes meta-device init unconditional; AI4Bharat says <4.50."""
        entry = next((d for d in extra if d.startswith("transformers")), None)
        assert entry is not None
        assert "<4.50" in entry

    def test_the_extra_includes_the_model_runtime(self, extra: list[str]) -> None:
        assert "mlvoice[models]" in extra


class TestExtraShape:
    def test_declared_extras(self, pyproject: dict[str, Any]) -> None:
        extras = set(pyproject["project"]["optional-dependencies"])
        assert extras == {"models", "indicf5", "data", "eval", "speaker", "browser", "dev"}

    def test_runtime_dependencies_carry_no_model_runtime(self, pyproject: dict[str, Any]) -> None:
        """The text frontend and the service must install without torch."""
        core = " ".join(pyproject["project"]["dependencies"])
        for heavy in ("torch", "transformers", "f5_tts", "speechbrain"):
            assert heavy not in core, f"{heavy} must stay in an extra"
