"""Pre-flight checks.

What is pinned here is that each check fires on the configuration that has
actually gone wrong in the field, and that severity tracks the environment: the
same setting can be a note on a laptop and a refusal in production. A check
that reports `ok` on a broken deployment is worse than no check, so most of
these assert the failure rather than the success.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import mlvoice.diagnostics
from mlvoice.config import Settings
from mlvoice.diagnostics import Check, Status, hf_cache_root, run_checks, worst_status

# Captured before the autouse fixture below replaces it, so the cache-detection
# test can exercise the real implementation rather than its own stub.
_real_is_cached = mlvoice.diagnostics._is_cached


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "development",
        "api_keys": "a-key",
        "tts_backend": "indicf5",
        "model_revision": "abc123",
        "consent_signing_key": "c",
        "watermark_key": "w",
    }
    base.update(overrides)
    return Settings(**base)


def _named(checks: list[Check], name: str) -> Check:
    found = next((check for check in checks if check.name == name), None)
    assert found is not None, f"no check named {name!r} in {[c.name for c in checks]}"
    return found


@pytest.fixture(autouse=True)
def _no_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend nothing is installed and nothing is cached.

    Every test that cares overrides one of these. Without the default, results
    would depend on whether the machine running the suite happens to have torch
    and a warm Hugging Face cache, which is how a diagnostic test quietly stops
    testing anything.
    """
    monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: None)
    monkeypatch.setattr("mlvoice.diagnostics._is_cached", lambda model_id: False)
    monkeypatch.setattr("mlvoice.diagnostics._f5_tts_is_the_fork", lambda: None)


class TestTheBuzz:
    """The dummy backend is the one that cost the most: it does not fail, it
    just sounds broken."""

    def test_the_dummy_backend_is_reported(self) -> None:
        check = _named(run_checks(_settings(tts_backend="dummy")), "tts_backend")
        assert check.status is Status.WARN
        assert "buzz" in check.detail
        assert "indicf5" in check.hint

    def test_it_is_fatal_in_production(self) -> None:
        checks = run_checks(_settings(tts_backend="dummy"), as_production=True)
        assert _named(checks, "tts_backend").status is Status.FAIL

    def test_the_runtime_is_not_inspected_for_the_dummy_backend(self) -> None:
        """It needs no model runtime, so reporting a missing torch would be
        noise that trains people to ignore the output."""
        names = {check.name for check in run_checks(_settings(tts_backend="dummy"))}
        assert not names & {"torch", "transformers", "f5_tts", "pydub", "device"}


class TestTheRuntimeTrapsFromTheFieldReport:
    def test_missing_torch(self) -> None:
        assert _named(run_checks(_settings()), "torch").status is Status.FAIL

    def test_transformers_too_new(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """4.51 made meta-device init unconditional; the model cannot load."""
        monkeypatch.setattr(
            "mlvoice.diagnostics._version",
            lambda package: {"torch": "2.4.0", "transformers": "4.57.6"}.get(package),
        )
        check = _named(run_checks(_settings()), "transformers")
        assert check.status is Status.FAIL
        assert "too new" in check.detail

    @pytest.mark.parametrize("version", ["4.44.0", "4.49.0"])
    def test_transformers_within_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        monkeypatch.setattr(
            "mlvoice.diagnostics._version",
            lambda package: {"torch": "2.4.0", "transformers": version}.get(package),
        )
        assert _named(run_checks(_settings()), "transformers").status is Status.OK

    def test_the_wrong_f5_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The PyPI package and the fork share the import name. This is the
        check that turns "missing 1 required positional argument" into a
        sentence naming the package to reinstall."""
        monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: "2.4.0")
        monkeypatch.setattr("mlvoice.diagnostics._f5_tts_is_the_fork", lambda: False)
        check = _named(run_checks(_settings()), "f5_tts")
        assert check.status is Status.FAIL
        assert "PyPI" in check.detail

    def test_the_right_f5_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: "2.4.0")
        monkeypatch.setattr("mlvoice.diagnostics._f5_tts_is_the_fork", lambda: True)
        assert _named(run_checks(_settings()), "f5_tts").status is Status.OK


class TestWeights:
    def test_an_empty_cache_is_a_warning_not_a_failure(self) -> None:
        """It will work; the first request just pays for a download it should
        not have been asked to pay for."""
        check = _named(run_checks(_settings()), "model_weights")
        assert check.status is Status.WARN
        assert "download" in check.hint

    def test_a_warm_cache_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("mlvoice.diagnostics._is_cached", lambda model_id: True)
        assert _named(run_checks(_settings()), "model_weights").status is Status.OK

    def test_an_unpinned_revision_is_fatal_in_production(self) -> None:
        checks = run_checks(_settings(model_revision=None), as_production=True)
        assert _named(checks, "model_revision").status is Status.FAIL

    def test_an_unpinned_revision_is_only_a_warning_locally(self) -> None:
        checks = run_checks(_settings(model_revision=None))
        assert _named(checks, "model_revision").status is Status.WARN


class TestCacheLocation:
    def test_hf_hub_cache_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "ignored"))
        assert hf_cache_root() == tmp_path / "hub"

    def test_hf_home_is_used_with_its_hub_subdirectory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert hf_cache_root() == tmp_path / "hub"

    def test_the_default_is_the_documented_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        assert hf_cache_root().parts[-3:] == (".cache", "huggingface", "hub")

    def test_a_materialised_snapshot_is_detected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An empty snapshot directory must not count: an interrupted download
        leaves the tree in place, and reporting that as cached is how a stalled
        fetch gets diagnosed as something else."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        snapshots = tmp_path / "models--org--model" / "snapshots" / "deadbeef"
        snapshots.mkdir(parents=True)
        assert _real_is_cached("org/model") is False
        (snapshots / "config.json").write_text("{}", encoding="utf-8")
        assert _real_is_cached("org/model") is True


class TestDevice:
    def test_an_unavailable_accelerator_is_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: "2.4.0")
        monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=False, mps=False))
        check = _named(run_checks(_settings(device="mps")), "device")
        assert check.status is Status.FAIL

    def test_cpu_while_an_accelerator_is_idle_is_worth_saying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Several times faster for this model, and nobody checks."""
        monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: "2.4.0")
        monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=False, mps=True))
        check = _named(run_checks(_settings(device="cpu")), "device")
        assert check.status is Status.WARN
        assert "MLVOICE_DEVICE=mps" in check.hint

    def test_cpu_on_a_cpu_only_box_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("mlvoice.diagnostics._version", lambda package: "2.4.0")
        monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=False, mps=False))
        assert _named(run_checks(_settings(device="cpu")), "device").status is Status.OK


def _fake_torch(*, cuda: bool, mps: bool) -> Any:
    import types

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    module.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return module


class TestSafetyPosture:
    def test_no_keys_is_a_warning_locally_and_fatal_in_production(self) -> None:
        assert _named(run_checks(_settings(api_keys="")), "auth").status is Status.WARN
        production = run_checks(_settings(api_keys=""), as_production=True)
        assert _named(production, "auth").status is Status.FAIL

    def test_consent_off_says_what_it_permits(self) -> None:
        check = _named(run_checks(_settings(require_consent=False)), "consent")
        assert check.status is Status.WARN
        assert "any uploaded clip can be cloned" in check.detail

    def test_watermarking_enabled_without_a_key_is_always_fatal(self) -> None:
        """Not environment-dependent: it is enabled and cannot work, which is a
        broken configuration rather than a relaxed one."""
        check = _named(run_checks(_settings(watermark_key="")), "watermark")
        assert check.status is Status.FAIL

    def test_transcription_off_is_fatal_where_consent_is_mandatory(self) -> None:
        checks = run_checks(_settings(asr_enabled=False), as_production=True)
        assert _named(checks, "transcription").status is Status.FAIL


class TestVerdict:
    def test_the_worst_status_wins(self) -> None:
        assert worst_status([]) is Status.OK
        ok = Check("a", Status.OK, "")
        warn = Check("b", Status.WARN, "")
        fail = Check("c", Status.FAIL, "")
        assert worst_status([ok, warn]) is Status.WARN
        assert worst_status([ok, warn, fail]) is Status.FAIL

    def test_a_fully_configured_deployment_passes_as_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive case matters too: if nothing can ever pass, the command
        is decoration."""
        monkeypatch.setattr(
            "mlvoice.diagnostics._version",
            lambda package: {"transformers": "4.49.0"}.get(package, "2.4.0"),
        )
        monkeypatch.setattr("mlvoice.diagnostics._is_cached", lambda model_id: True)
        monkeypatch.setattr("mlvoice.diagnostics._f5_tts_is_the_fork", lambda: True)
        monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=True, mps=False))
        checks = run_checks(_settings(device="cuda"), as_production=True)
        assert worst_status(checks) is Status.OK, [c for c in checks if c.status is not Status.OK]

    def test_checking_as_production_does_not_mutate_the_settings(self) -> None:
        settings = _settings()
        run_checks(settings, as_production=True)
        assert settings.env.value == "development"
