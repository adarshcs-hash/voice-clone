"""IndicF5 backend behaviour that can be checked without the weights.

The weights are gated and multi-gigabyte, so the model itself is exercised by
the ``slow``-marked tests. What is worth pinning here is the failure reporting:
the first-run errors an operator will actually hit.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from mlvoice.errors import BackendUnavailableError
from mlvoice.tts.indicf5 import IndicF5Synthesizer


def _stub_modules(error: Exception) -> dict[str, types.ModuleType]:
    """Stand in for torch and transformers, with from_pretrained raising."""
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda name: name  # type: ignore[attr-defined]

    class FailingAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise error

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModel = FailingAutoModel  # type: ignore[attr-defined]
    return {"torch": torch_stub, "transformers": transformers_stub}


class TestInfo:
    def test_reports_its_configuration(self) -> None:
        info = IndicF5Synthesizer(revision="abc123", device="cpu").info
        assert info.backend == "indicf5"
        assert info.revision == "abc123"
        assert info.sample_rate == 24_000
        assert info.supports_cloning is True

    def test_is_not_ready_before_loading(self) -> None:
        assert IndicF5Synthesizer().is_ready() is False


class TestLoadFailures:
    """The gated-repo 401 is the most likely first-run error, and the raw
    message does not say what to do about it."""

    @pytest.mark.parametrize(
        "message",
        [
            "401 Client Error. Cannot access gated repo for url ...",
            "Access to model ai4bharat/IndicF5 is restricted. You must have access",
            "GatedRepoError: 401 Client Error",
        ],
    )
    def test_gating_errors_carry_an_actionable_hint(self, message: str) -> None:
        with (
            patch.dict(sys.modules, _stub_modules(OSError(message))),
            pytest.raises(BackendUnavailableError) as excinfo,
        ):
            IndicF5Synthesizer(revision="abc123").load()
        hint = excinfo.value.context.get("hint")
        assert hint is not None
        assert "gated" in hint
        assert "HF_TOKEN" in hint

    def test_unrelated_errors_get_no_misleading_hint(self) -> None:
        with (
            patch.dict(sys.modules, _stub_modules(OSError("No space left on device"))),
            pytest.raises(BackendUnavailableError) as excinfo,
        ):
            IndicF5Synthesizer(revision="abc123").load()
        assert excinfo.value.context.get("hint") is None
        assert "No space left" in excinfo.value.context["reason"]

    def test_failure_reports_the_model_and_revision(self) -> None:
        with (
            patch.dict(sys.modules, _stub_modules(OSError("boom"))),
            pytest.raises(BackendUnavailableError) as excinfo,
        ):
            IndicF5Synthesizer(model_id="org/model", revision="rev1").load()
        assert excinfo.value.context["model_id"] == "org/model"
        assert excinfo.value.context["revision"] == "rev1"

    def test_missing_extra_is_reported_clearly(self) -> None:
        """Without the models extra, the import itself fails."""
        with (
            patch.dict(sys.modules, {"torch": None, "transformers": None}),
            pytest.raises(BackendUnavailableError, match="models"),
        ):
            IndicF5Synthesizer().load()


class TestSynthesisGuards:
    def test_synthesis_before_load_is_refused(self) -> None:
        from mlvoice.text.pipeline import TextPipeline
        from mlvoice.tts.base import SynthesisRequest

        request = SynthesisRequest(text=TextPipeline().process("നാട്"))
        with pytest.raises(BackendUnavailableError, match="not loaded"):
            IndicF5Synthesizer()._synthesize_chunk("നാട്", request)
