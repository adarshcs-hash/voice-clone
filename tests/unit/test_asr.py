"""Transcriber behaviour that can be checked without model weights.

The model itself needs the ``models`` extra and a multi-gigabyte download, so
what is pinned here is the contract and the failure reporting: that it satisfies
the protocol the rest of the system depends on, resamples to what these models
expect, and degrades usefully rather than cryptically.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mlvoice.asr import ASR_MODEL_SAMPLE_RATE, TransformersTranscriber
from mlvoice.audio.io import Audio
from mlvoice.errors import BackendUnavailableError
from mlvoice.protocols import Transcriber


def _stubs(pipeline_impl: Any) -> dict[str, types.ModuleType]:
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda name: name  # type: ignore[attr-defined]
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.pipeline = pipeline_impl  # type: ignore[attr-defined]
    return {"torch": torch_stub, "transformers": transformers_stub}


class TestContract:
    def test_satisfies_the_transcriber_protocol(self) -> None:
        assert isinstance(TransformersTranscriber(), Transcriber)

    def test_reports_its_model(self) -> None:
        transcriber = TransformersTranscriber(model_id="org/model")
        assert "org/model" in transcriber.name

    def test_not_ready_before_loading(self) -> None:
        assert TransformersTranscriber().is_ready() is False


class TestTranscription:
    @pytest.fixture
    def recorded(self) -> dict[str, Any]:
        return {}

    def _transcriber(self, recorded: dict[str, Any], text: str = "ഇത് ഒരു പരീക്ഷണം") -> Any:
        def fake_pipeline(task: str, **kwargs: Any) -> Any:
            recorded["task"] = task
            recorded["pipeline_kwargs"] = kwargs

            def run(payload: Any, **call_kwargs: Any) -> dict[str, str]:
                recorded["payload"] = payload
                recorded["call_kwargs"] = call_kwargs
                return {"text": f"  {text}  "}

            return run

        return fake_pipeline

    def test_returns_stripped_text(self, recorded: dict[str, Any], clip: Audio) -> None:
        transcriber = TransformersTranscriber()
        with patch.dict(sys.modules, _stubs(self._transcriber(recorded))):
            assert transcriber.transcribe(clip) == "ഇത് ഒരു പരീക്ഷണം"

    def test_audio_is_resampled_for_the_model(self, recorded: dict[str, Any], clip: Audio) -> None:
        """Whisper and the conformer families all expect 16 kHz."""
        assert clip.sample_rate != ASR_MODEL_SAMPLE_RATE
        with patch.dict(sys.modules, _stubs(self._transcriber(recorded))):
            TransformersTranscriber().transcribe(clip)
        assert recorded["payload"]["sampling_rate"] == ASR_MODEL_SAMPLE_RATE

    def test_language_hint_is_passed(self, recorded: dict[str, Any], clip: Audio) -> None:
        """Whisper detects Malayalam unreliably on clips this short."""
        with patch.dict(sys.modules, _stubs(self._transcriber(recorded))):
            TransformersTranscriber(language="ml").transcribe(clip)
        assert recorded["call_kwargs"]["generate_kwargs"] == {"language": "ml"}

    def test_language_hint_can_be_omitted(self, recorded: dict[str, Any], clip: Audio) -> None:
        with patch.dict(sys.modules, _stubs(self._transcriber(recorded))):
            TransformersTranscriber(language=None).transcribe(clip)
        assert recorded["call_kwargs"] == {}

    def test_a_model_rejecting_the_hint_falls_back_to_detection(self, clip: Audio) -> None:
        """Not every recogniser accepts generate_kwargs; failing the request
        over a hint would be worse than detecting the language."""
        calls: list[dict[str, Any]] = []

        def fake_pipeline(task: str, **kwargs: Any) -> Any:
            def run(payload: Any, **call_kwargs: Any) -> dict[str, str]:
                calls.append(call_kwargs)
                if call_kwargs:
                    raise ValueError("unexpected keyword generate_kwargs")
                return {"text": "ശരി"}

            return run

        with patch.dict(sys.modules, _stubs(fake_pipeline)):
            assert TransformersTranscriber().transcribe(clip) == "ശരി"
        assert len(calls) == 2
        assert calls[1] == {}

    def test_model_is_loaded_once(self, recorded: dict[str, Any], clip: Audio) -> None:
        constructions = 0

        def counting_pipeline(task: str, **kwargs: Any) -> Any:
            nonlocal constructions
            constructions += 1
            return lambda payload, **call_kwargs: {"text": "ok"}

        transcriber = TransformersTranscriber()
        with patch.dict(sys.modules, _stubs(counting_pipeline)):
            transcriber.transcribe(clip)
            transcriber.transcribe(clip)
        assert constructions == 1
        assert transcriber.is_ready()

    def test_plain_string_result_is_accepted(self, clip: Audio) -> None:
        def fake_pipeline(task: str, **kwargs: Any) -> Any:
            return lambda payload, **call_kwargs: "നേരിട്ടുള്ള സ്ട്രിംഗ്"

        with patch.dict(sys.modules, _stubs(fake_pipeline)):
            assert TransformersTranscriber().transcribe(clip) == "നേരിട്ടുള്ള സ്ട്രിംഗ്"


class TestFailures:
    def test_missing_extra_is_named(self, clip: Audio) -> None:
        with (
            patch.dict(sys.modules, {"torch": None, "transformers": None}),
            pytest.raises(BackendUnavailableError, match="models"),
        ):
            TransformersTranscriber().transcribe(clip)

    def test_load_failure_reports_the_model(self, clip: Audio) -> None:
        def exploding_pipeline(task: str, **kwargs: Any) -> Any:
            raise OSError("no such model")

        with (
            patch.dict(sys.modules, _stubs(exploding_pipeline)),
            pytest.raises(BackendUnavailableError) as excinfo,
        ):
            TransformersTranscriber(model_id="org/absent", revision="r1").transcribe(clip)
        assert excinfo.value.context["model_id"] == "org/absent"
        assert excinfo.value.context["revision"] == "r1"

    def test_silence_transcribes_to_empty_without_raising(self) -> None:
        silence = Audio(samples=np.zeros(24_000, dtype=np.float32), sample_rate=24_000)

        def fake_pipeline(task: str, **kwargs: Any) -> Any:
            return lambda payload, **call_kwargs: {"text": ""}

        with patch.dict(sys.modules, _stubs(fake_pipeline)):
            assert TransformersTranscriber().transcribe(silence) == ""


class TestStartupLoading:
    """When the recogniser is loaded, and when it deliberately is not.

    This exists because the eager version caused an outage: on a cold cache the
    default model is a three-gigabyte download, and doing it inside the lifespan
    handler left the process stuck before it bound a socket -- the API, the web
    client and ``/healthz`` all unreachable while an optional feature fetched
    weights. The rule is now "load on first use, unless a readiness probe makes
    a slow rollout the cheaper failure".
    """

    class RecordingTranscriber:
        """A transcriber that reports whether anyone asked it to load."""

        name = "recording"

        def __init__(self, *, fails: bool = False) -> None:
            self.loads = 0
            self.fails = fails

        def load(self) -> None:
            self.loads += 1
            if self.fails:
                raise BackendUnavailableError("no weights here")

        def transcribe(self, audio: Audio) -> str:
            return ""

    def _settings(self, **overrides: Any) -> Any:
        from mlvoice.config import Settings

        base: dict[str, Any] = {
            "_env_file": None,
            "env": "development",
            "api_keys": "k",
            "tts_backend": "dummy",
            "consent_signing_key": "c",
            "watermark_key": "w",
        }
        base.update(overrides)
        return Settings(**base)

    def test_the_default_defers_the_download(self) -> None:
        from mlvoice.api.app import _load_transcriber

        transcriber = self.RecordingTranscriber()
        assert _load_transcriber(transcriber, self._settings()) is transcriber
        assert transcriber.loads == 0

    def test_eager_load_is_opt_in(self) -> None:
        from mlvoice.api.app import _load_transcriber

        transcriber = self.RecordingTranscriber()
        _load_transcriber(transcriber, self._settings(asr_eager_load=True))
        assert transcriber.loads == 1

    def test_a_deferred_transcriber_is_still_wired_up(self) -> None:
        """Deferring the load must not disable the feature -- the object has to
        reach the enrolment service or nothing can transcribe later."""
        from mlvoice.api.app import _load_transcriber

        transcriber = self.RecordingTranscriber(fails=True)
        assert _load_transcriber(transcriber, self._settings()) is transcriber

    def test_an_eager_failure_disables_transcription_outside_production(self) -> None:
        from mlvoice.api.app import _load_transcriber

        transcriber = self.RecordingTranscriber(fails=True)
        assert _load_transcriber(transcriber, self._settings(asr_eager_load=True)) is None

    def test_production_with_consent_loads_eagerly_whatever_the_setting(self) -> None:
        """A replica that mandates consent while unable to verify it must not
        report itself ready, so here a slow rollout is the cheaper failure."""
        from mlvoice.api.app import _load_transcriber

        transcriber = self.RecordingTranscriber()
        production = self._settings(
            env="production",
            model_revision="abc123",
            tts_backend="indicf5",
            asr_eager_load=False,
        )
        _load_transcriber(transcriber, production)
        assert transcriber.loads == 1

    def test_production_refuses_to_start_when_it_cannot_verify_consent(self) -> None:
        from mlvoice.api.app import _load_transcriber
        from mlvoice.errors import ConfigurationError

        production = self._settings(
            env="production", model_revision="abc123", tts_backend="indicf5"
        )
        with pytest.raises(ConfigurationError):
            _load_transcriber(self.RecordingTranscriber(fails=True), production)
