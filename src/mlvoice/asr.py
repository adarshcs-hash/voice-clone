"""Speech recognition, for transcribing reference clips.

Why this module exists at all: a reference-prompt TTS model conditions on the
reference clip *and its transcript*, so a product that asks the user to type
that transcript has made its own quality depend on the least reliable step in
the flow. People mistype, paraphrase, or -- most commonly -- paste the text they
want generated. Every one of those clones the voice correctly and garbles the
words, which is the hardest failure to diagnose from the outside.

Transcribing the clip removes the question. The right interaction is to
transcribe, *show* the result, and let the user correct it: ASR is not perfect
either, but a visible approximate transcript beats an invisible wrong one.

Model choice matters, and general Whisper is not good enough
------------------------------------------------------------
This is not a tuning preference. Handed a 9.5-second Malayalam clip,
``openai/whisper-large-v3`` returned **Devanagari** -- it recognised Indic
phonology and chose the wrong writing system -- and then looped one word until
it ran out of tokens. The output was not approximately right; it was a
different script repeated sixty times, and because a reference transcript is
fed to the synthesiser as ``ref_text``, using it would have cloned the voice
faithfully and made it say gibberish.

So the default is a Malayalam-only fine-tune. A model trained on one language
cannot emit the wrong script, which removes the failure mode rather than
mitigating it. Candidates:

``thennal/whisper-medium-ml`` (default)
    ``openai/whisper-medium`` fine-tuned on Malayalam. Reported as the
    strongest Malayalam Whisper model on Common Voice.
``thennal/whisper-large-v2-ml``
    The same lineage at large-v2 scale, fine-tuned on the ICFOSS Malayalam
    Speech Corpus. Better, and roughly twice the download.
``thennal/whisper-small-ml-imasc``
    Small and Malayalam-specific. The one to reach for when the download or
    the latency of the others is the binding constraint.
``ai4bharat/indic-conformer-600m-multilingual``
    AI4Bharat's IndicConformer, trained across Indian languages. Needs
    ``trust_remote_code``, which is why it is not the default.
``openai/whisper-large-v3``
    Multilingual and dependable to *load*, but see above: on Malayalam it is
    capable of producing another script entirely. Not recommended here.

None of these have been benchmarked in this repository. Their ordering comes
from the published evaluations and should be treated as a starting point for
your own measurement.

Whichever is configured, :func:`assess_transcript` checks the output before
anything uses it, because "pick a better model" is not a guarantee.
"""

from __future__ import annotations

import itertools
from typing import Any, Final

from mlvoice.audio.io import Audio, resample
from mlvoice.config import Settings
from mlvoice.errors import BackendUnavailableError
from mlvoice.logging import get_logger
from mlvoice.protocols import Transcriber

__all__ = [
    "ASR_MODEL_SAMPLE_RATE",
    "TransformersTranscriber",
    "assess_transcript",
    "build_transcriber",
]

log = get_logger(__name__)

ASR_MODEL_SAMPLE_RATE: Final = 16_000
"""Whisper and the wav2vec2/conformer families all expect 16 kHz."""

_DECODING_GUARDS: Final[dict[str, Any]] = {
    # Whisper's documented repetition failure: each window is conditioned on
    # the previous window's tokens, so once the decoder starts repeating a
    # phrase it keeps being fed its own loop and never escapes. Turning the
    # conditioning off costs a little cross-window coherence and removes the
    # failure mode where a ten-second clip transcribes as one word sixty times.
    "condition_on_prev_tokens": False,
}

_REPETITION_LIMIT: Final = 4
"""Consecutive identical words tolerated before the output is called a loop."""

_MIN_DISTINCT_RATIO: Final = 0.35
"""Below this share of distinct words, a transcript is degenerate, not terse."""

_LOOP_FLOOR: Final = 8
"""Word count under which the ratio test is not meaningful."""


def assess_transcript(text: str, *, expect_malayalam: bool = True) -> list[str]:
    """Report why ``text`` should not be trusted as a reference transcript.

    Recognition output is not merely sometimes inaccurate; it fails in two
    specific ways that make it *worse than nothing* here, because the
    transcript is fed to the synthesiser as ``ref_text``:

    *   **The wrong script.** A multilingual model handed Malayalam speech can
        return Devanagari -- it heard Indic phonology and picked the wrong
        writing system. The words are not merely misspelled, they are not
        Malayalam, and conditioning the model on them corrupts the clone.
    *   **A decoder loop.** The same word or phrase repeated to the end of the
        output. Recognisable instantly by eye and, without a check, passed on
        as though it were a transcript.

    Both are returned as human-readable reasons rather than raised, because the
    caller's correct response is not to fail: it is to leave the field empty,
    say what happened, and let the user type the sentence.

    Args:
        text: Recognised text.
        expect_malayalam: Whether Malayalam script is the expected output.

    Returns:
        One message per problem found; empty when the transcript looks usable.
    """
    from mlvoice.text.chars import is_malayalam

    stripped = text.strip()
    if not stripped:
        return ["recognition returned nothing"]

    problems: list[str] = []
    letters = [ch for ch in stripped if ch.isalpha()]
    if expect_malayalam and letters and not any(is_malayalam(ch) for ch in letters):
        problems.append(
            "the transcript contains no Malayalam script, so the recogniser "
            "transcribed this clip into the wrong writing system"
        )

    words = stripped.split()
    run = 1
    longest_run = 1
    for previous, current in itertools.pairwise(words):
        run = run + 1 if current == previous else 1
        longest_run = max(longest_run, run)
    if longest_run > _REPETITION_LIMIT:
        problems.append(
            f"one word repeats {longest_run} times in a row, which is a "
            "recogniser loop rather than speech"
        )
    elif len(words) >= _LOOP_FLOOR and len(set(words)) / len(words) < _MIN_DISTINCT_RATIO:
        problems.append(
            f"only {len(set(words))} distinct words in {len(words)}, which is a "
            "recogniser loop rather than speech"
        )
    return problems


def build_transcriber(settings: Settings) -> Transcriber | None:
    """Construct the configured transcriber, or ``None`` when disabled.

    Returning ``None`` rather than a stub is deliberate: callers must decide
    what to do without recognition, and the answers differ. Enrolment can
    require the caller to supply a transcript; consent verification cannot
    proceed at all, because an unverified consent record is worse than none.

    The transcriber is not loaded here; call
    :meth:`TransformersTranscriber.load` during startup so that a
    weight-loading failure surfaces before the process accepts traffic.
    """
    if not settings.asr_enabled:
        return None
    return TransformersTranscriber(
        model_id=settings.asr_model_id,
        revision=settings.asr_revision,
        device=settings.device,
        language=settings.asr_language,
        trust_remote_code=settings.asr_trust_remote_code,
    )


class TransformersTranscriber:
    """Speech recognition through the ``transformers`` ASR pipeline.

    Implements :class:`mlvoice.protocols.Transcriber`. Loads lazily and once,
    because model construction dominates the cost of a single transcription and
    a service transcribes on every enrolment.

    Args:
        model_id: Hub repository id. See the module docstring for candidates.
        revision: Commit to pin. Recommended for the same reason it is for the
            synthesis model, and required when ``trust_remote_code`` is set.
        device: ``cpu``, ``cuda`` or ``mps``.
        language: Language hint, where the model accepts one. Whisper will
            otherwise detect the language, and it detects Malayalam
            unreliably on short clips -- which reference clips always are.
        trust_remote_code: Needed by IndicConformer and models like it.
    """

    def __init__(
        self,
        *,
        model_id: str = "openai/whisper-large-v3",
        revision: str | None = None,
        device: str = "cpu",
        language: str | None = "ml",
        trust_remote_code: bool = False,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._language = language
        self._trust_remote_code = trust_remote_code
        self._pipeline: Any | None = None

    @property
    def name(self) -> str:
        """Identifier recorded in reports and audit records."""
        return f"transformers-asr:{self._model_id}"

    def is_ready(self) -> bool:
        """True once the model is resident."""
        return self._pipeline is not None

    def load(self) -> None:
        """Load the recogniser. Idempotent.

        Raises:
            BackendUnavailableError: The ``models`` extra is missing, or the
                model cannot be loaded.
        """
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise BackendUnavailableError(
                "transcription requires the model runtime: pip install -e '.[models]'",
                hint="run `mlvoice doctor` to see everything that is missing at once",
            ) from exc

        # Logged *before* the call, not after. On a cold cache this line is
        # followed by a multi-gigabyte download whose only other output is a
        # progress bar on stderr; without the model id in the log, a stalled
        # fetch is indistinguishable from a hung process.
        log.info(
            "loading transcriber",
            model_id=self._model_id,
            revision=self._revision,
            device=self._device,
            hint="a cold cache downloads the weights first; this can take minutes",
        )
        try:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                revision=self._revision,
                device=torch.device(self._device),
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:
            raise BackendUnavailableError(
                "could not load the transcription model",
                model_id=self._model_id,
                revision=self._revision,
                reason=str(exc),
            ) from exc
        log.info(
            "transcriber loaded",
            model_id=self._model_id,
            revision=self._revision,
            device=self._device,
        )

    def transcribe(self, audio: Audio) -> str:
        """Return the recognised text for ``audio``.

        Raises:
            BackendUnavailableError: The model is not loaded and cannot be.
        """
        self.load()
        assert self._pipeline is not None

        working = resample(audio, ASR_MODEL_SAMPLE_RATE)
        payload = {"raw": working.samples, "sampling_rate": working.sample_rate}
        generate: dict[str, Any] = dict(_DECODING_GUARDS)
        if self._language is not None:
            generate["language"] = self._language
            # Without this, a multilingual model is free to *translate* rather
            # than transcribe, and the output arrives in another language
            # entirely -- which reads as a broken recogniser rather than as a
            # missing argument.
            generate["task"] = "transcribe"

        try:
            result = self._pipeline(payload, generate_kwargs=generate)
        except (TypeError, ValueError) as exc:
            # Some architectures accept none of this. Retrying bare is better
            # than failing, but the retry has no language pinned, so its output
            # may come back in whatever script the model guessed -- which is
            # why the caller is told the hint was lost rather than being handed
            # a transcript that silently means something else.
            log.warning(
                "transcriber rejected the decoding options; retrying without them",
                model_id=self._model_id,
                reason=str(exc),
                consequence="language is auto-detected for this transcript",
            )
            result = self._pipeline(payload)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip()
