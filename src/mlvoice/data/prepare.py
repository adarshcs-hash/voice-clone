"""Corpus preparation.

Turns a heterogeneous pile of audio and transcripts into a manifest a training
run can consume, applying the same treatment to every source so the model sees
one consistent distribution.

Per utterance::

    load -> resample -> trim silence -> denoise -> loudness-normalise
         -> quality gate -> text frontend -> ASR verification -> manifest row

Two decisions in here matter more than the rest.

**ASR verification is the real filter.** Source transcripts are wrong often
enough to matter: misaligned segments, truncated sentences, the wrong take.
Re-transcribing the audio and rejecting rows whose CER against the transcript
exceeds a threshold removes those. This one step is worth more to final quality
than any hyperparameter.

**Splits are speaker-disjoint.** A validation set sharing speakers with train
measures memorisation, not generalisation, and for a *voice cloning* model that
is precisely the quantity being faked. Assignment hashes the speaker id, so it
is deterministic across runs and stable as the corpus grows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from mlvoice.audio.denoise import Denoiser, NoopDenoiser
from mlvoice.audio.io import load_audio, save_audio
from mlvoice.audio.loudness import BROADCAST_TARGET_LUFS, normalize_loudness
from mlvoice.audio.quality import QualityGate, measure_quality
from mlvoice.audio.vad import Vad, trim_silence
from mlvoice.data.manifest import Dialect, Gender, Split, Utterance
from mlvoice.errors import MlvoiceError
from mlvoice.eval.metrics import character_error_rate
from mlvoice.logging import get_logger
from mlvoice.protocols import Transcriber
from mlvoice.text.g2p import Notation
from mlvoice.text.pipeline import TextPipeline

__all__ = [
    "CorpusPreparer",
    "PrepareConfig",
    "PrepareResult",
    "RawEntry",
    "Rejection",
    "Transcriber",
    "assign_split",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RawEntry:
    """One input row, before any processing."""

    id: str
    audio_path: Path
    text: str
    speaker_id: str
    source: str
    license_id: str
    commercial_use_permitted: bool
    gender: Gender = Gender.UNKNOWN
    dialect: Dialect = Dialect.UNKNOWN
    consent_id: str | None = None


@dataclass(frozen=True, slots=True)
class Rejection:
    """An entry that did not make it into the corpus, and why."""

    entry_id: str
    stage: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    """Corpus preparation settings.

    Attributes:
        output_dir: Where processed audio is written.
        target_sample_rate: Corpus-wide rate. 24 kHz is the common TTS choice
            and what IndicF5-family models expect.
        target_lufs: Loudness target applied to every clip.
        quality_gate: Objective thresholds for admission.
        max_asr_cer: Reject when re-transcription disagrees with the transcript
            by more than this. 0.15 is a practical starting point for Malayalam
            read speech; tighten it once the recogniser's own error rate on
            clean data is known.
        trim: Trim leading and trailing silence.
        validation_fraction: Share of *speakers* held out for validation.
        test_fraction: Share of *speakers* held out for test.
        split_salt: Changes the speaker-to-split assignment. Fix it per project
            and never change it mid-project.
    """

    output_dir: Path
    target_sample_rate: int = 24_000
    target_lufs: float = BROADCAST_TARGET_LUFS
    quality_gate: QualityGate = field(default_factory=QualityGate)
    max_asr_cer: float = 0.15
    trim: bool = True
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    split_salt: str = "mlvoice-v1"


@dataclass(slots=True)
class PrepareResult:
    """Outcome of a preparation run."""

    accepted: list[Utterance] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def accepted_hours(self) -> float:
        """Total accepted duration in hours."""
        return sum(u.duration_seconds for u in self.accepted) / 3600.0

    def rejection_summary(self) -> dict[str, int]:
        """Rejection counts by stage, for the run report."""
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.stage] = counts.get(rejection.stage, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def assign_split(speaker_id: str, config: PrepareConfig) -> Split:
    """Deterministically assign a speaker to a split.

    Hashing the speaker id (not the utterance id) is what makes the splits
    speaker-disjoint, and hashing rather than shuffling makes the assignment
    stable when the corpus grows.
    """
    digest = hashlib.sha256(f"{config.split_salt}:{speaker_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if position < config.test_fraction:
        return Split.TEST
    if position < config.test_fraction + config.validation_fraction:
        return Split.VALIDATION
    return Split.TRAIN


class CorpusPreparer:
    """Runs the preparation pipeline over raw entries.

    Args:
        config: Preparation settings.
        text_pipeline: Malayalam text frontend. Defaults to a stock pipeline.
        denoiser: Applied after trimming. Defaults to a pass-through.
        vad: Detector used for trimming. Defaults to the energy VAD.
        transcriber: ASR used for verification. When ``None``, verification is
            skipped and the manifest records that -- which the operator should
            treat as an unverified corpus, not a clean one.
    """

    def __init__(
        self,
        config: PrepareConfig,
        *,
        text_pipeline: TextPipeline | None = None,
        denoiser: Denoiser | None = None,
        vad: Vad | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._config = config
        self._text = text_pipeline or TextPipeline()
        self._denoiser = denoiser or NoopDenoiser()
        self._vad = vad
        self._transcriber = transcriber

    def run(self, entries: Iterable[RawEntry]) -> PrepareResult:
        """Process every entry, collecting accepted rows and rejections."""
        result = PrepareResult()
        for entry in entries:
            try:
                utterance = self._process_one(entry, result)
            except MlvoiceError as exc:
                result.rejected.append(Rejection(entry.id, "error", (exc.message,)))
                log.warning("entry failed", entry_id=entry.id, code=exc.code, reason=exc.message)
                continue
            if utterance is not None:
                result.accepted.append(utterance)
        log.info(
            "corpus preparation complete",
            accepted=len(result.accepted),
            rejected=len(result.rejected),
            hours=round(result.accepted_hours, 3),
            rejections=result.rejection_summary(),
        )
        return result

    def _process_one(self, entry: RawEntry, result: PrepareResult) -> Utterance | None:
        raw = load_audio(entry.audio_path, target_sample_rate=self._config.target_sample_rate)

        # Noise metrics are measured before trimming: the SNR estimator needs
        # the non-speech frames that trimming is about to remove.
        raw_report = measure_quality(raw)

        audio = trim_silence(raw, vad=self._vad) if self._config.trim else raw
        audio = self._denoiser.process(audio)
        audio, _ = normalize_loudness(audio, target_lufs=self._config.target_lufs)

        # Duration and silence share are properties of what will be trained on.
        final_report = measure_quality(audio)
        report = replace(
            raw_report,
            duration_seconds=audio.duration_seconds,
            silence_ratio=final_report.silence_ratio,
            peak_dbfs=final_report.peak_dbfs,
            clipping_ratio=final_report.clipping_ratio,
        )
        passed, reasons = self._config.quality_gate.evaluate(report)
        if not passed:
            result.rejected.append(Rejection(entry.id, "quality", tuple(reasons)))
            return None

        processed = self._text.process(entry.text)
        normalized_text = processed.routed
        phonemes = " | ".join(
            processed.phoneme_string(i, notation=Notation.ASCII)
            for i in range(len(processed.chunks))
        )

        asr_text: str | None = None
        asr_cer: float | None = None
        if self._transcriber is not None:
            asr_text = self._transcriber.transcribe(audio)
            asr_cer = character_error_rate(normalized_text, asr_text).rate
            if asr_cer > self._config.max_asr_cer:
                result.rejected.append(
                    Rejection(
                        entry.id,
                        "asr_mismatch",
                        (f"cer {asr_cer:.3f} > {self._config.max_asr_cer:.3f}",),
                    )
                )
                return None

        destination = self._config.output_dir / entry.source / f"{entry.id}.wav"
        save_audio(audio, destination)

        return Utterance(
            id=entry.id,
            audio_path=str(destination),
            source=entry.source,
            license_id=entry.license_id,
            commercial_use_permitted=entry.commercial_use_permitted,
            text=entry.text,
            normalized_text=normalized_text,
            phonemes=phonemes,
            speaker_id=entry.speaker_id,
            gender=entry.gender,
            dialect=entry.dialect,
            consent_id=entry.consent_id,
            duration_seconds=audio.duration_seconds,
            sample_rate=audio.sample_rate,
            denoiser=self._denoiser.name,
            loudness_lufs=self._config.target_lufs,
            quality=report.as_dict(),
            asr_text=asr_text,
            asr_cer=asr_cer,
            split=assign_split(entry.speaker_id, self._config),
        )


def iter_entries_from_tsv(
    path: str | Path,
    *,
    source: str,
    license_id: str,
    commercial_use_permitted: bool,
    audio_root: str | Path | None = None,
) -> Iterator[RawEntry]:
    """Read entries from a tab-separated ``id<TAB>audio<TAB>text<TAB>speaker`` file.

    The layout most Indic corpora ship in, or can be converted to in one
    ``awk``. Rows with the wrong number of columns are skipped with a warning
    rather than aborting a multi-hour run.
    """
    root = Path(audio_root) if audio_root else Path()
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns: Sequence[str] = line.rstrip("\n").split("\t")
            if len(columns) < 4:
                if line.strip():
                    log.warning("skipping malformed tsv row", path=str(manifest), line=line_number)
                continue
            utterance_id, audio, text, speaker = columns[:4]
            yield RawEntry(
                id=utterance_id,
                audio_path=root / audio,
                text=text,
                speaker_id=speaker,
                source=source,
                license_id=license_id,
                commercial_use_permitted=commercial_use_permitted,
            )
