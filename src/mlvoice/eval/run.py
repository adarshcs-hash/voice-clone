"""Evaluation harness.

Two reports, deliberately separate:

*   :meth:`EvaluationHarness.evaluate_frontend` scores the **text frontend**
    alone. It needs no model, no GPU and no listener, runs in milliseconds, and
    catches the class of bug that produces confidently mispronounced audio. This
    runs in CI on every commit.
*   :meth:`EvaluationHarness.evaluate_synthesis` scores **generated audio**:
    intelligibility via ASR round-trip CER, speaker similarity against the
    reference clip, objective audio quality, and real-time factor.

Both aggregate per :class:`~mlvoice.eval.testset.Category`, because a single
average hides exactly the information that decides what to work on next.

What this harness cannot tell you
---------------------------------
Round-trip CER is a proxy, not the truth. An ASR model can share the TTS
model's confusions and transcribe a mispronounced word back to the correct
characters, and it is blind to prosody entirely. Treat CER as the fast signal
between rating rounds and native-speaker judgements
(:func:`~mlvoice.eval.testset.scoring_instructions`) as the gate.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mlvoice.audio.quality import measure_quality
from mlvoice.eval.metrics import character_error_rate, word_error_rate
from mlvoice.eval.testset import TEST_CASES, Category, TestCase
from mlvoice.logging import get_logger
from mlvoice.protocols import MosEstimator, SpeakerVerifier, Transcriber
from mlvoice.text.g2p import Notation
from mlvoice.text.pipeline import TextPipeline
from mlvoice.tts.base import ReferencePrompt, SynthesisRequest, Synthesizer

__all__ = [
    "EvaluationHarness",
    "FrontendCaseResult",
    "FrontendReport",
    "SynthesisCaseResult",
    "SynthesisReport",
]

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Frontend evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrontendCaseResult:
    """One case's frontend output."""

    id: str
    category: str
    input_text: str
    normalized: str
    routed: str
    chunk_count: int
    phonemes: str
    expected_text: str | None
    matches_expected: bool | None
    """``None`` when the case carries no expectation."""
    contains_malayalam: bool
    normalization_changes: int


@dataclass(frozen=True, slots=True)
class FrontendReport:
    """Aggregate frontend report."""

    results: tuple[FrontendCaseResult, ...]

    @property
    def asserted(self) -> tuple[FrontendCaseResult, ...]:
        """Cases that carry an expected normalised form."""
        return tuple(r for r in self.results if r.matches_expected is not None)

    @property
    def failures(self) -> tuple[FrontendCaseResult, ...]:
        """Cases whose output differs from the expectation."""
        return tuple(r for r in self.asserted if r.matches_expected is False)

    @property
    def pass_rate(self) -> float:
        """Share of asserted cases that match. ``1.0`` when none are asserted."""
        asserted = self.asserted
        if not asserted:
            return 1.0
        return sum(1 for r in asserted if r.matches_expected) / len(asserted)

    def by_category(self) -> dict[str, dict[str, float | int]]:
        """Per-category counts and pass rates."""
        buckets: dict[str, list[FrontendCaseResult]] = {}
        for result in self.results:
            buckets.setdefault(result.category, []).append(result)
        summary: dict[str, dict[str, float | int]] = {}
        for category, items in sorted(buckets.items()):
            asserted = [i for i in items if i.matches_expected is not None]
            summary[category] = {
                "cases": len(items),
                "asserted": len(asserted),
                "passed": sum(1 for i in asserted if i.matches_expected),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialisable report."""
        return {
            "kind": "frontend",
            "cases": len(self.results),
            "asserted": len(self.asserted),
            "pass_rate": round(self.pass_rate, 4),
            "failures": [
                {"id": r.id, "expected": r.expected_text, "got": r.routed} for r in self.failures
            ],
            "by_category": self.by_category(),
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """JSON report."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------
# Synthesis evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynthesisCaseResult:
    """One case's synthesis measurements."""

    id: str
    category: str
    input_text: str
    normalized_text: str
    audio_seconds: float
    generation_seconds: float
    real_time_factor: float
    quality: dict[str, float]
    asr_text: str | None = None
    cer: float | None = None
    wer: float | None = None
    speaker_similarity: float | None = None
    mos: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SynthesisReport:
    """Aggregate synthesis report."""

    backend: str
    model_id: str
    results: tuple[SynthesisCaseResult, ...]
    started_at: float = field(default_factory=time.time)

    @property
    def failures(self) -> tuple[SynthesisCaseResult, ...]:
        """Cases that raised during synthesis."""
        return tuple(r for r in self.results if r.error is not None)

    def _mean(self, attribute: str) -> float | None:
        values = [
            getattr(r, attribute)
            for r in self.results
            if getattr(r, attribute) is not None and r.error is None
        ]
        return sum(values) / len(values) if values else None

    def by_category(self) -> dict[str, dict[str, float | int | None]]:
        """Per-category means of the headline metrics."""
        buckets: dict[str, list[SynthesisCaseResult]] = {}
        for result in self.results:
            buckets.setdefault(result.category, []).append(result)

        def mean(items: Sequence[SynthesisCaseResult], attribute: str) -> float | None:
            values = [
                getattr(i, attribute)
                for i in items
                if getattr(i, attribute) is not None and i.error is None
            ]
            return round(sum(values) / len(values), 4) if values else None

        return {
            category: {
                "cases": len(items),
                "errors": sum(1 for i in items if i.error is not None),
                "cer": mean(items, "cer"),
                "speaker_similarity": mean(items, "speaker_similarity"),
                "mos": mean(items, "mos"),
                "real_time_factor": mean(items, "real_time_factor"),
            }
            for category, items in sorted(buckets.items())
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable report."""
        return {
            "kind": "synthesis",
            "backend": self.backend,
            "model_id": self.model_id,
            "cases": len(self.results),
            "errors": len(self.failures),
            "mean_cer": _round(self._mean("cer")),
            "mean_wer": _round(self._mean("wer")),
            "mean_speaker_similarity": _round(self._mean("speaker_similarity")),
            "mean_mos": _round(self._mean("mos")),
            "mean_real_time_factor": _round(self._mean("real_time_factor")),
            "by_category": self.by_category(),
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """JSON report."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


class EvaluationHarness:
    """Runs the test set against a text frontend and a synthesis backend.

    Args:
        pipeline: Text frontend under test.
        transcriber: ASR for the intelligibility round-trip. Without one, CER is
            not reported.
        speaker_verifier: Speaker similarity against the reference prompt.
        mos_estimator: No-reference perceptual quality estimator.
    """

    def __init__(
        self,
        *,
        pipeline: TextPipeline | None = None,
        transcriber: Transcriber | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        mos_estimator: MosEstimator | None = None,
    ) -> None:
        self._pipeline = pipeline or TextPipeline()
        self._transcriber = transcriber
        self._speaker = speaker_verifier
        self._mos = mos_estimator

    def evaluate_frontend(self, cases: Sequence[TestCase] = TEST_CASES) -> FrontendReport:
        """Score the text frontend. Needs no model and no audio."""
        results: list[FrontendCaseResult] = []
        for case in cases:
            processed = self._pipeline.process(case.text)
            phonemes = " | ".join(
                processed.phoneme_string(i, notation=Notation.ASCII)
                for i in range(len(processed.chunks))
            )
            results.append(
                FrontendCaseResult(
                    id=case.id,
                    category=case.category.value,
                    input_text=case.text,
                    normalized=processed.normalized,
                    routed=processed.routed,
                    chunk_count=len(processed.chunks),
                    phonemes=phonemes,
                    expected_text=case.expected_text,
                    matches_expected=(
                        None
                        if case.expected_text is None
                        else processed.routed == case.expected_text
                    ),
                    contains_malayalam=processed.contains_malayalam,
                    normalization_changes=processed.normalization.total_changes,
                )
            )
        report = FrontendReport(results=tuple(results))
        log.info(
            "frontend evaluation complete",
            cases=len(results),
            asserted=len(report.asserted),
            pass_rate=round(report.pass_rate, 4),
            failures=[r.id for r in report.failures],
        )
        return report

    def evaluate_synthesis(
        self,
        synthesizer: Synthesizer,
        cases: Sequence[TestCase] = TEST_CASES,
        *,
        prompt: ReferencePrompt | None = None,
        categories: Sequence[Category] | None = None,
    ) -> SynthesisReport:
        """Synthesise every case and measure the result.

        A case that raises is recorded with its error rather than aborting the
        run: a report covering 70 of 75 cases plus five named failures is far
        more useful than a traceback.
        """
        selected = [c for c in cases if categories is None or c.category in set(categories)]
        results: list[SynthesisCaseResult] = []
        for case in selected:
            results.append(self._evaluate_one(synthesizer, case, prompt))
        report = SynthesisReport(
            backend=synthesizer.info.backend,
            model_id=synthesizer.info.model_id,
            results=tuple(results),
        )
        log.info(
            "synthesis evaluation complete",
            **{
                k: v
                for k, v in report.to_dict().items()
                if k in {"cases", "errors", "mean_cer", "mean_real_time_factor"}
            },
        )
        return report

    def _evaluate_one(
        self, synthesizer: Synthesizer, case: TestCase, prompt: ReferencePrompt | None
    ) -> SynthesisCaseResult:
        try:
            processed = self._pipeline.process(case.text)
            result = synthesizer.synthesize(
                SynthesisRequest(text=processed, prompt=prompt, seed=1234)
            )
        except Exception as exc:
            log.warning("evaluation case failed", case_id=case.id, reason=str(exc))
            return SynthesisCaseResult(
                id=case.id,
                category=case.category.value,
                input_text=case.text,
                normalized_text="",
                audio_seconds=0.0,
                generation_seconds=0.0,
                real_time_factor=float("inf"),
                quality={},
                error=str(exc),
            )

        asr_text: str | None = None
        cer: float | None = None
        wer: float | None = None
        if self._transcriber is not None:
            asr_text = self._transcriber.transcribe(result.audio)
            cer = character_error_rate(processed.routed, asr_text).rate
            wer = word_error_rate(processed.routed, asr_text).rate

        similarity: float | None = None
        if self._speaker is not None and prompt is not None:
            similarity = self._speaker.similarity(prompt.audio, result.audio)

        mos: float | None = self._mos.score(result.audio) if self._mos is not None else None

        return SynthesisCaseResult(
            id=case.id,
            category=case.category.value,
            input_text=case.text,
            normalized_text=processed.routed,
            audio_seconds=round(result.audio.duration_seconds, 3),
            generation_seconds=round(result.generation_seconds, 3),
            real_time_factor=round(result.real_time_factor, 4),
            quality=measure_quality(result.audio).as_dict(),
            asr_text=asr_text,
            cer=cer,
            wer=wer,
            speaker_similarity=similarity,
            mos=mos,
        )
