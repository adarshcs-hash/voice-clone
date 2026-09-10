"""Corpus manifest schema and JSONL IO.

The manifest is the contract between corpus preparation and training. Making it
an explicit, validated schema rather than an ad-hoc CSV buys three things that
matter once more than one dataset is involved:

*   **Provenance.** ``source`` and ``license_id`` travel with every utterance,
    so a dataset with non-commercial terms cannot silently end up in a
    commercial training run. This is a legal requirement, not a nicety -- see
    ``docs/data-and-licensing.md``.
*   **Reproducibility.** Every derived field (normalised text, phonemes, the
    quality measurements, the ASR check) is recorded, so a training run can be
    reconstructed from the manifest alone.
*   **Filtering.** Splits and quality gates operate on the manifest, not on the
    filesystem, so an experiment is a query rather than a copy of the audio.

JSONL is used rather than a single JSON document so a multi-hundred-thousand
row manifest streams instead of loading whole.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlvoice.errors import ValidationError

__all__ = [
    "Dialect",
    "Gender",
    "Split",
    "Utterance",
    "count_by",
    "read_manifest",
    "total_duration_hours",
    "write_manifest",
]


class Split(StrEnum):
    """Dataset partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Gender(StrEnum):
    """Speaker-reported gender, used for corpus balance reporting only."""

    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNKNOWN = "unknown"


class Dialect(StrEnum):
    """Regional variety.

    Malayalam dialect variation is large enough to be audible in synthesis, and
    open corpora skew heavily to the formal read register of news broadcasting.
    Tracking this field is what makes the skew visible; without it a corpus
    looks balanced because nobody measured.
    """

    THIRUVANANTHAPURAM = "thiruvananthapuram"
    KOLLAM = "kollam"
    KOTTAYAM = "kottayam"
    ERNAKULAM = "ernakulam"
    THRISSUR = "thrissur"
    PALAKKAD = "palakkad"
    KOZHIKODE = "kozhikode"
    MALABAR = "malabar"
    KASARAGOD = "kasaragod"
    STANDARD_FORMAL = "standard_formal"
    UNKNOWN = "unknown"


NonEmptyStr = Annotated[str, Field(min_length=1)]


class Utterance(BaseModel):
    """One audio/text pair, with provenance and derived fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    # -- identity and provenance -------------------------------------------
    id: NonEmptyStr
    audio_path: NonEmptyStr
    source: NonEmptyStr
    """Dataset name, e.g. ``indicvoices-r`` or ``studio-2026-batch-01``."""
    license_id: NonEmptyStr
    """SPDX identifier or a documented key from ``docs/data-and-licensing.md``."""
    commercial_use_permitted: bool
    """Set from the source licence. Training runs filter on this field."""

    # -- text ---------------------------------------------------------------
    text: NonEmptyStr
    """Verbatim transcript as distributed by the source."""
    normalized_text: str = ""
    """Output of the text frontend; the training target."""
    phonemes: str = ""
    """ASCII phoneme string from :mod:`mlvoice.text.g2p`."""

    # -- speaker ------------------------------------------------------------
    speaker_id: NonEmptyStr
    gender: Gender = Gender.UNKNOWN
    dialect: Dialect = Dialect.UNKNOWN
    consent_id: str | None = None
    """Consent record identifier. Required for any voice offered for cloning."""

    # -- audio --------------------------------------------------------------
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    denoiser: str = "none"
    loudness_lufs: float | None = None

    # -- verification -------------------------------------------------------
    quality: dict[str, float] = Field(default_factory=dict)
    asr_text: str | None = None
    asr_cer: float | None = Field(default=None, ge=0)
    """CER between ``normalized_text`` and a re-transcription. The main filter."""

    # -- bookkeeping --------------------------------------------------------
    split: Split = Split.TRAIN
    language: str = "ml"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("language")
    @classmethod
    def _only_malayalam(cls, value: str) -> str:
        if value != "ml":
            raise ValueError("this corpus is Malayalam-only; language must be 'ml'")
        return value

    def to_json_line(self) -> str:
        """Serialise as one JSONL record."""
        return self.model_dump_json(exclude_none=False)


def read_manifest(path: str | Path) -> Iterator[Utterance]:
    """Stream utterances from a JSONL manifest.

    Args:
        path: Manifest file.

    Yields:
        Validated :class:`Utterance` records.

    Raises:
        ValidationError: A line is not valid JSON or fails schema validation.
            The message carries the line number, because a manifest with one bad
            row in three hundred thousand is otherwise unfindable.
    """
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload: Any = json.loads(stripped)
                yield Utterance.model_validate(payload)
            except Exception as exc:
                raise ValidationError(
                    "invalid manifest line",
                    path=str(manifest),
                    line=line_number,
                    reason=str(exc),
                ) from exc


def write_manifest(path: str | Path, utterances: Iterable[Utterance]) -> int:
    """Write utterances to a JSONL manifest, creating parent directories.

    Returns:
        The number of records written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for utterance in utterances:
            handle.write(utterance.to_json_line())
            handle.write("\n")
            count += 1
    return count


def total_duration_hours(utterances: Iterable[Utterance]) -> float:
    """Total duration in hours."""
    return sum(u.duration_seconds for u in utterances) / 3600.0


def count_by(utterances: Iterable[Utterance], field: str) -> dict[str, int]:
    """Count utterances grouped by one field, for corpus balance reports.

    Raises:
        ValidationError: ``field`` is not a field of :class:`Utterance`.
    """
    if field not in Utterance.model_fields:
        raise ValidationError("unknown manifest field", field=field)
    counts: dict[str, int] = {}
    for utterance in utterances:
        value = getattr(utterance, field)
        key = value.value if isinstance(value, StrEnum) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
