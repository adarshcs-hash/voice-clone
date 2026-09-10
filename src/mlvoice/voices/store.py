"""Voice and consent persistence.

SQLAlchemy 2.0 with a SQLite default so a developer needs no infrastructure,
and a Postgres URL in production with no code change. The API layer never sees
ORM rows: :class:`VoiceStore` returns the plain :class:`Voice` and
:class:`~mlvoice.voices.consent.ConsentRecord` domain objects, which keeps the
schema free to change and keeps sessions from escaping into request handlers.

Deleting a voice deletes the reference audio from disk as well as the row. A
voice is biometric data: "delete" has to mean the recording is gone, both
because that is what the subject asked for and because India's DPDP Act treats
it as personal data with a deletion right.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from mlvoice.errors import VoiceNotFoundError
from mlvoice.logging import get_logger
from mlvoice.voices.consent import ConsentRecord, ConsentStatus, ConsentVerification

__all__ = ["Voice", "VoiceStatus", "VoiceStore"]

log = get_logger(__name__)


class VoiceStatus(StrEnum):
    """Whether a voice may be used for synthesis."""

    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class Voice:
    """An enrolled voice, as the rest of the system sees it."""

    id: str
    name: str
    owner_id: str
    reference_audio_path: str
    reference_text: str
    reference_duration_seconds: float
    sample_rate: int
    status: VoiceStatus
    consent_id: str | None
    created_at: datetime
    updated_at: datetime
    dialect: str = "unknown"
    gender: str = "unknown"
    quality: dict[str, float] | None = None

    @property
    def is_usable(self) -> bool:
        """True if this voice may be selected for synthesis."""
        return self.status is VoiceStatus.ACTIVE


class Base(DeclarativeBase):
    """Declarative base for the persistence layer."""


class VoiceRow(Base):
    """``voices`` table."""

    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    reference_audio_path: Mapped[str] = mapped_column(String(1024))
    reference_text: Mapped[str] = mapped_column(String(4096))
    reference_duration_seconds: Mapped[float] = mapped_column(Float)
    sample_rate: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=VoiceStatus.ACTIVE.value)
    consent_id: Mapped[str | None] = mapped_column(
        ForeignKey("consents.id", ondelete="SET NULL"), nullable=True
    )
    dialect: Mapped[str] = mapped_column(String(32), default="unknown")
    gender: Mapped[str] = mapped_column(String(16), default="unknown")
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ConsentRow(Base):
    """``consents`` table."""

    __tablename__ = "consents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    voice_id: Mapped[str] = mapped_column(String(64), index=True)
    subject_name: Mapped[str] = mapped_column(String(256))
    phrase_text: Mapped[str] = mapped_column(String(2048))
    nonce: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    verification: Mapped[dict[str, Any]] = mapped_column(JSON)
    consent_audio_path: Mapped[str] = mapped_column(String(1024))
    jurisdiction: Mapped[str] = mapped_column(String(8), default="IN")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _as_aware(value: datetime) -> datetime:
    """SQLite drops timezone information; restore UTC on the way out."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class VoiceStore:
    """Repository over the voice and consent tables.

    Args:
        database_url: SQLAlchemy URL.
        create_tables: Create the schema if missing. Convenient for SQLite and
            for tests; a production Postgres deployment should run migrations
            instead and pass ``False``.
    """

    def __init__(self, database_url: str, *, create_tables: bool = True) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(database_url, future=True, connect_args=connect_args)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False, future=True)
        if create_tables:
            Base.metadata.create_all(self._engine)

    def session(self) -> Session:
        """Open a new session. Callers are responsible for closing it."""
        return self._sessions()

    # -- voices -------------------------------------------------------------

    def create_voice(self, voice: Voice) -> Voice:
        """Insert a voice."""
        with self._sessions.begin() as session:
            session.add(
                VoiceRow(
                    id=voice.id,
                    name=voice.name,
                    owner_id=voice.owner_id,
                    reference_audio_path=voice.reference_audio_path,
                    reference_text=voice.reference_text,
                    reference_duration_seconds=voice.reference_duration_seconds,
                    sample_rate=voice.sample_rate,
                    status=voice.status.value,
                    consent_id=voice.consent_id,
                    dialect=voice.dialect,
                    gender=voice.gender,
                    quality=voice.quality,
                    created_at=voice.created_at,
                    updated_at=voice.updated_at,
                )
            )
        log.info("voice enrolled", voice_id=voice.id, owner_id=voice.owner_id)
        return voice

    def get_voice(self, voice_id: str, *, owner_id: str | None = None) -> Voice:
        """Fetch one voice.

        Args:
            voice_id: Identifier.
            owner_id: When given, the voice must belong to this owner. Passing
                it is what stops one tenant reading another's voices.

        Raises:
            VoiceNotFoundError: No such voice, or it belongs to someone else.
                Both cases return the same error so the endpoint does not leak
                the existence of other tenants' voices.
        """
        with self._sessions() as session:
            row = session.get(VoiceRow, voice_id)
            if row is None or (owner_id is not None and row.owner_id != owner_id):
                raise VoiceNotFoundError("voice not found", voice_id=voice_id)
            return _to_voice(row)

    def list_voices(self, *, owner_id: str, limit: int = 50, offset: int = 0) -> Sequence[Voice]:
        """List an owner's voices, newest first."""
        with self._sessions() as session:
            statement = (
                select(VoiceRow)
                .where(VoiceRow.owner_id == owner_id)
                .order_by(VoiceRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [_to_voice(row) for row in session.scalars(statement)]

    def set_voice_status(self, voice_id: str, status: VoiceStatus) -> Voice:
        """Enable or disable a voice.

        Raises:
            VoiceNotFoundError: No such voice.
        """
        with self._sessions.begin() as session:
            row = session.get(VoiceRow, voice_id)
            if row is None:
                raise VoiceNotFoundError("voice not found", voice_id=voice_id)
            row.status = status.value
            row.updated_at = datetime.now(UTC)
            return _to_voice(row)

    def delete_voice(self, voice_id: str, *, owner_id: str | None = None) -> None:
        """Delete a voice, its consent record and its reference audio from disk.

        Raises:
            VoiceNotFoundError: No such voice, or it belongs to someone else.
        """
        with self._sessions.begin() as session:
            row = session.get(VoiceRow, voice_id)
            if row is None or (owner_id is not None and row.owner_id != owner_id):
                raise VoiceNotFoundError("voice not found", voice_id=voice_id)
            paths = [Path(row.reference_audio_path)]
            if row.consent_id is not None:
                consent = session.get(ConsentRow, row.consent_id)
                if consent is not None:
                    paths.append(Path(consent.consent_audio_path))
                    session.delete(consent)
            session.delete(row)

        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # The row is already gone; surface the orphan rather than
                # failing a deletion the caller has a right to.
                log.error("could not delete voice audio", path=str(path), reason=str(exc))
        log.info("voice deleted", voice_id=voice_id)

    # -- consent ------------------------------------------------------------

    def save_consent(self, record: ConsentRecord) -> ConsentRecord:
        """Insert a consent record."""
        with self._sessions.begin() as session:
            session.add(
                ConsentRow(
                    id=record.id,
                    voice_id=record.voice_id,
                    subject_name=record.subject_name,
                    phrase_text=record.phrase_text,
                    nonce=list(record.nonce),
                    status=record.status.value,
                    verification=record.verification.as_dict(),
                    consent_audio_path=record.consent_audio_path,
                    jurisdiction=record.jurisdiction,
                    created_at=record.created_at,
                    expires_at=record.expires_at,
                    revoked_at=record.revoked_at,
                )
            )
        return record

    def get_consent(self, consent_id: str) -> ConsentRecord:
        """Fetch a consent record.

        Raises:
            VoiceNotFoundError: No such record.
        """
        with self._sessions() as session:
            row = session.get(ConsentRow, consent_id)
            if row is None:
                raise VoiceNotFoundError("consent record not found", consent_id=consent_id)
            return _to_consent(row)

    def revoke_consent(self, consent_id: str) -> ConsentRecord:
        """Revoke consent and disable every voice that relied on it.

        Revocation has to cascade: a consent record that is revoked while the
        voice stays usable is not a consent mechanism.

        Raises:
            VoiceNotFoundError: No such record.
        """
        with self._sessions.begin() as session:
            row = session.get(ConsentRow, consent_id)
            if row is None:
                raise VoiceNotFoundError("consent record not found", consent_id=consent_id)
            row.revoked_at = datetime.now(UTC)
            row.status = ConsentStatus.REVOKED.value
            affected = session.scalars(
                select(VoiceRow).where(VoiceRow.consent_id == consent_id)
            ).all()
            for voice_row in affected:
                voice_row.status = VoiceStatus.DISABLED.value
                voice_row.updated_at = datetime.now(UTC)
            record = _to_consent(row)
        log.info("consent revoked", consent_id=consent_id, voices_disabled=len(affected))
        return record


def _to_voice(row: VoiceRow) -> Voice:
    return Voice(
        id=row.id,
        name=row.name,
        owner_id=row.owner_id,
        reference_audio_path=row.reference_audio_path,
        reference_text=row.reference_text,
        reference_duration_seconds=row.reference_duration_seconds,
        sample_rate=row.sample_rate,
        status=VoiceStatus(row.status),
        consent_id=row.consent_id,
        dialect=row.dialect,
        gender=row.gender,
        quality=dict(row.quality) if row.quality else None,
        created_at=_as_aware(row.created_at),
        updated_at=_as_aware(row.updated_at),
    )


def _to_consent(row: ConsentRow) -> ConsentRecord:
    payload = dict(row.verification)
    verification = ConsentVerification(
        passed=bool(payload.get("passed", False)),
        phrase_cer=float(payload.get("phrase_cer", 1.0)),
        nonce_words_found=int(payload.get("nonce_words_found", 0)),
        speaker_similarity=(
            float(payload["speaker_similarity"])
            if payload.get("speaker_similarity") is not None
            else None
        ),
        reasons=tuple(payload.get("reasons", ())),
    )
    return ConsentRecord(
        id=row.id,
        voice_id=row.voice_id,
        subject_name=row.subject_name,
        phrase_text=row.phrase_text,
        nonce=tuple(row.nonce),
        status=ConsentStatus(row.status),
        verification=verification,
        consent_audio_path=row.consent_audio_path,
        created_at=_as_aware(row.created_at),
        expires_at=_as_aware(row.expires_at),
        revoked_at=_as_aware(row.revoked_at) if row.revoked_at else None,
        jurisdiction=row.jurisdiction,
    )
