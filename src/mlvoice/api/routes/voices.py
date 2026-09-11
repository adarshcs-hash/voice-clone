"""Voice lifecycle endpoints.

The enrolment flow is two calls, and the split is what makes consent meaningful:

1.  ``POST /v1/voices/challenge`` returns a phrase containing the subject's
    name, today's date and a random nonce, plus a signed token.
2.  ``POST /v1/voices`` takes the reference clip, its transcript, the token and
    a recording of the subject reading the phrase. The service checks that the
    phrase was spoken and that it is the same speaker as the reference.

A one-call enrolment cannot establish consent, because any audio could be
supplied for both roles. The nonce is what binds the recording to this
enrolment, at this time.

Deletion is real deletion: the row, the consent record and the audio files.
Revocation of consent cascades to disable every voice that relied on it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status

from mlvoice.api.deps import Caller, get_enrollment_service, get_store, rate_limit
from mlvoice.api.schemas import (
    ChallengeRequest,
    ChallengeResponse,
    TranscribeResponse,
    VoiceListResponse,
    VoiceResponse,
    WatermarkResponse,
)
from mlvoice.asr import assess_transcript
from mlvoice.audio.io import load_audio
from mlvoice.audio.quality import measure_quality
from mlvoice.errors import ValidationError
from mlvoice.logging import get_logger
from mlvoice.tts.base import ReferencePrompt
from mlvoice.voices.enrollment import EnrollmentRequest, EnrollmentService
from mlvoice.voices.store import Voice, VoiceStatus, VoiceStore

router = APIRouter(tags=["voices"])
log = get_logger(__name__)

_CHALLENGE_TTL_SECONDS = 3600
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def _to_response(voice: Voice, *, consent_verified: bool) -> VoiceResponse:
    return VoiceResponse(
        id=voice.id,
        name=voice.name,
        status=voice.status.value,
        reference_text=voice.reference_text,
        reference_duration_seconds=round(voice.reference_duration_seconds, 3),
        sample_rate=voice.sample_rate,
        dialect=voice.dialect,
        gender=voice.gender,
        consent_id=voice.consent_id,
        consent_verified=consent_verified,
        created_at=voice.created_at,
    )


async def _read_upload(upload: UploadFile, field: str) -> bytes:
    """Read an upload, refusing anything larger than the cap.

    Raises:
        ValidationError: The upload is empty or over the size limit.
    """
    data = await upload.read()
    if not data:
        raise ValidationError("uploaded file is empty", field=field)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise ValidationError(
            "uploaded file is too large",
            field=field,
            bytes=len(data),
            limit=_MAX_UPLOAD_BYTES,
        )
    return data


@router.post(
    "/v1/voices/challenge",
    response_model=ChallengeResponse,
    summary="Issue a consent challenge",
)
def issue_challenge(
    body: ChallengeRequest,
    service: Annotated[EnrollmentService, Depends(get_enrollment_service)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> ChallengeResponse:
    """Return a phrase for the subject to read, and the token that binds it."""
    issued = service.issue_challenge(body.subject_name)
    return ChallengeResponse(
        phrase=issued.phrase.text,
        nonce=list(issued.phrase.nonce),
        token=issued.token,
        expires_in_seconds=_CHALLENGE_TTL_SECONDS,
    )


@router.post(
    "/v1/voices",
    response_model=VoiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enrol a voice",
    responses={
        403: {"description": "Consent could not be verified, or the name is blocked."},
        422: {"description": "Reference audio failed the quality gate."},
    },
)
async def enroll_voice(
    service: Annotated[EnrollmentService, Depends(get_enrollment_service)],
    caller: Annotated[Caller, Depends(rate_limit)],
    name: Annotated[str, Form(min_length=1, max_length=200)],
    reference_audio: Annotated[UploadFile, File()],
    consent_token: Annotated[str | None, Form()] = None,
    consent_audio: Annotated[UploadFile | None, File()] = None,
    reference_text: Annotated[str | None, Form(max_length=4000)] = None,
    dialect: Annotated[str, Form()] = "unknown",
    gender: Annotated[str, Form()] = "unknown",
) -> VoiceResponse:
    """Enrol a voice from a reference clip and a verified consent recording.

    ``reference_text`` is the verbatim Malayalam transcript of
    ``reference_audio``. It may be omitted, in which case the clip is
    transcribed: the cloning backend conditions on this text, and a wrong
    transcript is the most common cause of a poor clone, so recognising it is
    better than asking a caller to type the transcript of their own recording.

    Clients that can show the transcript should call ``POST /v1/transcribe``
    first and submit the corrected text, which is better than either extreme.

    ``consent_token`` and ``consent_audio`` are optional in the signature and
    required in practice: enrolment refuses without them unless
    ``MLVOICE_REQUIRE_CONSENT`` is off, which production will not start with.
    They are optional here rather than mandatory so that a deployment which has
    turned consent off is not made to post a challenge it does not check --
    ``GET /v1/info`` reports ``consent_required`` so a client knows which it is.
    """
    voice = service.enroll(
        EnrollmentRequest(
            name=name,
            owner_id=caller.owner_id,
            reference_audio=await _read_upload(reference_audio, "reference_audio"),
            reference_text=reference_text,
            consent_token=consent_token,
            consent_audio=(
                await _read_upload(consent_audio, "consent_audio")
                if consent_audio is not None
                else None
            ),
            dialect=dialect,
            gender=gender,
        )
    )
    return _to_response(voice, consent_verified=voice.consent_id is not None)


@router.post(
    "/v1/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe a clip, for review before enrolment",
)
async def transcribe(
    request: Request,
    caller: Annotated[Caller, Depends(rate_limit)],
    audio: Annotated[UploadFile, File()],
) -> TranscribeResponse:
    """Recognise the speech in an uploaded clip and measure its quality.

    The intended flow is: upload, show this text, let the user fix it, then
    enrol with the corrected transcript. That is better than requiring the user
    to type it from scratch and better than using recognition output blind.

    Raises:
        ValidationError: Recognition is disabled on this deployment.
    """
    transcriber = request.app.state.transcriber
    if transcriber is None:
        raise ValidationError(
            "transcription is disabled on this deployment; supply the "
            "reference transcript with the enrolment request instead"
        )
    settings = request.app.state.settings
    data = await _read_upload(audio, "audio")
    clip = load_audio(data, target_sample_rate=settings.sample_rate)

    recognised = transcriber.transcribe(clip)
    # A transcript in the wrong script, or a decoder loop, is worse than no
    # transcript at all: it is fed to the synthesiser as ref_text, so it
    # corrupts the clone while looking like a filled-in field. Report it as
    # unusable and return nothing in `text`, rather than handing the caller
    # something they have no reason to distrust.
    problems = assess_transcript(recognised, expect_malayalam=settings.asr_language == "ml")
    usable = not problems
    advisories = list(problems)
    if usable:
        advisories.extend(
            ReferencePrompt(audio=clip, text=recognised, voice_id="preview").advisories()
        )
    else:
        log.warning(
            "discarded an unusable transcript",
            transcriber=getattr(transcriber, "name", "unknown"),
            problems=problems,
            recognised=recognised[:200],
        )
    return TranscribeResponse(
        text=recognised if usable else "",
        usable=usable,
        duration_seconds=round(clip.duration_seconds, 3),
        transcriber=getattr(transcriber, "name", "unknown"),
        quality=measure_quality(clip).as_dict(),
        advisories=advisories,
    )


@router.get("/v1/voices", response_model=VoiceListResponse, summary="List voices")
def list_voices(
    store: Annotated[VoiceStore, Depends(get_store)],
    caller: Annotated[Caller, Depends(rate_limit)],
    limit: int = 50,
    offset: int = 0,
) -> VoiceListResponse:
    """List the caller's voices, newest first."""
    voices = store.list_voices(owner_id=caller.owner_id, limit=min(limit, 200), offset=offset)
    return VoiceListResponse(
        voices=[_to_response(v, consent_verified=v.consent_id is not None) for v in voices],
        count=len(voices),
    )


@router.get("/v1/voices/{voice_id}", response_model=VoiceResponse, summary="Get a voice")
def get_voice(
    voice_id: str,
    store: Annotated[VoiceStore, Depends(get_store)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> VoiceResponse:
    """Return one voice belonging to the caller."""
    voice = store.get_voice(voice_id, owner_id=caller.owner_id)
    consent_active = False
    if voice.consent_id is not None:
        consent_active = store.get_consent(voice.consent_id).is_active
    return _to_response(voice, consent_verified=consent_active)


@router.post(
    "/v1/voices/{voice_id}/disable",
    response_model=VoiceResponse,
    summary="Disable a voice",
)
def disable_voice(
    voice_id: str,
    store: Annotated[VoiceStore, Depends(get_store)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> VoiceResponse:
    """Stop a voice being usable for synthesis, keeping its record."""
    store.get_voice(voice_id, owner_id=caller.owner_id)
    voice = store.set_voice_status(voice_id, VoiceStatus.DISABLED)
    return _to_response(voice, consent_verified=voice.consent_id is not None)


@router.delete(
    "/v1/voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice",
)
def delete_voice(
    voice_id: str,
    store: Annotated[VoiceStore, Depends(get_store)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> None:
    """Delete the voice, its consent record and its audio from disk."""
    store.delete_voice(voice_id, owner_id=caller.owner_id)


@router.post(
    "/v1/consents/{consent_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke consent",
)
def revoke_consent(
    consent_id: str,
    store: Annotated[VoiceStore, Depends(get_store)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> None:
    """Revoke a consent record and disable every voice that relied on it."""
    store.revoke_consent(consent_id)


@router.post(
    "/v1/watermark/detect",
    response_model=WatermarkResponse,
    summary="Detect a watermark",
)
async def detect_watermark(
    request: Request,
    caller: Annotated[Caller, Depends(rate_limit)],
    audio: Annotated[UploadFile, File()],
) -> WatermarkResponse:
    """Report whether this service generated the uploaded audio.

    This is the endpoint an abuse report is answered with: a positive detection
    returns the payload, which is a key into the request log.

    Raises:
        ValidationError: Watermarking is disabled on this deployment, so
            nothing can be detected.
    """
    watermarker = request.app.state.watermarker
    if watermarker is None:
        raise ValidationError("watermarking is disabled on this deployment")
    data = await _read_upload(audio, "audio")
    detection = watermarker.detect(load_audio(data))
    return WatermarkResponse(
        detected=detection.detected,
        payload=detection.payload,
        confidence=round(detection.confidence, 4),
        watermarker=watermarker.name,
    )
