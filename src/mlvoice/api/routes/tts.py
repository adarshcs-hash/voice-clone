"""Synthesis endpoints.

``POST /v1/tts`` returns a complete, watermarked WAV. ``POST /v1/tts/stream``
returns the same audio as a chunked WAV stream for interactive playback, at the
cost of the watermark (see :meth:`mlvoice.synthesis.SynthesisService.stream`).

``POST /v1/text/analyze`` returns the text frontend's output at every stage. It
is the endpoint that makes a pronunciation complaint diagnosable: paste the
text, see exactly what the model was asked to say.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from mlvoice.api.deps import Caller, get_pipeline, rate_limit
from mlvoice.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChunkInfo,
    SynthesizeRequest,
)
from mlvoice.audio.io import encode_wav, to_pcm16, wav_header
from mlvoice.synthesis import SynthesisService
from mlvoice.text.g2p import Notation
from mlvoice.text.pipeline import TextPipeline

router = APIRouter(tags=["synthesis"])


def _service(request: Request) -> SynthesisService:
    service: SynthesisService = request.app.state.synthesis
    return service


@router.post(
    "/v1/tts",
    summary="Synthesise Malayalam speech",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "16-bit PCM WAV."},
        403: {"description": "Refused by the safety policy, or consent revoked."},
        404: {"description": "Unknown voice."},
        413: {"description": "Text exceeds the per-request budget."},
        429: {"description": "Rate limited."},
    },
)
def synthesize(
    body: SynthesizeRequest,
    request: Request,
    caller: Annotated[Caller, Depends(rate_limit)],
) -> Response:
    """Generate speech for ``body.text``.

    Returns the whole utterance as a WAV file, with generation metadata in the
    response headers (``X-Audio-Duration``, ``X-Real-Time-Factor``,
    ``X-Watermark-Payload``).
    """
    outcome = _service(request).synthesize(
        body.text,
        owner_id=caller.owner_id,
        voice_id=body.voice_id,
        speed=body.speed,
        seed=body.seed,
        request_id=getattr(request.state, "request_id", "unknown"),
    )
    return Response(
        content=encode_wav(outcome.audio),
        media_type="audio/wav",
        headers=outcome.headers(),
    )


@router.post(
    "/v1/tts/stream",
    summary="Stream Malayalam speech",
    response_class=StreamingResponse,
    responses={200: {"content": {"audio/wav": {}}, "description": "Chunked WAV stream."}},
)
def synthesize_stream(
    body: SynthesizeRequest,
    request: Request,
    caller: Annotated[Caller, Depends(rate_limit)],
) -> StreamingResponse:
    """Stream speech chunk by chunk for a low time-to-first-byte.

    The response is a WAV whose header declares an unbounded data length, which
    players and ``ffmpeg`` accept for a live stream. Audio in this path is not
    watermarked; use ``POST /v1/tts`` when a provenance-marked artefact is
    needed.
    """
    service = _service(request)
    settings = request.app.state.settings

    def frames() -> Iterator[bytes]:
        yield wav_header(settings.sample_rate)
        for chunk in service.stream(
            body.text,
            owner_id=caller.owner_id,
            voice_id=body.voice_id,
            speed=body.speed,
            seed=body.seed,
        ):
            yield to_pcm16(chunk)

    return StreamingResponse(
        frames(),
        media_type="audio/wav",
        headers={"X-Watermarked": "false", "Cache-Control": "no-store"},
    )


@router.post(
    "/v1/text/analyze",
    response_model=AnalyzeResponse,
    summary="Inspect the Malayalam text frontend",
)
def analyze(
    body: AnalyzeRequest,
    pipeline: Annotated[TextPipeline, Depends(get_pipeline)],
    caller: Annotated[Caller, Depends(rate_limit)],
) -> AnalyzeResponse:
    """Return every stage of text processing for ``body.text``.

    Use it to answer "why did the voice say that?": the response shows the
    Unicode normalisation applied, the expanded numbers and dates, the code-mix
    routing, the chunk boundaries and the phoneme string per chunk.
    """
    processed = pipeline.process(body.text)
    return AnalyzeResponse(
        original=processed.original,
        normalized=processed.normalized,
        expanded=processed.expanded,
        routed=processed.routed,
        contains_malayalam=processed.contains_malayalam,
        normalization_changes=processed.normalization.total_changes,
        legacy_nta_fixed=processed.normalization.legacy_nta_fixed,
        chunks=[
            ChunkInfo(
                index=chunk.index,
                text=chunk.text,
                break_after=chunk.break_after.value,
                pause_ms=chunk.pause_ms,
                phonemes=processed.phoneme_string(chunk.index, notation=Notation.ASCII),
            )
            for chunk in processed.chunks
        ],
    )
