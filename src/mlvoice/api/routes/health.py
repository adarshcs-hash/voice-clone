"""Health, readiness and capability endpoints.

Liveness and readiness are separate on purpose. ``/healthz`` answers "is this
process alive?" and must never depend on the model, or a slow first load will
have the orchestrator kill a container that was about to become useful.
``/readyz`` answers "should traffic come here?" and does depend on the backend
being loaded.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from mlvoice.__version__ import __version__
from mlvoice.api.deps import get_settings_dep, get_synthesizer
from mlvoice.api.schemas import HealthResponse, InfoResponse
from mlvoice.config import Settings
from mlvoice.tts.base import Synthesizer

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness")
def healthz(
    synthesizer: Annotated[Synthesizer, Depends(get_synthesizer)],
) -> HealthResponse:
    """Report that the process is running. Always 200 while it can serve."""
    return HealthResponse(status="ok", version=__version__, backend_ready=synthesizer.is_ready())


@router.get("/readyz", response_model=HealthResponse, summary="Readiness")
def readyz(
    response: Response,
    synthesizer: Annotated[Synthesizer, Depends(get_synthesizer)],
) -> HealthResponse:
    """Report whether this instance should receive traffic.

    Returns 503 until the synthesis backend has finished loading, so that a
    rolling deploy does not route requests into a cold replica.
    """
    ready = synthesizer.is_ready()
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ready else "degraded", version=__version__, backend_ready=ready
    )


@router.get("/v1/info", response_model=InfoResponse, summary="Service capabilities")
def info(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    synthesizer: Annotated[Synthesizer, Depends(get_synthesizer)],
) -> InfoResponse:
    """Describe the running configuration, for clients and operators."""
    return InfoResponse(
        version=__version__,
        environment=settings.env.value,
        backend=synthesizer.info.as_dict(),
        watermarking=settings.watermark_enabled,
        consent_required=settings.require_consent,
        max_chars_per_request=settings.max_chars_per_request,
    )
