"""Synthesis backends."""

from mlvoice.tts.base import (
    ReferencePrompt,
    SynthesisRequest,
    SynthesisResult,
    Synthesizer,
    SynthesizerInfo,
)
from mlvoice.tts.dummy import DummySynthesizer
from mlvoice.tts.registry import available_backends, build_synthesizer, register_backend

__all__ = [
    "DummySynthesizer",
    "ReferencePrompt",
    "SynthesisRequest",
    "SynthesisResult",
    "Synthesizer",
    "SynthesizerInfo",
    "available_backends",
    "build_synthesizer",
    "register_backend",
]
