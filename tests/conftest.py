"""Shared test fixtures.

Synthetic audio rather than recordings: a real voice clip in the repository is
both a licensing question and, for a voice-cloning project, biometric data
nobody consented to distribute.

The generator produces voiced bursts separated by **silence**, which matters.
The SNR estimator in :mod:`mlvoice.audio.quality` derives its noise floor from
non-speech frames, so a continuous tone is not a valid stand-in for speech and
would exercise a different branch than production does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from mlvoice.audio.io import Audio
from mlvoice.config import Settings
from mlvoice.voices.consent import ConsentVerifier
from mlvoice.voices.store import VoiceStore

SAMPLE_RATE = 24_000

SpeechFactory = Callable[..., Audio]


def make_speech(
    *,
    words: int = 8,
    word_seconds: float = 0.32,
    gap_seconds: float = 0.14,
    lead_seconds: float = 0.25,
    sample_rate: int = SAMPLE_RATE,
    f0: float = 190.0,
    noise_level: float = 0.0008,
    seed: int = 0,
) -> Audio:
    """Generate speech-like audio: harmonic bursts separated by silence."""
    rng = np.random.default_rng(seed)
    gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
    lead = np.zeros(int(sample_rate * lead_seconds), dtype=np.float32)

    parts: list[np.ndarray] = [lead]
    for index in range(words):
        count = int(sample_rate * word_seconds)
        t = np.arange(count, dtype=np.float32) / sample_rate
        pitch = f0 * (1.0 + 0.06 * np.sin(index)) - 8.0 * t
        burst = np.zeros(count, dtype=np.float32)
        for harmonic, amplitude in enumerate((0.34, 0.17, 0.09, 0.05, 0.025, 0.012), start=1):
            burst += amplitude * np.sin(
                2 * np.pi * harmonic * pitch * t + rng.uniform(0, 2 * np.pi)
            ).astype(np.float32)
        parts.append((burst * np.hanning(count).astype(np.float32)).astype(np.float32))
        parts.append(gap)
    parts.append(lead)

    signal = np.concatenate(parts)
    signal = signal + rng.normal(0, noise_level, signal.shape).astype(np.float32)
    return Audio(
        samples=(signal / np.max(np.abs(signal)) * 0.5).astype(np.float32), sample_rate=sample_rate
    )


@pytest.fixture
def speech() -> SpeechFactory:
    """Factory for speech-like audio."""
    return make_speech


@pytest.fixture
def clip(speech: SpeechFactory) -> Audio:
    """A five-second speech-like clip."""
    return speech(words=10)


class FakeTranscriber:
    """ASR stub returning a caller-controlled string."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio: Audio) -> str:
        """Return the configured text."""
        self.calls += 1
        return self.text


class FakeSpeakerVerifier:
    """Speaker verifier stub returning a fixed similarity."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    @property
    def name(self) -> str:
        """Identifier."""
        return "fake-speaker-verifier"

    def similarity(self, first: Audio, second: Audio) -> float:
        """Return the configured score."""
        return self.score


@pytest.fixture
def transcriber() -> FakeTranscriber:
    """A controllable ASR stub."""
    return FakeTranscriber()


@pytest.fixture
def speaker_verifier() -> FakeSpeakerVerifier:
    """A controllable speaker-verifier stub."""
    return FakeSpeakerVerifier()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Development settings pointing at a temporary directory."""
    return Settings(
        _env_file=None,
        env="development",
        api_keys="test-key",
        tts_backend="dummy",
        database_url=f"sqlite:///{tmp_path}/mlvoice.db",
        voice_storage_dir=tmp_path / "voices",
        consent_signing_key="test-consent-signing-key",
        watermark_key="test-watermark-key",
        log_level="WARNING",
        log_json=True,
    )


@pytest.fixture
def store(settings: Settings) -> VoiceStore:
    """A voice store on a temporary SQLite database."""
    return VoiceStore(settings.database_url)


@pytest.fixture
def verifier(
    transcriber: FakeTranscriber, speaker_verifier: FakeSpeakerVerifier
) -> ConsentVerifier:
    """A consent verifier wired to the stubs."""
    return ConsentVerifier(transcriber=transcriber, speaker_verifier=speaker_verifier)


@pytest.fixture
def client(settings: Settings, verifier: ConsentVerifier) -> Iterator[object]:
    """A TestClient over the assembled application."""
    from fastapi.testclient import TestClient

    from mlvoice.api.app import Overrides, create_app

    app = create_app(Overrides(settings=settings, consent_verifier=verifier))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    """Headers carrying the test API key."""
    return {"X-API-Key": "test-key"}
