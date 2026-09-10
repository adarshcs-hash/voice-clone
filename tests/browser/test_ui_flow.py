"""The web client, driven by a real browser.

Everything else about the UI is checked statically -- that the script's element
ids exist, that the fields it posts are the ones the endpoints accept. None of
that runs the code. A typo in a handler, a missing ``await``, or an API shape
the script mis-reads all fail the same way in a browser: nothing happens when
you click, with the reason sitting in a console nobody is watching.

So this walks the whole flow against a live service on the dummy backend:
upload a clip, convert it in the browser, take a consent challenge, record it,
enrol, and synthesise. It asserts on what the page *shows* and fails on any
uncaught script error.

Skipped unless Playwright and a Chromium build are both present, since neither
is a dependency of the service itself. Install with the ``browser`` extra, then
``playwright install chromium``.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from mlvoice.audio.io import Audio, encode_wav

pytestmark = [pytest.mark.browser, pytest.mark.integration]

sync_api = pytest.importorskip("playwright.sync_api", reason="needs the browser extra")

_CHROMIUM_ARGS: Final = [
    # A MediaRecorder needs a microphone. These give it a synthetic one and
    # auto-accept the permission prompt, so the consent step is exercised
    # rather than skipped.
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
]
_REFERENCE_SECONDS: Final = 6.0


_EXECUTABLE_ENV: Final = "MLVOICE_BROWSER_EXECUTABLE"


def _executable(playwright: Any) -> str | None:
    """Where to find Chromium, or ``None`` if there is none to find.

    ``MLVOICE_BROWSER_EXECUTABLE`` wins, because a CI image or a sandbox often
    ships a browser at a path Playwright's own revision pinning will not look
    at, and re-downloading 170 MB to satisfy a version number is waste.
    """
    override = os.environ.get(_EXECUTABLE_ENV)
    if override:
        if not Path(override).exists():
            pytest.fail(f"{_EXECUTABLE_ENV} points at {override}, which does not exist")
        return override
    with contextlib.suppress(Exception):
        bundled = Path(playwright.chromium.executable_path)
        if bundled.exists():
            return None  # let Playwright resolve its own
    return "missing"


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    """A Chromium instance, or a skip if no browser is installed.

    Started and stopped explicitly rather than through the context manager, so
    the driver is shut down on the skip path too. Playwright still writes a
    ``TargetClosedError`` traceback to stderr as its driver goes away; it is
    cosmetic, arrives after the run, and does not affect the exit status.
    """
    playwright = sync_api.sync_playwright().start()
    executable = _executable(playwright)
    if executable == "missing":
        playwright.stop()
        pytest.skip(
            "no Chromium build: run `playwright install chromium`, "
            f"or set {_EXECUTABLE_ENV} to an existing one"
        )
    instance = playwright.chromium.launch(args=_CHROMIUM_ARGS, executable_path=executable)
    try:
        yield instance
    finally:
        instance.close()
        playwright.stop()


@pytest.fixture(scope="module")
def reference_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A clip that passes the enrolment quality gate.

    Voiced bursts separated by silence, not a continuous tone: the SNR
    estimator needs a noise floor to measure against, and a tone that never
    stops has none.
    """
    rate = 24_000
    rng = np.random.default_rng(3)
    pieces: list[np.ndarray[Any, Any]] = []
    while sum(piece.size for piece in pieces) < _REFERENCE_SECONDS * rate:
        length = int(rate * 0.35)
        t = np.arange(length) / rate
        f0 = 120.0 + 20.0 * rng.random()
        harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        pieces.append((harmonics * np.hanning(length) * 0.5).astype(np.float32))
        pieces.append(np.zeros(int(rate * 0.08), dtype=np.float32))
    samples = np.concatenate(pieces)
    path = tmp_path_factory.mktemp("browser") / "reference.wav"
    path.write_bytes(encode_wav(Audio(samples=samples, sample_rate=rate)))
    return path


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The real application on a real socket, with consent verification stubbed.

    A TestClient cannot serve a browser, so this runs uvicorn on a thread. The
    consent verifier is the one component that cannot work here: proving the
    subject read the challenge needs an ASR model, and the fake microphone
    produces a beep. Everything else -- storage, quality gates, watermarking,
    the text frontend -- is the production path.
    """
    import uvicorn

    from mlvoice.api.app import Overrides, create_app
    from mlvoice.config import Settings
    from mlvoice.voices.consent import ConsentVerifier

    class EchoTranscriber:
        """Returns whatever it was last told the phrase is."""

        name = "echo-transcriber"

        def __init__(self) -> None:
            self.text = ""

        def transcribe(self, audio: Audio, language: str | None = None) -> str:
            return self.text

    class SameSpeaker:
        name = "same-speaker"

        def similarity(self, first: Audio, second: Audio) -> float:
            return 0.95

    transcriber = EchoTranscriber()

    class EchoVerifier(ConsentVerifier):
        """Accepts the recording as a reading of the issued phrase."""

        def verify(self, phrase: Any, consent_audio: Any, enrolment_audio: Any, **kw: Any) -> Any:
            transcriber.text = phrase.text
            return super().verify(phrase, consent_audio, enrolment_audio, **kw)

    root = tmp_path_factory.mktemp("service")
    settings = Settings(
        _env_file=None,
        env="development",
        api_keys="test-key",
        tts_backend="dummy",
        asr_enabled=False,
        database_url=f"sqlite:///{root}/mlvoice.db",
        voice_storage_dir=root / "voices",
        consent_signing_key="browser-test-consent-key",
        watermark_key="browser-test-watermark-key",
        log_level="WARNING",
    )
    app = create_app(
        Overrides(
            settings=settings,
            consent_verifier=EchoVerifier(transcriber=transcriber, speaker_verifier=SameSpeaker()),
        )
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - only on a broken environment
        pytest.fail("the test service did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def page(browser: Any, service: str) -> Iterator[Any]:
    """A page on the running service, failing the test on any script error."""
    context = browser.new_context(permissions=["microphone"])
    errors: list[str] = []
    active = context.new_page()
    active.on("pageerror", lambda error: errors.append(str(error)))
    active.goto(f"{service}/ui", wait_until="networkidle")
    active.fill("#apiKey", "test-key")
    active.dispatch_event("#apiKey", "change")
    try:
        yield active
    finally:
        context.close()
    assert not errors, f"uncaught script errors: {errors}"


def _settled(page: Any, element: str, *, not_starting_with: str, timeout: int = 30_000) -> str:
    """Wait for a status element to stop showing its in-progress message."""
    page.wait_for_function(
        "([id, prefix]) => { const n = document.getElementById(id);"
        " return n.textContent && !n.textContent.startsWith(prefix); }",
        arg=[element, not_starting_with],
        timeout=timeout,
    )
    return str(page.inner_text(f"#{element}"))


class TestPageLoads:
    def test_it_reports_the_backend_it_is_talking_to(self, page: Any) -> None:
        assert "dummy" in page.inner_text("#backendInfo")

    def test_it_warns_that_the_dummy_backend_is_not_speech(self, page: Any) -> None:
        """Without this, a first-time user hears a buzz and concludes the whole
        thing is broken."""
        assert "dummy backend" in page.inner_text("#backendWarning")


class TestSynthesisWithoutEnrolment:
    def test_the_model_voice_is_usable_before_anything_is_enrolled(self, page: Any) -> None:
        page.fill("#speakText", "ഹലോ, ഇത് ഒരു പരീക്ഷണം ആണ്.")
        page.click("#speakButton")
        status = _settled(page, "speakStatus", not_starting_with="Generating", timeout=60_000)
        assert "of audio in" in status
        assert not page.locator("#outputPlayer").is_hidden()
        assert not page.locator("#downloadLink").is_hidden()

    def test_the_preview_expands_numbers_before_anything_is_generated(self, page: Any) -> None:
        page.fill("#speakText", "2025 ജനുവരി 5.")
        page.click("#analyseButton")
        page.wait_for_selector("#analysisBody tr")
        assert "രണ്ടായിരത്തിയിരുപത്തിയഞ്ച്" in page.inner_text("#analysisSummary")

    def test_markup_in_the_input_is_shown_as_text(self, page: Any) -> None:
        """The preview echoes the caller's own text back into the page. Written
        through innerHTML it would execute; it must render as characters."""
        page.fill("#speakText", "<img src=x onerror=window.__xss=1> ഹലോ.")
        page.click("#analyseButton")
        page.wait_for_selector("#analysisBody tr")
        assert "<img" in page.inner_text("#analysisSummary")
        assert page.evaluate("() => window.__xss === undefined")
        assert page.locator("#analysisSummary img").count() == 0


class TestEnrolment:
    def test_upload_transcribe_consent_enrol_and_speak(
        self, page: Any, reference_wav: Path
    ) -> None:
        page.set_input_files("#referenceFile", str(reference_wav))
        page.wait_for_selector("#referencePlayer:not([hidden])")

        # Transcription is off on this service, so the fallback must appear and
        # must leave the field editable rather than dead-ending the flow.
        status = _settled(page, "referenceStatus", not_starting_with="Converting")
        assert "transcri" in status.lower()
        page.fill("#referenceText", "ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്")

        page.fill("#subjectName", "രാജൻ നായർ")
        page.click("#challengeButton")
        page.wait_for_selector("#challengeBlock:not([hidden])")
        phrase = page.inner_text("#challengePhrase")
        assert "രാജൻ നായർ" in phrase

        page.click("#consentRecord")
        page.wait_for_timeout(1_500)
        page.click("#consentRecord")
        page.wait_for_selector("#consentPlayer:not([hidden])", timeout=15_000)

        page.fill("#voiceName", "Rajan voice")
        page.click("#enrolButton")
        enrolled = _settled(page, "enrolStatus", not_starting_with="Enrolling")
        assert enrolled.startswith("Enrolled as voice_"), enrolled
        assert "consent verified" in enrolled

        # The status settles before the voice list is refetched, so wait for
        # the option rather than reading the select the instant it appears.
        page.wait_for_function("() => document.querySelectorAll('#voiceSelect option').length > 1")
        options = page.locator("#voiceSelect option").all_inner_texts()
        assert any("Rajan voice" in option for option in options)
        assert options[0].startswith("—"), "the model's own voice must stay selectable"

        page.select_option("#voiceSelect", index=1)
        page.fill("#speakText", "നന്ദി.")
        page.click("#speakButton")
        spoken = _settled(page, "speakStatus", not_starting_with="Generating", timeout=60_000)
        assert "of audio in" in spoken, spoken

    def test_enrolling_without_a_reference_says_so(self, page: Any) -> None:
        page.click("#enrolButton")
        assert "reference" in page.inner_text("#enrolStatus")

    def test_a_challenge_needs_a_name(self, page: Any) -> None:
        page.click("#challengeButton")
        assert "name" in page.inner_text("#consentStatus")
