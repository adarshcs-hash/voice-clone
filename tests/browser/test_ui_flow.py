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
    yield from _serve(app)


@pytest.fixture(scope="module")
def service_without_consent(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The same service with ``MLVOICE_REQUIRE_CONSENT`` off.

    Development-only -- production refuses to start this way -- and the client
    is expected to drop its consent step entirely rather than collect a
    recording nobody checks.
    """
    from mlvoice.api.app import Overrides, create_app
    from mlvoice.config import Settings

    root = tmp_path_factory.mktemp("no-consent")
    settings = Settings(
        _env_file=None,
        env="development",
        api_keys="test-key",
        tts_backend="dummy",
        asr_enabled=False,
        require_consent=False,
        database_url=f"sqlite:///{root}/mlvoice.db",
        voice_storage_dir=root / "voices",
        consent_signing_key="browser-test-consent-key",
        watermark_key="browser-test-watermark-key",
        log_level="WARNING",
    )
    yield from _serve(create_app(Overrides(settings=settings)))


def _serve(app: Any) -> Iterator[str]:
    """Run an application on a real socket for the duration of a fixture.

    A TestClient cannot serve a browser, so this is uvicorn on a thread with
    an ephemeral port, chosen by binding one and releasing it.
    """
    import uvicorn

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


def _open(browser: Any, base_url: str) -> Iterator[Any]:
    """A page on a running service, failing the test on any script error."""
    context = browser.new_context(permissions=["microphone"])
    errors: list[str] = []
    active = context.new_page()
    active.on("pageerror", lambda error: errors.append(str(error)))
    active.goto(f"{base_url}/ui", wait_until="networkidle")
    # Only where the page decided a key is needed. Filling it unconditionally
    # would hang on the deployments this suite exists to check, since the field
    # is not merely empty there -- it is not on the page.
    if active.locator("#authCard").is_visible():
        active.fill("#apiKey", "test-key")
        active.dispatch_event("#apiKey", "change")
    try:
        yield active
    finally:
        context.close()
    assert not errors, f"uncaught script errors: {errors}"


@pytest.fixture
def page(browser: Any, service: str) -> Iterator[Any]:
    """A page on the service that requires consent."""
    yield from _open(browser, service)


@pytest.fixture
def page_without_consent(browser: Any, service_without_consent: str) -> Iterator[Any]:
    """A page on the service that does not."""
    yield from _open(browser, service_without_consent)


def _reveal(page: Any, details_id: str) -> None:
    """Open a collapsed ``<details>`` by clicking its summary.

    The preview and the transcript field are deliberately tucked away, so a
    test that wants them has to do what a user would.
    """
    page.click(f"#{details_id} > summary")


def _settled(page: Any, element: str, *, not_starting_with: str, timeout: int = 30_000) -> str:
    """Wait for a status element to stop showing its in-progress message."""
    page.wait_for_function(
        "([id, prefix]) => { const n = document.getElementById(id);"
        " return n.textContent && !n.textContent.startsWith(prefix); }",
        arg=[element, not_starting_with],
        timeout=timeout,
    )
    return str(page.inner_text(f"#{element}"))


@pytest.fixture(scope="module")
def service_simple(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """No API keys and no consent: the plainest local configuration.

    This is what someone cloning their own voice on their own laptop ends up
    with, and the page is expected to show them exactly two inputs.
    """
    from mlvoice.api.app import Overrides, create_app
    from mlvoice.config import Settings

    root = tmp_path_factory.mktemp("simple")
    settings = Settings(
        _env_file=None,
        env="development",
        api_keys="",
        tts_backend="dummy",
        asr_enabled=False,
        require_consent=False,
        database_url=f"sqlite:///{root}/mlvoice.db",
        voice_storage_dir=root / "voices",
        consent_signing_key="browser-test-consent-key",
        watermark_key="browser-test-watermark-key",
        log_level="WARNING",
    )
    yield from _serve(create_app(Overrides(settings=settings)))


@pytest.fixture
def page_simple(browser: Any, service_simple: str) -> Iterator[Any]:
    """A page on the plainest configuration."""
    yield from _open(browser, service_simple)


_SLOW_SECONDS: Final = 1.5


@pytest.fixture(scope="module")
def service_slow(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A service whose synthesis takes long enough to observe.

    The dummy backend returns in milliseconds, which is useless for testing
    what the page shows *during* a request: the assertions race the response
    and pass or fail by scheduling luck. Real synthesis takes about a minute,
    so this reproduces the only property that matters -- that it is slow.
    """
    from mlvoice.api.app import Overrides, create_app
    from mlvoice.config import Settings
    from mlvoice.tts.base import SynthesisRequest
    from mlvoice.tts.dummy import DummySynthesizer

    class SlowSynthesizer(DummySynthesizer):
        """The dummy backend, paced like the real one."""

        def _synthesize_chunk(self, chunk_text: str, request: SynthesisRequest) -> Any:
            time.sleep(_SLOW_SECONDS)
            return super()._synthesize_chunk(chunk_text, request)

    root = tmp_path_factory.mktemp("slow")
    settings = Settings(
        _env_file=None,
        env="development",
        api_keys="",
        tts_backend="dummy",
        asr_enabled=False,
        require_consent=False,
        database_url=f"sqlite:///{root}/mlvoice.db",
        voice_storage_dir=root / "voices",
        consent_signing_key="browser-test-consent-key",
        watermark_key="browser-test-watermark-key",
        log_level="WARNING",
    )
    app = create_app(Overrides(settings=settings, synthesizer=SlowSynthesizer()))
    yield from _serve(app)


@pytest.fixture
def page_slow(browser: Any, service_slow: str) -> Iterator[Any]:
    """A page whose generations are slow enough to watch."""
    yield from _open(browser, service_slow)


class TestPageLoads:
    def test_it_reports_the_backend_it_is_talking_to(self, page: Any) -> None:
        assert "dummy" in page.inner_text("#backendInfo")

    def test_it_warns_that_the_dummy_backend_is_not_speech(self, page: Any) -> None:
        """Without this, a first-time user hears a buzz and concludes the whole
        thing is broken."""
        assert "dummy backend" in page.inner_text("#backendWarning")


class TestOnlyWhatTheServerNeedsIsShown:
    """The page asks ``/v1/info`` what this deployment requires and hides the
    rest. Every field on screen is one the user has to think about, so a field
    the server ignores is worse than no field at all."""

    def test_the_plainest_configuration_shows_two_inputs(self, page_simple: Any) -> None:
        assert page_simple.locator("#authCard").is_hidden()
        assert page_simple.locator("#consentBlock").is_hidden()
        assert page_simple.locator("#voiceFile").count() == 1
        assert page_simple.locator("#speakText").is_visible()

    def test_the_key_field_appears_when_keys_are_configured(self, page: Any) -> None:
        assert page.locator("#authCard").is_visible()

    def test_the_consent_step_appears_when_consent_is_required(self, page: Any) -> None:
        assert page.locator("#consentBlock").is_visible()

    def test_the_consent_step_is_gone_when_it_is_not(self, page_without_consent: Any) -> None:
        assert page_without_consent.locator("#consentBlock").is_hidden()


class TestSpeaking:
    def test_the_model_voice_works_before_any_voice_is_added(self, page_simple: Any) -> None:
        """Nothing should stand between opening the page and hearing output."""
        page_simple.fill("#speakText", "ഹലോ, ഇത് ഒരു പരീക്ഷണം ആണ്.")
        page_simple.click("#speakButton")
        status = _settled(
            page_simple, "speakStatus", not_starting_with="Generating", timeout=60_000
        )
        assert "of audio in" in status
        assert not page_simple.locator("#outputPlayer").is_hidden()
        assert not page_simple.locator("#downloadLink").is_hidden()

    def test_the_preview_expands_numbers(self, page_simple: Any) -> None:
        page_simple.fill("#speakText", "2025 ജനുവരി 5.")
        _reveal(page_simple, "analysisDetails")
        page_simple.click("#analyseButton")
        page_simple.wait_for_selector("#analysisBody tr")
        assert "രണ്ടായിരത്തിയിരുപത്തിയഞ്ച്" in page_simple.inner_text("#analysisSummary")

    def test_markup_in_the_input_is_shown_as_text(self, page_simple: Any) -> None:
        """The preview echoes the caller's own text back into the page. Written
        through innerHTML it would execute; it must render as characters."""
        page_simple.fill("#speakText", "<img src=x onerror=window.__xss=1> ഹലോ.")
        _reveal(page_simple, "analysisDetails")
        page_simple.click("#analyseButton")
        page_simple.wait_for_selector("#analysisBody tr")
        assert "<img" in page_simple.inner_text("#analysisSummary")
        assert page_simple.evaluate("() => window.__xss === undefined")
        assert page_simple.locator("#analysisSummary img").count() == 0


class TestAddingAVoice:
    def test_choosing_a_file_is_the_whole_interaction(
        self, page_simple: Any, reference_wav: Path
    ) -> None:
        """No enrol button, no transcript to type, no steps: pick a clip and
        the voice is ready. This service cannot transcribe, so the page must
        say what is missing and still accept a typed transcript."""
        page = page_simple
        page.set_input_files("#voiceFile", str(reference_wav))
        page.wait_for_selector("#voicePlayer:not([hidden])")
        status = _settled(page, "voiceStatus", not_starting_with="Reading")
        assert "transcribe" in status.lower()
        assert page.locator("#voiceAdvanced").get_attribute("open") is not None, (
            "the panel holding the transcript field must open itself, or the user "
            "is told to type something they cannot see"
        )

        page.fill("#referenceText", "ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്")
        page.click("#enrolButton")
        ready = _settled(page, "voiceStatus", not_starting_with="Adding")
        assert "Voice ready" in ready, ready

        page.wait_for_function("() => document.querySelectorAll('#voiceSelect option').length > 1")
        page.fill("#speakText", "നന്ദി.")
        page.click("#speakButton")
        spoken = _settled(page, "speakStatus", not_starting_with="Generating", timeout=60_000)
        assert "of audio in" in spoken, spoken

    def test_the_consent_flow_still_works_where_it_is_required(
        self, page: Any, reference_wav: Path
    ) -> None:
        page.set_input_files("#voiceFile", str(reference_wav))
        page.wait_for_selector("#voicePlayer:not([hidden])")
        _settled(page, "voiceStatus", not_starting_with="Reading")
        page.fill("#referenceText", "ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്")

        page.fill("#subjectName", "രാജൻ നായർ")
        page.click("#challengeButton")
        page.wait_for_function(
            "() => document.getElementById('challengePhrase').textContent.length > 0"
        )
        assert "രാജൻ നായർ" in page.inner_text("#challengePhrase")

        page.click("#consentRecord")
        page.wait_for_timeout(1_500)
        page.click("#consentRecord")
        page.wait_for_selector("#consentPlayer:not([hidden])", timeout=15_000)

        page.click("#enrolButton")
        ready = _settled(page, "voiceStatus", not_starting_with="Adding")
        assert "Voice ready" in ready, ready

    def test_speaking_without_text_says_so(self, page_simple: Any) -> None:
        page_simple.click("#speakButton")
        assert "Type something" in page_simple.inner_text("#speakStatus")

    def test_a_challenge_needs_a_name(self, page: Any) -> None:
        page.click("#challengeButton")
        assert "name" in page.inner_text("#voiceStatus")


class TestNothingIsShownBeforeItIsUsable:
    """Everything the script reveals must start hidden, in the browser.

    This is a regression test for a CSS bug that no static check would catch:
    the browser's own ``[hidden]`` rule is ``display: none`` at author strength
    zero, so a stylesheet setting ``display: flex`` on a row or
    ``inline-block`` on a link silently beats it. The page rendered with "Use
    this voice" and "Download WAV" already showing, offering actions that could
    not work yet.
    """

    def test_deferred_controls_are_not_visible_on_load(self, page_simple: Any) -> None:
        for element in (
            "enrolRow",
            "voicePlayer",
            "consentPlayer",
            "outputPlayer",
            "downloadLink",
            "analysisBlock",
            "voicesBlock",
        ):
            assert page_simple.locator(f"#{element}").is_hidden(), (
                f"#{element} is visible before anything has revealed it -- check that "
                "no `display` rule outranks the [hidden] attribute"
            )

    def test_the_hidden_attribute_is_enforced_over_display_rules(self, page_simple: Any) -> None:
        """Directly: give a flex row the attribute and it must disappear."""
        hidden = page_simple.evaluate(
            "() => { const row = document.querySelector('.row');"
            " row.hidden = true;"
            " const gone = getComputedStyle(row).display === 'none';"
            " row.hidden = false;"
            " return gone; }"
        )
        assert hidden, "setting .hidden on a .row does not hide it"


class TestTheUserCanTellWhatIsHappening:
    """Feedback during the slow operations.

    Transcription downloads a recogniser on its first run and synthesis takes
    about a minute per sentence on CPU. Both used to show one line of static
    grey text, which is indistinguishable from a frozen page -- so the page
    looked broken exactly when it was working hardest.

    These run against ``page_slow``, whose backend is paced like the real one.
    Racing the dummy backend's millisecond response would make the assertions
    pass or fail by scheduling luck.
    """

    def _busy_state(self, page: Any, status: str, progress: str) -> dict[str, Any]:
        return page.evaluate(
            "([statusId, progressId]) => ({"
            " text: document.getElementById(statusId).textContent,"
            " busy: document.getElementById(statusId).className.includes('busy'),"
            " spinners: document.querySelectorAll(`#${statusId} .spin`).length,"
            " bar: !document.getElementById(progressId).hidden })",
            arg=[status, progress],
        )

    def test_generating_shows_a_spinner_a_bar_and_a_running_clock(self, page_slow: Any) -> None:
        page_slow.fill("#speakText", "ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച്.")
        page_slow.click("#speakButton")

        state = self._busy_state(page_slow, "speakStatus", "speakProgress")
        assert state["busy"], "the status is not marked busy"
        assert state["spinners"] == 1, "no spinner while generating"
        assert state["bar"], "no progress bar while generating"
        assert "Generating" in state["text"]
        assert "minute" in state["text"], "the expected duration is not stated"

        # The counter is what distinguishes working from hung.
        page_slow.wait_for_function(
            "() => /\\d+s/.test(document.querySelector('#speakStatus .elapsed').textContent)",
            timeout=5_000,
        )

        _settled(page_slow, "speakStatus", not_starting_with="Generating", timeout=60_000)
        after = self._busy_state(page_slow, "speakStatus", "speakProgress")
        assert not after["bar"], "the progress bar outlived the request"
        assert not after["busy"]

    def test_the_speak_button_cannot_be_clicked_twice(self, page_slow: Any) -> None:
        """A second request during a slow one is how a queue forms."""
        page_slow.fill("#speakText", "ഒന്ന് രണ്ട്.")
        page_slow.click("#speakButton")
        assert page_slow.locator("#speakButton").is_disabled()
        assert "Generating" in page_slow.inner_text("#speakButton")

        _settled(page_slow, "speakStatus", not_starting_with="Generating", timeout=60_000)
        assert page_slow.locator("#speakButton").is_enabled()
        assert page_slow.inner_text("#speakButton") == "Speak", (
            "the button must return to its original label, not keep the busy one"
        )

    def test_a_loaded_clip_is_confirmed_and_stays_confirmed(
        self, page_simple: Any, reference_wav: Path
    ) -> None:
        """The status line is transient -- it is about to say "Transcribing…" --
        so the confirmation lives in its own element. Written to the status
        line it vanished in the same tick it appeared, which is how "there is
        no indication the upload worked" happens.
        """
        page_simple.set_input_files("#voiceFile", str(reference_wav))
        page_simple.wait_for_selector("#voiceSummary:not([hidden])", timeout=15_000)
        summary = page_simple.inner_text("#voiceSummary")
        assert "reference.wav" in summary
        assert "6.0s" in summary

        # Whatever the next step says, the confirmation is still on screen.
        _settled(page_simple, "voiceStatus", not_starting_with="Reading")
        assert page_simple.locator("#voiceSummary").is_visible()
        assert "reference.wav" in page_simple.inner_text("#voiceSummary")

    def test_no_progress_bar_outlives_its_operation(
        self, page_simple: Any, reference_wav: Path
    ) -> None:
        """Every exit from a busy state has to clear its bar, including the
        early returns. One did not, and the voice card kept animating
        indefinitely under a finished message -- which says "still working"
        about something that has stopped.
        """
        page_simple.set_input_files("#voiceFile", str(reference_wav))
        _settled(page_simple, "voiceStatus", not_starting_with="Reading")
        assert page_simple.locator("#voiceProgress").is_hidden()
        assert page_simple.locator("#speakProgress").is_hidden()

    def test_the_elapsed_clock_stops_when_the_status_changes(self, page_slow: Any) -> None:
        """A leaked interval would keep rewriting a finished status line."""
        page_slow.fill("#speakText", "ഒന്ന്.")
        page_slow.click("#speakButton")
        _settled(page_slow, "speakStatus", not_starting_with="Generating", timeout=60_000)
        settled = page_slow.inner_text("#speakStatus")
        page_slow.wait_for_timeout(2_200)
        assert page_slow.inner_text("#speakStatus") == settled

    def test_a_previous_result_is_cleared_before_the_next_run(self, page_slow: Any) -> None:
        """Leaving the old player visible during a new generation invites
        listening to the previous take and calling it the new one."""
        page_slow.fill("#speakText", "ഒന്ന്.")
        page_slow.click("#speakButton")
        _settled(page_slow, "speakStatus", not_starting_with="Generating", timeout=60_000)
        assert page_slow.locator("#outputPlayer").is_visible()

        page_slow.fill("#speakText", "രണ്ട്.")
        page_slow.click("#speakButton")
        assert page_slow.locator("#outputPlayer").is_hidden()
        assert page_slow.locator("#downloadLink").is_hidden()

    def test_the_text_preview_still_renders_its_line_breaks(self, page_simple: Any) -> None:
        """The summary shares the .status class with the busy indicators, so a
        flex container added for the spinner would lay its <br> breaks out
        sideways."""
        page_simple.fill("#speakText", "2025 ജനുവരി 5.")
        _reveal(page_simple, "analysisDetails")
        page_simple.click("#analyseButton")
        page_simple.wait_for_selector("#analysisBody tr")
        assert (
            page_simple.evaluate(
                "() => getComputedStyle(document.getElementById('analysisSummary')).display"
            )
            == "block"
        )
        assert page_simple.locator("#analysisSummary br").count() > 0
