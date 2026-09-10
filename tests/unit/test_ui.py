"""The browser client's contract with the API.

The point of these tests is not that a route returns 200. It is that the page
and the script keep agreeing with each other and with the endpoints they call.
Nothing else catches a renamed element id or a dropped response field -- there
is no type checker across the HTML/JS/HTTP boundary, and the failure mode is a
silently dead button rather than an error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest
from fastapi.testclient import TestClient

from mlvoice.api.app import Overrides, create_app
from mlvoice.config import Environment, Settings

STATIC = Path(__file__).resolve().parents[2] / "src" / "mlvoice" / "api" / "static"


@pytest.fixture(scope="module")
def markup() -> str:
    """The page source."""
    return (STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    """The client script source."""
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', markup))


def _referenced_ids(script: str) -> set[str]:
    """Ids the script looks up through its ``$`` helper."""
    return set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script))


class TestPageAndScriptAgree:
    def test_every_id_the_script_uses_exists_in_the_page(self, markup: str, script: str) -> None:
        missing = _referenced_ids(script) - _element_ids(markup)
        assert not missing, f"app.js binds ids the page does not define: {sorted(missing)}"

    def test_every_status_target_is_a_status_element(self, markup: str, script: str) -> None:
        """``setStatus`` overwrites ``className``, so the target must be a status div.

        Pointing it at an input or a player would wipe that element's classes
        and leave the message invisible.
        """
        targets = set(re.findall(r'setStatus\(\s*"([A-Za-z0-9_-]+)"', script))
        for target in targets:
            pattern = rf'id="{target}"[^>]*class="[^"]*\bstatus\b'
            assert re.search(pattern, markup) or re.search(
                rf'class="[^"]*\bstatus\b[^"]*"[^>]*id="{target}"', markup
            ), f"{target} receives status text but is not a .status element"

    def test_the_page_loads_the_script_from_the_served_path(self, markup: str) -> None:
        assert 'src="/ui/app.js"' in markup

    def test_recorder_buttons_have_no_child_markup(self, markup: str, script: str) -> None:
        """``bindRecorder`` stashes ``textContent`` and restores it as text.

        A button containing elements would lose them the first time it is used.
        """
        for button in ("referenceRecord", "consentRecord"):
            body = re.search(rf'<button id="{button}"[^>]*>(.*?)</button>', markup, re.S)
            assert body is not None
            assert "<" not in body.group(1)


class TestScriptUsesRealApiFields:
    """Response fields the script reads must exist in the schemas."""

    def test_transcribe_fields(self, script: str) -> None:
        from mlvoice.api.schemas import TranscribeResponse

        for field in ("text", "duration_seconds", "advisories", "quality"):
            assert field in TranscribeResponse.model_fields
        assert "estimated_snr_db" in script

    def test_analyze_fields(self, script: str) -> None:
        from mlvoice.api.schemas import AnalyzeResponse, ChunkInfo

        for field in ("chunks", "routed", "original", "legacy_nta_fixed"):
            assert field in AnalyzeResponse.model_fields
        for field in ("index", "text", "phonemes", "break_after", "pause_ms"):
            assert field in ChunkInfo.model_fields

    def test_challenge_and_voice_fields(self, script: str) -> None:
        from mlvoice.api.schemas import ChallengeResponse, VoiceResponse

        for field in ("phrase", "token"):
            assert field in ChallengeResponse.model_fields
        for field in ("id", "reference_duration_seconds", "consent_verified", "name", "status"):
            assert field in VoiceResponse.model_fields

    def test_tts_headers_the_script_reads_are_emitted(self, script: str) -> None:
        headers = set(re.findall(r'headers\.get\("(X-[A-Za-z-]+)"', script))
        assert headers, "the script is expected to read the generation headers"
        emitted = _outcome_header_names()
        unknown = headers - emitted
        assert not unknown, f"script reads headers the service never sets: {sorted(unknown)}"

    def test_enrolment_form_fields_match_the_endpoint(self, script: str) -> None:
        import inspect

        from mlvoice.api.routes.voices import enroll_voice

        sent = set(re.findall(r'form\.append\("([a-z_]+)"', _function_body(script, "enrol")))
        accepted = set(inspect.signature(enroll_voice).parameters)
        assert sent <= accepted, f"unknown form fields: {sorted(sent - accepted)}"
        required = {"name", "consent_token", "reference_audio", "consent_audio"}
        assert required <= sent


def _reads_of(name: str, settings: Settings) -> int:
    """How many times two requests for ``/ui`` hit the filesystem."""
    from mlvoice.api.routes import ui as ui_module

    calls: list[str] = []
    original = ui_module._read

    def counting_read(asset: str) -> str:
        calls.append(asset)
        return original(asset)

    ui_module._read_cached.cache_clear()
    ui_module._read = counting_read  # type: ignore[assignment]
    try:
        with TestClient(create_app(Overrides(settings=settings))) as client:
            client.get("/ui")
            client.get("/ui")
    finally:
        ui_module._read = original  # type: ignore[assignment]
        ui_module._read_cached.cache_clear()
    return calls.count(name)


def _function_body(script: str, name: str) -> str:
    """The source of one top-level ``async function`` in the script.

    Scoping matters: several handlers build a ``FormData``, so matching field
    names across the whole file would compare the transcribe form against the
    enrolment endpoint.
    """
    start = script.index(f"async function {name}(")
    end = script.index("\nasync function ", start + 1)
    return script[start:end]


def _outcome_header_names() -> set[str]:
    """The header names :meth:`SynthesisOutcome.headers` can produce.

    Read out of the source rather than by building an outcome, which would
    need audio and a full synthesis result for no extra confidence.
    """
    import inspect

    from mlvoice.synthesis import SynthesisOutcome

    source = inspect.getsource(SynthesisOutcome.headers)
    return set(re.findall(r'"(X-[A-Za-z-]+)"', source))


class TestRoutes:
    def test_index_is_served_without_a_key(self, client: TestClient) -> None:
        """The page holds no secrets, and asking for a key to see the form
        that collects the key would be circular."""
        response = client.get("/ui")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "/ui/app.js" in response.text

    def test_index_is_not_cached(self, client: TestClient) -> None:
        """A cached page outlives a redeploy and desynchronises from the script."""
        assert client.get("/ui").headers["cache-control"] == "no-store"

    def test_script_is_served_as_javascript(self, client: TestClient) -> None:
        response = client.get("/ui/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]
        assert "decodeAudioData" in response.text

    def test_the_index_advertises_the_ui(self, client: TestClient) -> None:
        assert client.get("/").json()["ui"] == "/ui"

    def test_the_ui_stays_out_of_the_openapi_schema(self, client: TestClient) -> None:
        """It is not an API resource; listing it would put a text/html route in
        the generated clients."""
        assert "/ui" not in client.get("/openapi.json").json()["paths"]

    def test_assets_are_reread_in_development(self, settings: Settings) -> None:
        """An edit must show up on reload, or developing the page is miserable."""
        assert _reads_of("index.html", settings) == 2

    def test_assets_are_read_once_outside_development(self, settings: Settings) -> None:
        """The files cannot change under a running deployment, so holding them
        in memory keeps a page load off the filesystem."""
        staging = settings.model_copy(update={"env": Environment.STAGING})
        assert _reads_of("index.html", staging) == 1

    def test_disabling_the_ui_hides_it(self, settings: Settings) -> None:
        disabled = settings.model_copy(update={"ui_enabled": False})
        with TestClient(create_app(Overrides(settings=disabled))) as client:
            for path in ("/ui", "/ui/app.js"):
                response = client.get(path)
                assert response.status_code == 404
                assert response.json()["code"] == "feature_disabled"


_ASSET_LIMIT_BYTES: Final = 64 * 1024


class TestPackaging:
    def test_assets_ship_inside_the_package(self) -> None:
        """Served through ``importlib.resources``, so they must be package data."""
        from importlib import resources

        for name in ("index.html", "app.js"):
            assert (resources.files("mlvoice.api") / "static" / name).is_file()

    def test_assets_stay_small_enough_to_serve_from_memory(self) -> None:
        """Production reads each asset once and holds it; a build artefact
        landing in this directory would be the first sign of trouble."""
        for name in ("index.html", "app.js"):
            assert (STATIC / name).stat().st_size < _ASSET_LIMIT_BYTES
