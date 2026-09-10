"""HTTP service, end to end.

These exercise the assembled application: auth, the synthesis path, the two-step
enrolment flow, consent revocation, error contracts and the watermark
round-trip through HTTP.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlvoice.audio.io import encode_wav, load_audio
from mlvoice.config import Settings
from mlvoice.safety.watermark import SpreadSpectrumWatermarker
from tests.conftest import FakeTranscriber, SpeechFactory

pytestmark = pytest.mark.integration


@pytest.fixture
def reference_bytes(speech: SpeechFactory) -> bytes:
    return encode_wav(speech(words=14, seed=11))


def _enroll(
    client: TestClient,
    auth: dict[str, str],
    transcriber: FakeTranscriber,
    audio: bytes,
    *,
    name: str = "Rajan voice",
    subject: str = "രാജൻ നായർ",
) -> dict[str, object]:
    challenge = client.post("/v1/voices/challenge", json={"subject_name": subject}, headers=auth)
    assert challenge.status_code == 200
    transcriber.text = challenge.json()["phrase"]
    response = client.post(
        "/v1/voices",
        headers=auth,
        data={
            "name": name,
            "reference_text": "ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്",
            "consent_token": challenge.json()["token"],
            "dialect": "kozhikode",
            "gender": "male",
        },
        files={
            "reference_audio": ("ref.wav", audio, "audio/wav"),
            "consent_audio": ("consent.wav", audio, "audio/wav"),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestHealth:
    def test_liveness(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["backend_ready"] is True

    def test_readiness(self, client: TestClient) -> None:
        assert client.get("/readyz").status_code == 200

    def test_health_needs_no_auth(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200

    def test_info_describes_the_deployment(self, client: TestClient) -> None:
        body = client.get("/v1/info").json()
        assert body["backend"]["backend"] == "dummy"
        assert body["consent_required"] is True
        assert body["watermarking"] is True

    def test_openapi_is_generated(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/tts" in paths
        assert "/v1/voices" in paths


class TestAuthentication:
    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്"})
        assert response.status_code == 401
        assert response.json()["code"] == "unauthenticated"

    def test_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്"}, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_error_does_not_reveal_which_keys_exist(self, client: TestClient) -> None:
        missing = client.post("/v1/tts", json={"text": "നാട്"}).json()
        wrong = client.post("/v1/tts", json={"text": "നാട്"}, headers={"X-API-Key": "wrong"}).json()
        assert missing["code"] == wrong["code"]


class TestSynthesis:
    def test_returns_wav_audio(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "ഞാൻ നാട്ടിൽ പോയി."}, headers=auth)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        audio = load_audio(response.content)
        assert audio.sample_rate == 24_000
        assert audio.duration_seconds > 0.1

    def test_reports_generation_metadata(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "ഞാൻ പോയി."}, headers=auth)
        assert float(response.headers["x-audio-duration"]) > 0
        assert float(response.headers["x-real-time-factor"]) > 0
        assert int(response.headers["x-chunks"]) >= 1
        assert response.headers["x-moderation"] == "allow"

    def test_every_response_carries_a_request_id(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്"}, headers=auth)
        assert response.headers["x-request-id"]

    def test_supplied_request_id_is_echoed(self, client: TestClient, auth: dict[str, str]) -> None:
        headers = {**auth, "X-Request-Id": "trace-me-123"}
        response = client.post("/v1/tts", json={"text": "നാട്"}, headers=headers)
        assert response.headers["x-request-id"] == "trace-me-123"

    def test_seed_makes_output_reproducible(self, client: TestClient, auth: dict[str, str]) -> None:
        body = {"text": "ഞാൻ നാട്ടിൽ പോയി.", "seed": 42}
        first = client.post("/v1/tts", json=body, headers=auth)
        second = client.post("/v1/tts", json=body, headers=auth)
        # The watermark payload depends on the request id, so compare durations
        # and lengths rather than bytes.
        assert len(first.content) == len(second.content)

    def test_manglish_input_is_synthesised(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "njan veedu poyi"}, headers=auth)
        assert response.status_code == 200

    def test_empty_text_is_rejected(self, client: TestClient, auth: dict[str, str]) -> None:
        assert client.post("/v1/tts", json={"text": ""}, headers=auth).status_code == 422

    def test_unknown_field_is_rejected(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്", "bogus": True}, headers=auth)
        assert response.status_code == 422

    def test_out_of_range_speed_is_rejected(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്", "speed": 9.0}, headers=auth)
        assert response.status_code == 422

    def test_fraud_template_is_refused(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "അടിയന്തരമായി പണം അയക്കണം"}, headers=auth)
        assert response.status_code == 403
        assert response.json()["code"] == "moderation_blocked"

    def test_unknown_voice_is_404(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts", json={"text": "നാട്", "voice_id": "absent"}, headers=auth)
        assert response.status_code == 404
        assert response.json()["code"] == "voice_not_found"

    def test_error_bodies_have_a_uniform_shape(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        body = client.post(
            "/v1/tts", json={"text": "നാട്", "voice_id": "absent"}, headers=auth
        ).json()
        assert set(body) >= {"code", "message", "request_id"}


class TestStreaming:
    def test_streams_a_wav(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/v1/tts/stream", json={"text": "ഒന്ന്. രണ്ട്. മൂന്ന്."}, headers=auth)
        assert response.status_code == 200
        assert response.content.startswith(b"RIFF")
        assert len(response.content) > 44

    def test_declares_that_it_is_not_watermarked(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        response = client.post("/v1/tts/stream", json={"text": "നാട്"}, headers=auth)
        assert response.headers["x-watermarked"] == "false"


class TestAnalyze:
    def test_reports_every_stage(self, client: TestClient, auth: dict[str, str]) -> None:
        body = client.post(
            "/v1/text/analyze", json={"text": "എൻറെ വീട്ടിൽ ₹250 ഉണ്ട്"}, headers=auth
        ).json()
        assert body["normalized"].startswith("എന്റെ")
        assert "ഇരുനൂറ്റിയമ്പത് രൂപ" in body["routed"]
        assert body["legacy_nta_fixed"] == 1
        assert body["contains_malayalam"] is True

    def test_chunks_carry_phonemes_and_pauses(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        body = client.post("/v1/text/analyze", json={"text": "ഒന്ന്. രണ്ട്."}, headers=auth).json()
        for chunk in body["chunks"]:
            assert chunk["phonemes"]
            assert chunk["pause_ms"] >= 0
            assert chunk["break_after"]


class TestVoiceLifecycle:
    def test_enrolment_then_synthesis(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        assert voice["status"] == "active"
        assert voice["consent_verified"] is True

        response = client.post(
            "/v1/tts", json={"text": "ഞാൻ പോയി.", "voice_id": voice["id"]}, headers=auth
        )
        assert response.status_code == 200

    def test_reference_audio_path_is_not_exposed(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        assert "reference_audio_path" not in voice

    def test_listing_and_reading(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        listing = client.get("/v1/voices", headers=auth).json()
        assert listing["count"] == 1
        assert client.get(f"/v1/voices/{voice['id']}", headers=auth).json()["id"] == voice["id"]

    def test_voices_are_scoped_to_the_api_key(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        _enroll(client, auth, transcriber, reference_bytes)
        # The app is configured with one key, so a second key is simply invalid;
        # the isolation itself is covered at the store level.
        assert client.get("/v1/voices", headers={"X-API-Key": "other"}).status_code == 401

    def test_disable_blocks_synthesis(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        assert client.post(f"/v1/voices/{voice['id']}/disable", headers=auth).status_code == 200
        response = client.post(
            "/v1/tts", json={"text": "നാട്", "voice_id": voice["id"]}, headers=auth
        )
        assert response.status_code == 404

    def test_revoking_consent_blocks_synthesis(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        revoke = client.post(f"/v1/consents/{voice['consent_id']}/revoke", headers=auth)
        assert revoke.status_code == 204
        response = client.post(
            "/v1/tts", json={"text": "നാട്", "voice_id": voice["id"]}, headers=auth
        )
        assert response.status_code in {403, 404}

    def test_deletion(
        self,
        client: TestClient,
        auth: dict[str, str],
        transcriber: FakeTranscriber,
        reference_bytes: bytes,
    ) -> None:
        voice = _enroll(client, auth, transcriber, reference_bytes)
        assert client.delete(f"/v1/voices/{voice['id']}", headers=auth).status_code == 204
        assert client.get(f"/v1/voices/{voice['id']}", headers=auth).status_code == 404

    def test_enrolment_requires_a_valid_token(
        self, client: TestClient, auth: dict[str, str], reference_bytes: bytes
    ) -> None:
        response = client.post(
            "/v1/voices",
            headers=auth,
            data={
                "name": "X",
                "reference_text": "ടെസ്റ്റ്",
                "consent_token": "forged.token",
            },
            files={
                "reference_audio": ("r.wav", reference_bytes, "audio/wav"),
                "consent_audio": ("c.wav", reference_bytes, "audio/wav"),
            },
        )
        assert response.status_code == 403
        assert response.json()["code"] == "consent_required"

    def test_enrolment_rejects_an_empty_upload(
        self, client: TestClient, auth: dict[str, str], transcriber: FakeTranscriber
    ) -> None:
        challenge = client.post(
            "/v1/voices/challenge", json={"subject_name": "രാജൻ"}, headers=auth
        ).json()
        transcriber.text = challenge["phrase"]
        response = client.post(
            "/v1/voices",
            headers=auth,
            data={
                "name": "X",
                "reference_text": "ടെസ്റ്റ്",
                "consent_token": challenge["token"],
            },
            files={
                "reference_audio": ("r.wav", b"", "audio/wav"),
                "consent_audio": ("c.wav", b"", "audio/wav"),
            },
        )
        assert response.status_code == 422


class TestWatermarkEndpoint:
    def test_generated_audio_is_detected(self, client: TestClient, auth: dict[str, str]) -> None:
        audio = client.post(
            "/v1/tts",
            json={"text": "ഞാൻ എന്റെ നാട്ടിലേക്ക് തിരികെ പോയി, മഴ പെയ്യുന്നുണ്ടായിരുന്നു."},
            headers=auth,
        )
        expected = int(audio.headers["x-watermark-payload"])
        detection = client.post(
            "/v1/watermark/detect",
            files={"audio": ("a.wav", audio.content, "audio/wav")},
            headers=auth,
        ).json()
        assert detection["detected"] is True
        assert detection["payload"] == expected

    def test_unmarked_audio_is_not_detected(
        self, client: TestClient, auth: dict[str, str], reference_bytes: bytes
    ) -> None:
        detection = client.post(
            "/v1/watermark/detect",
            files={"audio": ("a.wav", reference_bytes, "audio/wav")},
            headers=auth,
        ).json()
        assert detection["detected"] is False


class TestWatermarkDisabled:
    def test_detection_reports_that_it_is_off(
        self, settings: Settings, verifier, reference_bytes: bytes
    ) -> None:
        from mlvoice.api.app import Overrides, create_app

        disabled = settings.model_copy(update={"watermark_enabled": False})
        app = create_app(Overrides(settings=disabled, consent_verifier=verifier))
        with TestClient(app) as client:
            response = client.post(
                "/v1/watermark/detect",
                files={"audio": ("a.wav", reference_bytes, "audio/wav")},
                headers={"X-API-Key": "test-key"},
            )
            assert response.status_code == 422
            assert "watermarking is disabled" in response.json()["message"]

    def test_synthesis_still_works_without_a_watermark(self, settings: Settings, verifier) -> None:
        from mlvoice.api.app import Overrides, create_app

        disabled = settings.model_copy(update={"watermark_enabled": False})
        app = create_app(Overrides(settings=disabled, consent_verifier=verifier))
        with TestClient(app) as client:
            response = client.post(
                "/v1/tts", json={"text": "നാട്"}, headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200
            assert "x-watermark-payload" not in response.headers


class TestFrontendSettingsAreWired:
    def _analyze(self, settings: Settings, verifier, text: str) -> dict[str, object]:
        from mlvoice.api.app import Overrides, create_app

        app = create_app(Overrides(settings=settings, consent_verifier=verifier))
        with TestClient(app) as client:
            return client.post(
                "/v1/text/analyze", json={"text": text}, headers={"X-API-Key": "test-key"}
            ).json()

    def test_transliteration_can_be_disabled(self, settings: Settings, verifier) -> None:
        on = self._analyze(settings, verifier, "njan paranju")
        off = self._analyze(
            settings.model_copy(update={"transliterate_latin": False}),
            verifier,
            "njan paranju",
        )
        assert on["routed"] != "njan paranju"
        assert off["routed"] == "njan paranju"

    def test_intervocalic_voicing_is_off_by_default(self, settings: Settings, verifier) -> None:
        assert settings.apply_intervocalic_voicing is False
        default = self._analyze(settings, verifier, "അതു")
        voiced = self._analyze(
            settings.model_copy(update={"apply_intervocalic_voicing": True}),
            verifier,
            "അതു",
        )
        assert default["chunks"][0]["phonemes"] != voiced["chunks"][0]["phonemes"]


class TestRateLimiting:
    def test_over_budget_requests_are_refused(self, settings: Settings, verifier) -> None:
        from mlvoice.api.app import Overrides, create_app
        from mlvoice.api.ratelimit import InMemoryRateLimiter

        app = create_app(
            Overrides(
                settings=settings,
                consent_verifier=verifier,
                rate_limiter=InMemoryRateLimiter(
                    requests_per_minute=2, characters_per_minute=1_000_000
                ),
            )
        )
        with TestClient(app) as client:
            headers = {"X-API-Key": "test-key"}
            statuses = [
                client.post("/v1/tts", json={"text": "നാട്"}, headers=headers).status_code
                for _ in range(8)
            ]
            assert 429 in statuses
            assert statuses[0] == 200

    def test_rate_limit_error_body(self, settings: Settings, verifier) -> None:
        from mlvoice.api.app import Overrides, create_app
        from mlvoice.api.ratelimit import InMemoryRateLimiter

        app = create_app(
            Overrides(
                settings=settings,
                consent_verifier=verifier,
                rate_limiter=InMemoryRateLimiter(requests_per_minute=1),
            )
        )
        with TestClient(app) as client:
            headers = {"X-API-Key": "test-key"}
            for _ in range(5):
                response = client.post("/v1/tts", json={"text": "നാട്"}, headers=headers)
            assert response.status_code == 429
            assert response.json()["code"] == "rate_limited"


class TestWatermarkKeyIsolation:
    def test_a_foreign_key_cannot_read_the_payload(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        audio = client.post(
            "/v1/tts",
            json={"text": "ഞാൻ എന്റെ നാട്ടിലേക്ക് തിരികെ പോയി, മഴ പെയ്യുന്നുണ്ടായിരുന്നു."},
            headers=auth,
        )
        foreign = SpreadSpectrumWatermarker(b"a-different-deployment-key")
        assert not foreign.detect(load_audio(audio.content)).detected
