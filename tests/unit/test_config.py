"""Configuration validation.

The production invariants are the point of this module: a deployment that
quietly disabled consent or watermarking would look identical from the outside,
so the process refuses to start instead.
"""

from __future__ import annotations

import pytest

from mlvoice.config import Environment, Settings, get_settings, reset_settings_cache
from mlvoice.errors import ConfigurationError


def _production(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "env": "production",
        "api_keys": "live-key",
        "tts_backend": "indicf5",
        "model_revision": "abc123",
        "consent_signing_key": "consent-key",
        "watermark_key": "watermark-key",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class TestParsing:
    def test_api_keys_split_on_commas_and_whitespace(self) -> None:
        settings = Settings(_env_file=None, api_keys="a, b\tc")
        assert settings.parsed_api_keys == frozenset({"a", "b", "c"})

    def test_empty_api_keys_disable_auth(self) -> None:
        assert Settings(_env_file=None, api_keys="  ").parsed_api_keys == frozenset()

    def test_defaults_are_development(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.env is Environment.DEVELOPMENT
        assert settings.is_production is False

    def test_sample_rate_bounds(self) -> None:
        with pytest.raises(Exception, match="sample_rate"):
            Settings(_env_file=None, sample_rate=1)

    def test_secrets_are_not_in_the_repr(self) -> None:
        settings = Settings(_env_file=None, api_keys="topsecret")
        assert "topsecret" not in repr(settings)


class TestBlankEnvironmentVariables:
    """A blank variable in a .env file arrives as "", not as absent.

    Regression: MLVOICE_BLOCKED_VOICE_NAMES_FILE= became Path("") -- which is
    Path("."), truthy, and exists -- so startup tried to read the working
    directory as a blocklist and raised IsADirectoryError. And
    MLVOICE_MODEL_REVISION= became "", which is not None and so satisfied the
    production "weights must be pinned" gate.
    """

    def test_blank_path_is_unset(self) -> None:
        assert (
            Settings(_env_file=None, blocked_voice_names_file="").blocked_voice_names_file is None
        )

    def test_whitespace_path_is_unset(self) -> None:
        assert (
            Settings(_env_file=None, blocked_voice_names_file="   ").blocked_voice_names_file
            is None
        )

    def test_blank_revision_is_unset(self) -> None:
        assert Settings(_env_file=None, model_revision="").model_revision is None

    def test_a_real_path_is_kept(self, tmp_path: object) -> None:
        settings = Settings(_env_file=None, blocked_voice_names_file="/tmp/blocked.txt")
        assert settings.blocked_voice_names_file is not None
        assert settings.blocked_voice_names_file.name == "blocked.txt"

    def test_a_real_revision_is_kept(self) -> None:
        assert Settings(_env_file=None, model_revision="abc123").model_revision == "abc123"

    def test_production_refuses_a_blank_revision(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            _production(model_revision="")
        assert "MODEL_REVISION" in " ".join(excinfo.value.context["problems"])


class TestProductionInvariants:
    def test_valid_production_config_is_accepted(self) -> None:
        assert _production().is_production

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ({"api_keys": ""}, "MLVOICE_API_KEYS"),
            ({"require_consent": False}, "REQUIRE_CONSENT"),
            ({"watermark_enabled": False}, "WATERMARK_ENABLED"),
            ({"consent_signing_key": ""}, "CONSENT_SIGNING_KEY"),
            ({"watermark_key": ""}, "WATERMARK_KEY"),
            ({"tts_backend": "dummy"}, "dummy"),
            ({"model_revision": None}, "MODEL_REVISION"),
        ],
    )
    def test_unsafe_production_config_is_refused(
        self, override: dict[str, object], expected: str
    ) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            _production(**override)
        problems = " ".join(excinfo.value.context["problems"])
        assert expected in problems

    def test_all_problems_are_reported_together(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            _production(api_keys="", watermark_enabled=False, tts_backend="dummy")
        assert len(excinfo.value.context["problems"]) >= 3

    def test_development_tolerates_the_same_settings(self) -> None:
        settings = Settings(_env_file=None, env="development", api_keys="", tts_backend="dummy")
        assert settings.parsed_api_keys == frozenset()


class TestCaching:
    def test_settings_are_cached(self) -> None:
        reset_settings_cache()
        assert get_settings() is get_settings()

    def test_cache_can_be_cleared(self) -> None:
        reset_settings_cache()
        first = get_settings()
        reset_settings_cache()
        assert get_settings() is not first
