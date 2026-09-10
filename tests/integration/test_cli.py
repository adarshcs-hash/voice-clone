"""Command line interface, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlvoice.audio.io import load_audio, save_audio
from mlvoice.cli import app
from tests.conftest import SpeechFactory

pytestmark = pytest.mark.integration


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CliRunner:
    """A runner with the environment pinned to a development configuration."""
    for key, value in {
        "MLVOICE_ENV": "development",
        "MLVOICE_TTS_BACKEND": "dummy",
        "MLVOICE_LOG_LEVEL": "ERROR",
        "MLVOICE_DATABASE_URL": f"sqlite:///{tmp_path}/db.sqlite",
        "MLVOICE_VOICE_STORAGE_DIR": str(tmp_path / "voices"),
        "MLVOICE_API_KEYS": "cli-key",
        "MLVOICE_CONSENT_SIGNING_KEY": "cli-consent-key",
        "MLVOICE_WATERMARK_KEY": "cli-watermark-key",
    }.items():
        monkeypatch.setenv(key, value)
    from mlvoice.config import reset_settings_cache

    reset_settings_cache()
    return CliRunner()


class TestBasics:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip()

    def test_help_lists_the_command_groups(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for group in ("text", "eval", "data", "watermark", "speak", "serve"):
            assert group in result.stdout

    def test_info_reports_the_configuration(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["backend"] == "dummy"
        assert "dummy" in payload["available_backends"]
        assert payload["test_set"]


class TestTextCommands:
    def test_normalize(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["text", "normalize", "എൻറെ വീട്"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "എന്റെ വീട്"

    def test_normalize_report(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["text", "normalize", "എൻറെ വീട്", "--report"])
        payload = json.loads(result.stdout)
        assert payload["legacy_nta_fixed"] == 1
        assert payload["notes"]

    def test_normalize_from_stdin(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["text", "normalize"], input="എൻറെ വീട്")
        assert result.exit_code == 0
        assert result.stdout.strip() == "എന്റെ വീട്"

    def test_normalize_from_a_file(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "in.txt"
        source.write_text("എൻറെ വീട്", encoding="utf-8")
        result = runner.invoke(app, ["text", "normalize", "--file", str(source)])
        assert result.stdout.strip() == "എന്റെ വീട്"

    def test_normalize_with_no_input_fails(self, runner: CliRunner) -> None:
        assert runner.invoke(app, ["text", "normalize"], input="").exit_code != 0

    def test_analyze(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["text", "analyze", "എൻറെ വീട്ടിൽ ₹250 ഉണ്ട്"])
        payload = json.loads(result.stdout)
        assert "ഇരുനൂറ്റിയമ്പത് രൂപ" in payload["routed"]
        assert payload["chunks"][0]["ipa"]
        assert payload["chunks"][0]["ascii"]

    def test_g2p_ipa_and_ascii(self, runner: CliRunner) -> None:
        ipa = runner.invoke(app, ["text", "g2p", "നാട്"]).stdout.strip()
        ascii_form = runner.invoke(app, ["text", "g2p", "നാട്", "--ascii"]).stdout.strip()
        assert ipa == "n aː ʈ ɨ"
        assert ascii_form == "n aa tt ax"

    def test_number(self, runner: CliRunner) -> None:
        payload = json.loads(runner.invoke(app, ["text", "number", "125000"]).stdout)
        assert payload["cardinal"] == "ഒരു ലക്ഷത്തി ഇരുപത്തിയയ്യായിരം"


class TestSpeak:
    def test_writes_a_wav(self, runner: CliRunner, tmp_path: Path) -> None:
        output = tmp_path / "out.wav"
        result = runner.invoke(app, ["speak", "ഞാൻ നാട്ടിൽ പോയി", "-o", str(output)])
        assert result.exit_code == 0, result.stdout
        assert output.exists()
        assert load_audio(output).duration_seconds > 0.1

    def test_text_can_come_from_a_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Long-form input should not go through shell quoting: a mis-quoted
        700-character Malayalam script is hard to spot."""
        script = tmp_path / "script.txt"
        script.write_text(
            "ഹലോ ഗയ്സ്! ഇത് ഒരു നീണ്ട വാക്യം ആണ്.\n\nഇത് രണ്ടാമത്തെ ഖണ്ഡിക ആണ്.",
            encoding="utf-8",
        )
        output = tmp_path / "out.wav"
        result = runner.invoke(app, ["speak", "--file", str(script), "-o", str(output)])
        assert result.exit_code == 0, result.stdout
        assert output.exists()
        assert json.loads(result.stdout)["chunks"] >= 1

    def test_text_can_come_from_stdin(self, runner: CliRunner, tmp_path: Path) -> None:
        output = tmp_path / "out.wav"
        result = runner.invoke(app, ["speak", "-o", str(output)], input="ഞാൻ നാട്ടിൽ പോയി")
        assert result.exit_code == 0, result.stdout
        assert output.exists()

    def test_reference_transcript_can_come_from_a_file(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        reference = tmp_path / "ref.wav"
        save_audio(speech(words=12), reference)
        transcript = tmp_path / "ref.txt"
        transcript.write_text("ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്", encoding="utf-8")
        output = tmp_path / "out.wav"
        result = runner.invoke(
            app,
            [
                "speak",
                "ഞാൻ പോയി",
                "-o",
                str(output),
                "--reference-audio",
                str(reference),
                "--reference-text-file",
                str(transcript),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert output.exists()

    def test_both_transcript_sources_is_rejected(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        reference = tmp_path / "ref.wav"
        save_audio(speech(words=12), reference)
        transcript = tmp_path / "ref.txt"
        transcript.write_text("x", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "speak",
                "ഞാൻ പോയി",
                "-o",
                str(tmp_path / "o.wav"),
                "--reference-audio",
                str(reference),
                "--reference-text",
                "y",
                "--reference-text-file",
                str(transcript),
            ],
        )
        assert result.exit_code != 0

    def test_reference_flags_must_come_together(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        reference = tmp_path / "ref.wav"
        save_audio(speech(words=10), reference)
        result = runner.invoke(
            app,
            ["speak", "നാട്", "-o", str(tmp_path / "o.wav"), "--reference-audio", str(reference)],
        )
        assert result.exit_code != 0

    def test_cloning_from_a_local_reference(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        reference = tmp_path / "ref.wav"
        save_audio(speech(words=12), reference)
        output = tmp_path / "cloned.wav"
        result = runner.invoke(
            app,
            [
                "speak",
                "ഞാൻ പോയി",
                "-o",
                str(output),
                "--reference-audio",
                str(reference),
                "--reference-text",
                "ഇത് എന്റെ ശബ്ദം",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert output.exists()


class TestEvalCommands:
    def test_frontend_passes_and_writes_a_report(self, runner: CliRunner, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        result = runner.invoke(app, ["eval", "frontend", "-o", str(report)])
        assert result.exit_code == 0, result.stdout
        assert "pass_rate=1.0000" in result.stdout
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["kind"] == "frontend"
        assert payload["failures"] == []

    def test_frontend_threshold_can_fail_the_build(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["eval", "frontend", "--fail-under", "1.1"])
        assert result.exit_code == 1

    def test_synthesis_report(self, runner: CliRunner, tmp_path: Path) -> None:
        report = tmp_path / "synth.json"
        result = runner.invoke(app, ["eval", "synthesis", "-o", str(report)])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["errors"] == 0
        assert payload["cases"] > 0

    def test_testset_counts(self, runner: CliRunner) -> None:
        payload = json.loads(runner.invoke(app, ["eval", "testset"]).stdout)
        assert payload["samvruthokaram"] >= 2

    def test_testset_by_category(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["eval", "testset", "--category", "numbers"])
        assert result.exit_code == 0
        assert "num-001" in result.stdout

    def test_testset_rejects_an_unknown_category(self, runner: CliRunner) -> None:
        assert runner.invoke(app, ["eval", "testset", "--category", "bogus"]).exit_code != 0

    def test_scoring_instructions(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["eval", "testset", "--instructions"])
        assert "PRONUNCIATION" in result.stdout


class TestDataCommands:
    def test_prepare_then_stats(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        audio_dir = tmp_path / "audio"
        for index in range(3):
            save_audio(speech(words=9, seed=index), audio_dir / f"{index}.wav")
        tsv = tmp_path / "corpus.tsv"
        tsv.write_text(
            "".join(f"u{i}\t{i}.wav\tഎന്റെ നാട്ടിൽ മഴ പെയ്തു\ts{i % 2}\n" for i in range(3)),
            encoding="utf-8",
        )
        manifest = tmp_path / "manifest.jsonl"
        result = runner.invoke(
            app,
            [
                "data",
                "prepare",
                str(tsv),
                "-o",
                str(tmp_path / "processed"),
                "-m",
                str(manifest),
                "--source",
                "test-corpus",
                "--license-id",
                "CC-BY-4.0",
                "--commercial",
                "--audio-root",
                str(audio_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["written"] == 3
        assert payload["rejected"] == 0

        stats = json.loads(runner.invoke(app, ["data", "stats", str(manifest)]).stdout)
        assert stats["utterances"] == 3
        assert stats["speakers"] == 2
        assert stats["by_source"] == {"test-corpus": 3}
        assert stats["commercial_use_permitted"] == {"True": 3}


class TestWatermarkCommands:
    def test_embed_then_detect(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        source = tmp_path / "in.wav"
        save_audio(speech(words=14), source)
        marked = tmp_path / "marked.wav"
        embed = runner.invoke(
            app,
            [
                "watermark",
                "embed",
                str(source),
                "-o",
                str(marked),
                "--payload",
                "424242",
                "--key",
                "cli-watermark-key",
            ],
        )
        assert embed.exit_code == 0, embed.stdout

        detect = runner.invoke(
            app, ["watermark", "detect", str(marked), "--key", "cli-watermark-key"]
        )
        payload = json.loads(detect.stdout)
        assert payload["detected"] is True
        assert payload["payload"] == 424242

    def test_detect_needs_a_key(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        source = tmp_path / "in.wav"
        save_audio(speech(words=12), source)
        result = runner.invoke(app, ["watermark", "detect", str(source), "--key", ""])
        assert result.exit_code != 0

    def test_unmarked_audio_is_not_detected(
        self, runner: CliRunner, tmp_path: Path, speech: SpeechFactory
    ) -> None:
        source = tmp_path / "clean.wav"
        save_audio(speech(words=14), source)
        payload = json.loads(
            runner.invoke(
                app, ["watermark", "detect", str(source), "--key", "cli-watermark-key"]
            ).stdout
        )
        assert payload["detected"] is False
