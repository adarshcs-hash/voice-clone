"""Corpus manifests and the preparation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from mlvoice.audio.io import Audio, save_audio
from mlvoice.data.manifest import (
    Dialect,
    Gender,
    Split,
    Utterance,
    count_by,
    read_manifest,
    total_duration_hours,
    write_manifest,
)
from mlvoice.data.prepare import (
    CorpusPreparer,
    PrepareConfig,
    RawEntry,
    assign_split,
    iter_entries_from_tsv,
)
from mlvoice.errors import ValidationError
from tests.conftest import FakeTranscriber, SpeechFactory


def _utterance(**overrides: object) -> Utterance:
    payload: dict[str, object] = {
        "id": "u1",
        "audio_path": "a/1.wav",
        "source": "imasc",
        "license_id": "CC-BY-4.0",
        "commercial_use_permitted": True,
        "text": "എന്റെ നാട്",
        "speaker_id": "s1",
        "duration_seconds": 3.5,
        "sample_rate": 24_000,
    }
    payload.update(overrides)
    return Utterance(**payload)  # type: ignore[arg-type]


class TestManifestSchema:
    def test_round_trip(self, tmp_path: Path) -> None:
        utterances = [_utterance(id=f"u{i}") for i in range(3)]
        written = write_manifest(tmp_path / "m.jsonl", utterances)
        assert written == 3
        assert [u.id for u in read_manifest(tmp_path / "m.jsonl")] == ["u0", "u1", "u2"]

    def test_unicode_survives_serialisation(self, tmp_path: Path) -> None:
        write_manifest(tmp_path / "m.jsonl", [_utterance(text="തിരുവനന്തപുരം")])
        assert next(iter(read_manifest(tmp_path / "m.jsonl"))).text == "തിരുവനന്തപുരം"

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        write_manifest(path, [_utterance()])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(list(read_manifest(path))) == 1

    def test_bad_line_names_the_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text('{"id":"ok"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValidationError) as excinfo:
            list(read_manifest(path))
        assert excinfo.value.context["line"] == 1

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(Exception, match="duration_seconds"):
            _utterance(duration_seconds=0)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(Exception, match="extra"):
            _utterance(unexpected="x")

    def test_language_must_be_malayalam(self) -> None:
        with pytest.raises(Exception, match="Malayalam"):
            _utterance(language="hi")

    def test_provenance_is_mandatory(self) -> None:
        """``source``, ``license_id`` and the commercial-use flag have no defaults."""
        with pytest.raises(PydanticValidationError) as excinfo:
            Utterance(  # type: ignore[call-arg]
                id="u",
                audio_path="a",
                text="t",
                speaker_id="s",
                duration_seconds=1.0,
                sample_rate=24_000,
            )
        missing = {error["loc"][0] for error in excinfo.value.errors()}
        assert {"source", "license_id", "commercial_use_permitted"} <= missing


class TestManifestReporting:
    def test_total_duration(self) -> None:
        utterances = [_utterance(id=f"u{i}", duration_seconds=3600.0) for i in range(2)]
        assert total_duration_hours(utterances) == pytest.approx(2.0)

    def test_count_by_enum_field(self) -> None:
        utterances = [
            _utterance(id="a", dialect=Dialect.KOZHIKODE),
            _utterance(id="b", dialect=Dialect.KOZHIKODE),
            _utterance(id="c", dialect=Dialect.THRISSUR),
        ]
        assert count_by(utterances, "dialect") == {"kozhikode": 2, "thrissur": 1}

    def test_count_by_boolean_field(self) -> None:
        counts = count_by([_utterance(commercial_use_permitted=False)], "commercial_use_permitted")
        assert counts == {"False": 1}

    def test_count_by_unknown_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            count_by([_utterance()], "nope")


class TestSplitAssignment:
    def test_assignment_is_deterministic(self, tmp_path: Path) -> None:
        config = PrepareConfig(output_dir=tmp_path)
        assert assign_split("speaker-7", config) == assign_split("speaker-7", config)

    def test_assignment_is_speaker_disjoint(self, tmp_path: Path) -> None:
        """Utterance ids must not influence the split; only the speaker may."""
        config = PrepareConfig(output_dir=tmp_path)
        assert assign_split("spk", config) == assign_split("spk", config)

    def test_salt_changes_the_assignment(self, tmp_path: Path) -> None:
        a = PrepareConfig(output_dir=tmp_path, split_salt="one")
        b = PrepareConfig(output_dir=tmp_path, split_salt="two")
        speakers = [f"s{i}" for i in range(200)]
        assert [assign_split(s, a) for s in speakers] != [assign_split(s, b) for s in speakers]

    def test_fractions_are_respected_approximately(self, tmp_path: Path) -> None:
        config = PrepareConfig(output_dir=tmp_path, validation_fraction=0.1, test_fraction=0.1)
        splits = [assign_split(f"s{i}", config) for i in range(2_000)]
        train = splits.count(Split.TRAIN) / len(splits)
        assert 0.75 < train < 0.85


class TestCorpusPreparation:
    @pytest.fixture
    def entry(self, tmp_path: Path, speech: SpeechFactory) -> RawEntry:
        save_audio(speech(words=8, seed=3), tmp_path / "in.wav")
        return RawEntry(
            id="u1",
            audio_path=tmp_path / "in.wav",
            text="എൻറെ നാട്",
            speaker_id="s1",
            source="test",
            license_id="CC-BY-4.0",
            commercial_use_permitted=True,
            gender=Gender.FEMALE,
            dialect=Dialect.KOZHIKODE,
        )

    def test_accepts_clean_audio(self, tmp_path: Path, entry: RawEntry) -> None:
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([entry])
        assert len(result.accepted) == 1
        assert result.rejected == []

    def test_normalises_the_transcript(self, tmp_path: Path, entry: RawEntry) -> None:
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([entry])
        utterance = result.accepted[0]
        assert utterance.text == "എൻറെ നാട്"  # verbatim source is preserved
        assert utterance.normalized_text == "എന്റെ നാട്"  # normalised for training
        assert utterance.phonemes

    def test_writes_processed_audio(self, tmp_path: Path, entry: RawEntry) -> None:
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([entry])
        assert Path(result.accepted[0].audio_path).exists()

    def test_asr_verification_accepts_a_match(
        self, tmp_path: Path, entry: RawEntry, transcriber: FakeTranscriber
    ) -> None:
        transcriber.text = "എന്റെ നാട്"
        preparer = CorpusPreparer(
            PrepareConfig(output_dir=tmp_path / "out"), transcriber=transcriber
        )
        result = preparer.run([entry])
        assert result.accepted[0].asr_cer == pytest.approx(0.0)

    def test_asr_verification_rejects_a_mismatch(
        self, tmp_path: Path, entry: RawEntry, transcriber: FakeTranscriber
    ) -> None:
        transcriber.text = "തികച്ചും വേറൊരു വാക്യം ആണിത്"
        preparer = CorpusPreparer(
            PrepareConfig(output_dir=tmp_path / "out"), transcriber=transcriber
        )
        result = preparer.run([entry])
        assert result.accepted == []
        assert result.rejection_summary() == {"asr_mismatch": 1}

    def test_quality_rejection_is_reported(self, tmp_path: Path, entry: RawEntry) -> None:
        import numpy as np

        noise = Audio(
            samples=np.random.default_rng(0).normal(0, 0.2, 48_000).astype(np.float32),
            sample_rate=24_000,
        )
        save_audio(noise, tmp_path / "noise.wav")
        bad = RawEntry(
            id="bad",
            audio_path=tmp_path / "noise.wav",
            text="നാട്",
            speaker_id="s1",
            source="test",
            license_id="CC-BY-4.0",
            commercial_use_permitted=True,
        )
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([bad])
        assert result.rejection_summary() == {"quality": 1}

    def test_a_broken_entry_does_not_abort_the_run(self, tmp_path: Path, entry: RawEntry) -> None:
        missing = RawEntry(
            id="missing",
            audio_path=tmp_path / "nope.wav",
            text="നാട്",
            speaker_id="s2",
            source="test",
            license_id="CC-BY-4.0",
            commercial_use_permitted=True,
        )
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([missing, entry])
        assert len(result.accepted) == 1
        assert "error" in result.rejection_summary()

    def test_accepted_hours(self, tmp_path: Path, entry: RawEntry) -> None:
        result = CorpusPreparer(PrepareConfig(output_dir=tmp_path / "out")).run([entry])
        assert 0.0 < result.accepted_hours < 0.01


class TestTsvReader:
    def test_reads_four_column_rows(self, tmp_path: Path) -> None:
        (tmp_path / "c.tsv").write_text(
            "u1\ta/1.wav\tഒന്ന്\ts1\nu2\ta/2.wav\tരണ്ട്\ts2\n", encoding="utf-8"
        )
        entries = list(
            iter_entries_from_tsv(
                tmp_path / "c.tsv",
                source="src",
                license_id="CC-BY-4.0",
                commercial_use_permitted=True,
                audio_root=tmp_path,
            )
        )
        assert [e.id for e in entries] == ["u1", "u2"]
        assert entries[0].audio_path == tmp_path / "a/1.wav"

    def test_malformed_rows_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "c.tsv").write_text("u1\tonly\ttwo\nu2\ta.wav\tട\ts1\n", encoding="utf-8")
        entries = list(
            iter_entries_from_tsv(
                tmp_path / "c.tsv",
                source="src",
                license_id="x",
                commercial_use_permitted=False,
            )
        )
        assert [e.id for e in entries] == ["u2"]
