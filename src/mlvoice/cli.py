"""Command line interface.

The operational surface of the project. Everything the service does is also
reachable here, which matters for three reasons: the text frontend can be
inspected and gated without starting a server, corpus preparation is a batch job
rather than an HTTP call, and the evaluation harness is meant to be run against
several systems from a terminal before any model is chosen.

Run ``mlvoice --help`` for the command list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from mlvoice.__version__ import __version__

app = typer.Typer(
    name="mlvoice",
    help="Malayalam text-to-speech and voice cloning.",
    no_args_is_help=True,
    add_completion=False,
)
text_app = typer.Typer(help="Inspect the Malayalam text frontend.", no_args_is_help=True)
eval_app = typer.Typer(help="Run the Malayalam evaluation harness.", no_args_is_help=True)
data_app = typer.Typer(help="Prepare training corpora.", no_args_is_help=True)
mark_app = typer.Typer(help="Embed and detect audio watermarks.", no_args_is_help=True)
app.add_typer(text_app, name="text")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(mark_app, name="watermark")


@app.callback()
def _configure(quiet: bool = False) -> None:
    """Configure logging before any command runs.

    Diagnostics go to stderr so that stdout carries only command output, which
    is JSON for most commands and therefore has to stay parseable.
    """
    from mlvoice.config import get_settings
    from mlvoice.logging import configure_logging

    settings = get_settings()
    configure_logging(
        level="ERROR" if quiet else settings.log_level,
        json_output=settings.log_json,
        stream=sys.stderr,
    )


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_text(text: str | None, file: Path | None) -> str:
    """Take text from an argument, a file, or standard input."""
    if text:
        return text
    if file is not None:
        return file.read_text(encoding="utf-8")
    data = sys.stdin.read()
    if not data.strip():
        raise typer.BadParameter("provide TEXT, --file, or pipe text on stdin")
    return data


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def info() -> None:
    """Print the effective configuration and the registered backends."""
    from mlvoice.config import get_settings
    from mlvoice.eval.testset import category_counts
    from mlvoice.tts.registry import available_backends

    settings = get_settings()
    _echo_json(
        {
            "version": __version__,
            "environment": settings.env.value,
            "backend": settings.tts_backend,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "device": settings.device,
            "sample_rate": settings.sample_rate,
            "consent_required": settings.require_consent,
            "watermarking": settings.watermark_enabled,
            "available_backends": list(available_backends()),
            "test_set": category_counts(),
        }
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: Annotated[int, typer.Option(help="Ignored when --reload is set.")] = 1,
) -> None:
    """Run the HTTP service."""
    import uvicorn

    # access_log=False because RequestContextMiddleware already emits one
    # structured line per request, with the request id bound. uvicorn installs
    # its own logging config when it starts, after configure_logging has run,
    # so silencing the access logger from our side does not survive; this flag
    # is what actually stops the duplicate.
    uvicorn.run(
        "mlvoice.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        access_log=False,
    )


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


@text_app.command("normalize")
def text_normalize(
    text: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    aggressive: Annotated[
        bool,
        typer.Option(
            help="Also repair chillu+consonant sequences that are usually, but "
            "not always, legacy mis-encodings."
        ),
    ] = False,
    report: bool = False,
) -> None:
    """Normalise Malayalam Unicode: chillu, legacy nta, archaic signs, digits."""
    from mlvoice.text.unicode_norm import normalize_with_report

    normalized, result = normalize_with_report(_read_text(text, file), aggressive_legacy=aggressive)
    if report:
        _echo_json(
            {
                "normalized": normalized,
                "changed": result.changed,
                "total_changes": result.total_changes,
                "chillu_composed": result.chillu_composed,
                "legacy_nta_fixed": result.legacy_nta_fixed,
                "archaic_mapped": result.archaic_mapped,
                "digits_converted": result.digits_converted,
                "notes": result.notes,
            }
        )
    else:
        typer.echo(normalized)


@text_app.command("analyze")
def text_analyze(
    text: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Show every frontend stage: normalisation, expansion, routing, chunks, phonemes."""
    from mlvoice.text.g2p import Notation
    from mlvoice.text.pipeline import TextPipeline

    processed = TextPipeline().process(_read_text(text, file))
    _echo_json(
        {
            "normalized": processed.normalized,
            "expanded": processed.expanded,
            "routed": processed.routed,
            "summary": processed.summary(),
            "chunks": [
                {
                    "index": chunk.index,
                    "text": chunk.text,
                    "break_after": chunk.break_after.value,
                    "pause_ms": chunk.pause_ms,
                    "ipa": processed.phoneme_string(chunk.index),
                    "ascii": processed.phoneme_string(chunk.index, notation=Notation.ASCII),
                }
                for chunk in processed.chunks
            ],
        }
    )


@text_app.command("g2p")
def text_g2p(
    text: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    ascii_symbols: Annotated[
        bool, typer.Option("--ascii", help="Emit the stable ASCII symbol set.")
    ] = False,
    intervocalic_voicing: bool = False,
) -> None:
    """Phonemise Malayalam text."""
    from mlvoice.text.g2p import G2PConfig, Notation, phonemize, to_string
    from mlvoice.text.unicode_norm import normalize

    source = normalize(_read_text(text, file))
    words = phonemize(source, G2PConfig(intervocalic_voicing=intervocalic_voicing))
    notation = Notation.ASCII if ascii_symbols else Notation.IPA
    typer.echo(to_string(words, notation=notation))


@text_app.command("number")
def text_number(value: int) -> None:
    """Read an integer aloud in Malayalam."""
    from mlvoice.text.numbers import cardinal, ordinal

    _echo_json({"cardinal": cardinal(value), "ordinal": ordinal(value) if value > 0 else None})


# ---------------------------------------------------------------------------
# speak
# ---------------------------------------------------------------------------


@app.command()
def speak(
    text: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Read the text to speak from a file. Preferred for long-form "
            "input: paragraph breaks and Malayalam punctuation survive intact "
            "instead of going through shell quoting.",
        ),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out.wav"),
    reference_audio: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Reference clip to clone."),
    ] = None,
    reference_text: Annotated[
        str | None, typer.Option(help="Verbatim transcript of the reference clip.")
    ] = None,
    reference_text_file: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Read the transcript from a file."),
    ] = None,
    speed: float = 1.0,
    seed: int | None = None,
) -> None:
    """Synthesise text to a WAV file.

    Text comes from the argument, ``--file``, or standard input. Long-form input
    should use ``--file``: shell quoting mangles paragraph breaks, and a
    mis-quoted 700-character Malayalam script is hard to spot.

    Cloning needs ``--reference-audio``. A transcript of that clip is what the
    backend conditions on; supply it with ``--reference-text`` or
    ``--reference-text-file``, or omit it and the clip is transcribed. The
    transcript must be what the recording *already says*, not the text being
    generated: the backend aligns the two, and a mismatch clones the voice
    correctly while garbling the words.
    """
    from mlvoice.asr import build_transcriber
    from mlvoice.audio.io import load_audio, save_audio
    from mlvoice.config import get_settings
    from mlvoice.text.pipeline import TextPipeline
    from mlvoice.tts.base import ReferencePrompt, SynthesisRequest
    from mlvoice.tts.registry import build_synthesizer

    source_text = _read_text(text, file)
    if reference_text_file is not None:
        if reference_text is not None:
            raise typer.BadParameter("give --reference-text or --reference-text-file, not both")
        reference_text = reference_text_file.read_text(encoding="utf-8").strip()
    settings = get_settings()
    synthesizer = build_synthesizer(settings)
    synthesizer.load()

    prompt = None
    if reference_audio is not None:
        reference = load_audio(reference_audio, target_sample_rate=settings.sample_rate)
        if reference_text is None:
            transcriber = build_transcriber(settings)
            if transcriber is None:
                raise typer.BadParameter(
                    "no transcript given and transcription is disabled: pass "
                    "--reference-text/--reference-text-file, or enable it with "
                    "MLVOICE_ASR_ENABLED=true"
                )
            typer.secho("transcribing the reference clip...", fg=typer.colors.CYAN, err=True)
            reference_text = transcriber.transcribe(reference)
            if not reference_text.strip():
                raise typer.BadParameter(
                    "the reference clip could not be transcribed; it may be "
                    "silent, too noisy, or not speech"
                )
            typer.secho(f"reference transcript: {reference_text}", fg=typer.colors.CYAN, err=True)
        prompt = ReferencePrompt(
            audio=reference,
            text=reference_text,
            voice_id="cli-reference",
        )
        for note in prompt.advisories():
            typer.secho(f"warning: {note}", fg=typer.colors.YELLOW, err=True)
        if reference_text.strip() == source_text.strip():
            typer.secho(
                "warning: --reference-text is identical to the text being "
                "generated. It should be the transcript of --reference-audio, "
                "not the target text; if they differ, the voice clones but the "
                "words garble.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    processed = TextPipeline().process(source_text)
    result = synthesizer.synthesize(
        SynthesisRequest(text=processed, prompt=prompt, speed=speed, seed=seed)
    )
    save_audio(result.audio, output)
    _echo_json({"output": str(output), **result.metrics()})


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@eval_app.command("frontend")
def eval_frontend(
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    fail_under: Annotated[
        float, typer.Option(help="Exit non-zero if the pass rate falls below this.")
    ] = 1.0,
) -> None:
    """Score the text frontend against the Malayalam hard test set.

    Needs no model and no GPU. This is the check that belongs in CI.
    """
    from mlvoice.eval.run import EvaluationHarness

    report = EvaluationHarness().evaluate_frontend()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_json(), encoding="utf-8")
    typer.echo(
        f"cases={len(report.results)} asserted={len(report.asserted)} "
        f"pass_rate={report.pass_rate:.4f}"
    )
    for failure in report.failures:
        typer.echo(
            f"  FAIL {failure.id}: expected {failure.expected_text!r}, got {failure.routed!r}"
        )
    if report.pass_rate < fail_under:
        raise typer.Exit(code=1)


@eval_app.command("synthesis")
def eval_synthesis(
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    reference_audio: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    reference_text: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Synthesise the whole test set and report objective metrics.

    Round-trip CER requires an ASR model and is omitted here; wire a
    :class:`mlvoice.protocols.Transcriber` in to enable it. Real-time factor and
    audio quality are always reported.
    """
    from mlvoice.audio.io import load_audio
    from mlvoice.config import get_settings
    from mlvoice.eval.run import EvaluationHarness
    from mlvoice.tts.base import ReferencePrompt
    from mlvoice.tts.registry import build_synthesizer

    settings = get_settings()
    synthesizer = build_synthesizer(settings)
    synthesizer.load()

    prompt = None
    if reference_audio is not None and reference_text is not None:
        prompt = ReferencePrompt(
            audio=load_audio(reference_audio, target_sample_rate=settings.sample_rate),
            text=reference_text,
            voice_id="cli-reference",
        )

    report = EvaluationHarness().evaluate_synthesis(synthesizer, prompt=prompt)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_json(), encoding="utf-8")
    payload = report.to_dict()
    typer.echo(
        json.dumps(
            {k: payload[k] for k in ("backend", "cases", "errors", "mean_real_time_factor")},
            ensure_ascii=False,
        )
    )


@eval_app.command("testset")
def eval_testset(
    category: Annotated[str | None, typer.Option(help="Restrict to one category.")] = None,
    instructions: Annotated[
        bool, typer.Option(help="Print the native-speaker rating instructions.")
    ] = False,
) -> None:
    """Print the Malayalam hard test set, or the rating instructions."""
    from mlvoice.eval.testset import TEST_CASES, Category, category_counts, scoring_instructions

    if instructions:
        typer.echo(scoring_instructions())
        return
    if category is None:
        _echo_json(category_counts())
        return
    try:
        wanted = Category(category)
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown category; choose from {[c.value for c in Category]}"
        ) from exc
    for case in TEST_CASES:
        if case.category is wanted:
            typer.echo(f"{case.id}\t{case.text}\t{case.probes}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@data_app.command("prepare")
def data_prepare(
    tsv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    manifest: Annotated[Path, typer.Option("--manifest", "-m")],
    source: Annotated[str, typer.Option(help="Dataset name recorded on every row.")],
    license_id: Annotated[str, typer.Option(help="SPDX id or documented licence key.")],
    commercial: Annotated[
        bool, typer.Option(help="Whether the source licence permits commercial use.")
    ] = False,
    audio_root: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    sample_rate: int = 24_000,
    no_trim: bool = False,
) -> None:
    """Prepare a corpus from a TSV of ``id<TAB>audio<TAB>text<TAB>speaker`` rows.

    Writes processed audio under ``--output-dir`` and a validated JSONL manifest
    to ``--manifest``. ASR verification, the single most valuable filter, is not
    wired in from the CLI: import :class:`mlvoice.data.CorpusPreparer` with a
    :class:`mlvoice.protocols.Transcriber` to enable it.
    """
    from mlvoice.data.manifest import write_manifest
    from mlvoice.data.prepare import (
        CorpusPreparer,
        PrepareConfig,
        iter_entries_from_tsv,
    )

    config = PrepareConfig(output_dir=output_dir, target_sample_rate=sample_rate, trim=not no_trim)
    entries = iter_entries_from_tsv(
        tsv,
        source=source,
        license_id=license_id,
        commercial_use_permitted=commercial,
        audio_root=audio_root,
    )
    result = CorpusPreparer(config).run(entries)
    written = write_manifest(manifest, result.accepted)
    _echo_json(
        {
            "manifest": str(manifest),
            "written": written,
            "accepted_hours": round(result.accepted_hours, 3),
            "rejected": len(result.rejected),
            "rejections": result.rejection_summary(),
        }
    )


@data_app.command("stats")
def data_stats(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Report corpus balance from a manifest: hours, speakers, dialects, splits."""
    from mlvoice.data.manifest import count_by, read_manifest, total_duration_hours

    utterances = list(read_manifest(manifest))
    _echo_json(
        {
            "utterances": len(utterances),
            "hours": round(total_duration_hours(utterances), 3),
            "speakers": len({u.speaker_id for u in utterances}),
            "by_split": count_by(utterances, "split"),
            "by_dialect": count_by(utterances, "dialect"),
            "by_gender": count_by(utterances, "gender"),
            "by_source": count_by(utterances, "source"),
            "commercial_use_permitted": count_by(utterances, "commercial_use_permitted"),
        }
    )


# ---------------------------------------------------------------------------
# watermark
# ---------------------------------------------------------------------------


@mark_app.command("embed")
def watermark_embed(
    audio_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    payload: Annotated[int, typer.Option(help="32-bit identifier to embed.")],
    key: Annotated[str, typer.Option(envvar="MLVOICE_WATERMARK_KEY")] = "",
) -> None:
    """Embed a watermark payload into an audio file."""
    from mlvoice.audio.io import load_audio, save_audio
    from mlvoice.safety.watermark import SpreadSpectrumWatermarker

    if not key:
        raise typer.BadParameter("--key or MLVOICE_WATERMARK_KEY is required")
    marker = SpreadSpectrumWatermarker(key.encode())
    save_audio(marker.embed(load_audio(audio_path), payload), output)
    _echo_json({"output": str(output), "payload": payload})


@mark_app.command("detect")
def watermark_detect(
    audio_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    key: Annotated[str, typer.Option(envvar="MLVOICE_WATERMARK_KEY")] = "",
) -> None:
    """Detect a watermark and print the payload it carries."""
    from mlvoice.audio.io import load_audio
    from mlvoice.safety.watermark import SpreadSpectrumWatermarker

    if not key:
        raise typer.BadParameter("--key or MLVOICE_WATERMARK_KEY is required")
    marker = SpreadSpectrumWatermarker(key.encode())
    _echo_json(marker.detect(load_audio(audio_path)).as_dict())


if __name__ == "__main__":  # pragma: no cover
    app()
