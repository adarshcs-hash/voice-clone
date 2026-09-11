"""Pre-flight checks: will this process actually synthesise speech?

Every failure this module reports is one that has already cost someone an hour.
They share a shape: the service starts, answers requests, and is wrong --
a backend that buzzes instead of speaking, weights that are not cached so the
first request stalls on a multi-gigabyte download, the PyPI ``f5-tts`` instead
of AI4Bharat's fork, a ``transformers`` new enough to make the model unloadable.
None of those announce themselves as misconfiguration.

So this answers the question directly, without loading a model: what is
configured, what is installed, what is cached, and what will break. It imports
nothing expensive and never touches the network, so it is safe to run in a
container build, a health script, or before a deploy.

The checks are data, not prints -- :func:`run_checks` returns them and the CLI
renders them -- so they can be asserted on in tests and consumed by tooling.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Final

from mlvoice.config import Environment, Settings

__all__ = ["Check", "Status", "run_checks", "worst_status"]

_DUMMY: Final = "dummy"
_TRANSFORMERS_CEILING: Final = (4, 50)


class Status(StrEnum):
    """How bad a finding is.

    ``FAIL`` means this process cannot do what it is configured to do. ``WARN``
    means it can, but something will be slow, unpinned, or unsafe.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    """One finding.

    Attributes:
        name: Short stable identifier, suitable for a script to match on.
        status: How bad it is.
        detail: What was observed.
        hint: What to do about it. Empty when there is nothing to do.
    """

    name: str
    status: Status
    detail: str
    hint: str = ""

    def as_dict(self) -> dict[str, str]:
        """Render for JSON output."""
        payload = {"check": self.name, "status": self.status.value, "detail": self.detail}
        if self.hint:
            payload["hint"] = self.hint
        return payload


def worst_status(checks: list[Check]) -> Status:
    """The most severe status in ``checks``, or ``OK`` when there are none."""
    if any(check.status is Status.FAIL for check in checks):
        return Status.FAIL
    if any(check.status is Status.WARN for check in checks):
        return Status.WARN
    return Status.OK


# --------------------------------------------------------------------- cache --


def hf_cache_root() -> Path:
    """Where the Hugging Face hub cache lives, following the documented vars.

    Resolved by hand rather than by importing ``huggingface_hub``, because this
    module must work on an install that has no model runtime at all -- which is
    exactly the install most likely to need diagnosing.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_cached(model_id: str) -> bool:
    """Whether ``model_id`` has at least one materialised snapshot locally."""
    folder = hf_cache_root() / f"models--{model_id.replace('/', '--')}"
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(entry.is_dir() and any(entry.iterdir()) for entry in snapshots.iterdir())


# ------------------------------------------------------------------- installs --


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _parsed_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _f5_tts_is_the_fork() -> bool | None:
    """Whether the installed ``f5_tts`` is AI4Bharat's, or ``None`` if unknown.

    The two distributions share the import name and the distribution name, so
    the only reliable discriminator is the API: upstream's ``load_model`` takes
    a required ``ckpt_path`` positional, AI4Bharat's does not, and IndicF5's
    bundled code calls the latter. Getting this wrong fails with
    "load_model() missing 1 required positional argument", which names neither
    package.
    """
    try:
        import inspect

        from f5_tts.infer.utils_infer import load_model
    except Exception:
        return None
    try:
        parameters = inspect.signature(load_model).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None
    ckpt = parameters.get("ckpt_path")
    if ckpt is None:
        return True
    return ckpt.default is not inspect.Parameter.empty


# -------------------------------------------------------------------- checks --


def _check_backend(settings: Settings) -> Iterator[Check]:
    if settings.tts_backend == _DUMMY:
        yield Check(
            "tts_backend",
            Status.FAIL if settings.is_production else Status.WARN,
            "the dummy backend is configured: output is a synthetic buzz, not speech",
            "set MLVOICE_TTS_BACKEND=indicf5",
        )
        return
    yield Check("tts_backend", Status.OK, f"{settings.tts_backend} ({settings.model_id})")

    if settings.model_revision:
        yield Check("model_revision", Status.OK, f"pinned at {settings.model_revision}")
    else:
        yield Check(
            "model_revision",
            Status.FAIL if settings.is_production else Status.WARN,
            "weights are unpinned",
            "IndicF5 runs with trust_remote_code, so an unpinned revision is "
            "remote code execution from a moving target; set MLVOICE_MODEL_REVISION",
        )

    if _is_cached(settings.model_id):
        yield Check("model_weights", Status.OK, f"cached under {hf_cache_root()}")
    else:
        yield Check(
            "model_weights",
            Status.WARN,
            f"{settings.model_id} is not in the local cache",
            "the first synthesis will download it; bake the weights into the "
            "image or warm a shared cache volume before serving traffic",
        )


def _check_runtime(settings: Settings) -> Iterator[Check]:
    if settings.tts_backend == _DUMMY:
        return

    torch_version = _version("torch")
    if torch_version is None:
        yield Check(
            "torch",
            Status.FAIL,
            "not installed",
            'pip install -e ".[indicf5]"',
        )
        return
    yield Check("torch", Status.OK, torch_version)

    transformers_version = _version("transformers")
    if transformers_version is None:
        yield Check("transformers", Status.FAIL, "not installed", 'pip install -e ".[indicf5]"')
    elif _parsed_version(transformers_version)[:2] >= _TRANSFORMERS_CEILING:
        yield Check(
            "transformers",
            Status.FAIL,
            f"{transformers_version} is too new for IndicF5",
            "4.51 made meta-device initialisation unconditional, under which "
            "the model's vocoder cannot be moved to a device; install "
            "'transformers>=4.44,<4.50'",
        )
    else:
        yield Check("transformers", Status.OK, transformers_version)

    fork = _f5_tts_is_the_fork()
    if fork is None:
        yield Check(
            "f5_tts",
            Status.FAIL,
            "not importable",
            'pip install -e ".[indicf5]", which installs AI4Bharat\'s fork from GitHub',
        )
    elif fork:
        yield Check("f5_tts", Status.OK, "AI4Bharat fork")
    else:
        yield Check(
            "f5_tts",
            Status.FAIL,
            "this is the f5-tts package from PyPI, not AI4Bharat's fork",
            "they share the import name but not the API; reinstall with "
            'pip install -e ".[indicf5]"',
        )

    if _version("pydub") is None:
        yield Check(
            "pydub",
            Status.FAIL,
            "not installed",
            "transformers checks for it by name before the model loads",
        )
    else:
        yield Check("pydub", Status.OK, "installed")


def _check_device(settings: Settings) -> Iterator[Check]:
    if settings.tts_backend == _DUMMY or _version("torch") is None:
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - a broken torch install
        yield Check("device", Status.FAIL, f"torch will not import: {exc}")
        return

    if settings.device == "cuda" and not torch.cuda.is_available():
        yield Check("device", Status.FAIL, "cuda is configured but unavailable")
        return
    if settings.device == "mps" and not torch.backends.mps.is_available():
        yield Check("device", Status.FAIL, "mps is configured but unavailable")
        return
    if settings.device == "cpu":
        accelerators = []
        if torch.cuda.is_available():
            accelerators.append("cuda")
        if torch.backends.mps.is_available():
            accelerators.append("mps")
        if accelerators:
            yield Check(
                "device",
                Status.WARN,
                f"running on cpu while {' and '.join(accelerators)} is available",
                f"set MLVOICE_DEVICE={accelerators[0]}; this model is several times faster there",
            )
            return
    yield Check("device", Status.OK, settings.device)


def _check_transcription(settings: Settings) -> Iterator[Check]:
    if not settings.asr_enabled:
        yield Check(
            "transcription",
            Status.FAIL if (settings.is_production and settings.require_consent) else Status.WARN,
            "disabled",
            "callers must type the transcript of their own reference clip, "
            "which is the main source of bad clones; consent cannot be verified",
        )
        return
    if _is_cached(settings.asr_model_id):
        yield Check("transcription", Status.OK, f"{settings.asr_model_id} (cached)")
        return
    yield Check(
        "transcription",
        Status.WARN,
        f"{settings.asr_model_id} is not in the local cache",
        "the first transcription will download it, which for whisper-large-v3 "
        "is about 3 GB; openai/whisper-small is ~480 MB if that is too slow",
    )


def _check_safety(settings: Settings) -> Iterator[Check]:
    if settings.parsed_api_keys:
        yield Check("auth", Status.OK, f"{len(settings.parsed_api_keys)} key(s) accepted")
    else:
        yield Check(
            "auth",
            Status.FAIL if settings.is_production else Status.WARN,
            "no API keys: every endpoint is open to anyone who can reach the port",
            "set MLVOICE_API_KEYS before exposing this beyond localhost",
        )

    if settings.require_consent:
        yield Check("consent", Status.OK, "required and verified per request")
    else:
        yield Check(
            "consent",
            Status.FAIL if settings.is_production else Status.WARN,
            "not required: any uploaded clip can be cloned",
            "acceptable for your own voice on your own machine, not for anyone "
            "else's; set MLVOICE_REQUIRE_CONSENT=true",
        )

    if not settings.watermark_enabled:
        yield Check(
            "watermark",
            Status.FAIL if settings.is_production else Status.WARN,
            "disabled: generated audio carries no provenance mark",
            "set MLVOICE_WATERMARK_ENABLED=true",
        )
    elif not settings.watermark_key.get_secret_value():
        yield Check("watermark", Status.FAIL, "enabled but MLVOICE_WATERMARK_KEY is empty")
    else:
        yield Check("watermark", Status.OK, "enabled")


def run_checks(settings: Settings, *, as_production: bool = False) -> list[Check]:
    """Run every pre-flight check against ``settings``.

    Args:
        settings: The configuration to inspect.
        as_production: Apply production severities regardless of
            ``MLVOICE_ENV``. Use this to find out what a deployment would
            refuse *before* deploying it, rather than from a failed rollout.

    Returns:
        Findings in the order they are worth reading: what will be loaded,
        whether it can be, then the safety posture.
    """
    subject = (
        settings.model_copy(update={"env": Environment.PRODUCTION})
        if as_production and settings.env is not Environment.PRODUCTION
        else settings
    )
    checks = [Check("environment", Status.OK, subject.env.value)]
    checks.extend(_check_backend(subject))
    checks.extend(_check_runtime(subject))
    checks.extend(_check_device(subject))
    checks.extend(_check_transcription(subject))
    checks.extend(_check_safety(subject))
    return checks
