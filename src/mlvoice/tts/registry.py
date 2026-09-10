"""Backend registry.

Maps a configuration string to a constructed :class:`~mlvoice.tts.base.Synthesizer`.
Keeping construction in one place means the service, the CLI and the evaluation
harness all resolve backends identically, and adding a backend touches exactly
one function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from mlvoice.config import Settings
from mlvoice.errors import ConfigurationError
from mlvoice.tts.base import Synthesizer

__all__ = ["available_backends", "build_synthesizer", "register_backend"]

_Factory = Callable[[Settings], Synthesizer]


def _build_dummy(settings: Settings) -> Synthesizer:
    from mlvoice.tts.dummy import DummySynthesizer

    return DummySynthesizer(sample_rate=settings.sample_rate)


def _build_indicf5(settings: Settings) -> Synthesizer:
    from mlvoice.tts.indicf5 import IndicF5Synthesizer

    return IndicF5Synthesizer(
        model_id=settings.model_id,
        revision=settings.model_revision,
        device=settings.device,
    )


_REGISTRY: Final[dict[str, _Factory]] = {
    "dummy": _build_dummy,
    "indicf5": _build_indicf5,
}


def register_backend(name: str, factory: _Factory) -> None:
    """Register a backend factory under ``name``.

    Raises:
        ConfigurationError: ``name`` is already registered. Overwriting a
            backend silently is how two deployments end up running different
            models under the same name.
    """
    if name in _REGISTRY:
        raise ConfigurationError("backend already registered", backend=name)
    _REGISTRY[name] = factory


def available_backends() -> tuple[str, ...]:
    """Registered backend names, sorted."""
    return tuple(sorted(_REGISTRY))


def build_synthesizer(settings: Settings) -> Synthesizer:
    """Construct the backend named by ``settings.tts_backend``.

    The returned backend is *not* loaded; call
    :meth:`~mlvoice.tts.base.Synthesizer.load` during application startup so
    that a weight-loading failure surfaces before the process accepts traffic.

    Raises:
        ConfigurationError: The configured backend is not registered.
    """
    factory = _REGISTRY.get(settings.tts_backend)
    if factory is None:
        raise ConfigurationError(
            "unknown tts backend",
            backend=settings.tts_backend,
            available=list(available_backends()),
        )
    return factory(settings)
