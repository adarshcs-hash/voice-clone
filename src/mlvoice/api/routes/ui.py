"""The browser client.

Two files served by explicit routes rather than a :class:`StaticFiles` mount.
A mount would serve whatever happens to be in the directory -- including a
stray file left there by a build or a developer -- whereas naming the two
assets means the served surface cannot grow by accident. It also lets each
file carry the cache policy it wants: the HTML must not be cached, or a
redeploy leaves browsers on a page whose element ids no longer match the
script.

The assets are read from disk on each request in development so an edit shows
up on reload, and read once and held in production where the files cannot
change under a running process.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Response

from mlvoice.api.deps import get_settings_dep
from mlvoice.config import Environment, Settings
from mlvoice.errors import FeatureDisabledError

__all__ = ["STATIC_PACKAGE", "router"]

router = APIRouter(tags=["ui"], include_in_schema=False)

STATIC_PACKAGE: Final = "mlvoice.api"
_STATIC_DIR: Final = "static"
_NO_STORE: Final = "no-store"
_IMMUTABLE_FOR_AN_HOUR: Final = "public, max-age=3600"


def _read(name: str) -> str:
    """Read a static asset out of the installed package.

    ``importlib.resources`` rather than ``__file__`` arithmetic, so this works
    from a wheel, a zip import, or an editable checkout alike. The anchor is
    ``mlvoice.api`` and not the static directory itself, because that directory
    has no ``__init__.py`` -- resolving a namespace package here would work on
    some Python versions and not others.
    """
    return (resources.files(STATIC_PACKAGE) / _STATIC_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def _read_cached(name: str) -> str:
    return _read(name)


def _asset(name: str, settings: Settings) -> str:
    if settings.env is Environment.DEVELOPMENT:
        return _read(name)
    return _read_cached(name)


def _guard(settings: Settings) -> None:
    if not settings.ui_enabled:
        raise FeatureDisabledError("the web client is disabled", resource="ui")


@router.get("/ui", summary="Web client")
def ui_index(settings: Annotated[Settings, Depends(get_settings_dep)]) -> Response:
    """Serve the single-page client."""
    _guard(settings)
    return Response(
        content=_asset("index.html", settings),
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": _NO_STORE},
    )


@router.get("/ui/app.js", summary="Web client script")
def ui_script(settings: Annotated[Settings, Depends(get_settings_dep)]) -> Response:
    """Serve the client script."""
    _guard(settings)
    return Response(
        content=_asset("app.js", settings),
        media_type="text/javascript; charset=utf-8",
        headers={"Cache-Control": _IMMUTABLE_FOR_AN_HOUR},
    )
