"""The browser client.

Two files served by explicit routes rather than a :class:`StaticFiles` mount.
A mount would serve whatever happens to be in the directory -- including a
stray file left there by a build or a developer -- whereas naming the two
assets means the served surface cannot grow by accident.

Caching, which this got wrong once in a way worth recording. The HTML was
served ``no-store`` and the script ``max-age=3600`` at a fixed URL, on the
reasoning that the page must never be stale. The effect was the opposite of
the intent: after an update the browser fetched the new HTML and kept the old
script for up to an hour, so the page bound handlers that no longer existed and
looked exactly as it had before the update. A user who pulled a fix saw no
change and reasonably concluded the fix had not worked.

The script URL now carries a hash of its own contents. The HTML is still
uncached, so it always names the current hash; the script is immutable at that
URL and cached hard. New bytes mean a new URL, so an update is picked up on the
next load and nothing is ever half-updated.

The assets are read from disk on each request in development so an edit shows
up on reload, and read once and held in production where the files cannot
change under a running process.
"""

from __future__ import annotations

import hashlib
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
# Safe to cache forever because the URL changes when the bytes do.
_IMMUTABLE: Final = "public, max-age=31536000, immutable"
_SCRIPT: Final = "app.js"
_INDEX: Final = "index.html"
_SCRIPT_URL: Final = "/ui/app.js"
_FINGERPRINT_LENGTH: Final = 12


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


def fingerprint(content: str) -> str:
    """A short content hash, used to version the script URL."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _versioned_index(settings: Settings) -> str:
    """The page with its script reference pinned to the script's contents.

    Rewritten at serve time rather than stored in the file, so the HTML on disk
    stays a plain page a developer can open, and the two assets cannot
    disagree about the version: the hash is computed from the bytes actually
    about to be served.
    """
    markup = _asset(_INDEX, settings)
    script = _asset(_SCRIPT, settings)
    return markup.replace(_SCRIPT_URL, f"{_SCRIPT_URL}?v={fingerprint(script)}")


@router.get("/ui", summary="Web client")
def ui_index(settings: Annotated[Settings, Depends(get_settings_dep)]) -> Response:
    """Serve the single-page client."""
    _guard(settings)
    return Response(
        content=_versioned_index(settings),
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": _NO_STORE},
    )


@router.get("/ui/app.js", summary="Web client script")
def ui_script(settings: Annotated[Settings, Depends(get_settings_dep)]) -> Response:
    """Serve the client script.

    Cached hard: the page requests it at a URL carrying a hash of these bytes,
    so a change produces a different URL rather than a stale hit. A request
    without the query string -- a bookmark, a curl -- gets the same bytes and
    the same policy, which is why the page is the only thing that must not be
    cached.
    """
    _guard(settings)
    return Response(
        content=_asset(_SCRIPT, settings),
        media_type="text/javascript; charset=utf-8",
        headers={"Cache-Control": _IMMUTABLE},
    )
