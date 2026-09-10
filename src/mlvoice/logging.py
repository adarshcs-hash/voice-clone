"""Structured logging.

Production emits one JSON object per line so that logs are queryable without
regex parsing; development emits a human-readable console renderer. A
``request_id`` bound by the API middleware propagates to every downstream log
line through structlog's context vars.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog

_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Configure stdlib logging and structlog. Idempotent.

    Args:
        level: Minimum level to emit.
        json_output: One JSON object per line, rather than a console renderer.
        stream: Where to write. Defaults to stdout, which is right for a
            service. The CLI passes stderr so that stdout stays a clean,
            machine-parseable channel for command output.
    """
    global _configured
    if _configured:
        return

    destination = stream if stream is not None else sys.stdout
    logging.basicConfig(
        format="%(message)s",
        stream=destination,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    # Uvicorn's access log duplicates the request middleware's structured
    # line. This handles the embedded case (someone calling create_app from
    # their own server); the CLI additionally passes access_log=False, because
    # uvicorn reinstalls its own logging config after this function has run.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=destination.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger
