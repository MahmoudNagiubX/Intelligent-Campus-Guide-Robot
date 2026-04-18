"""
Navigator - Logging Setup
Configures structlog for the entire application.

Usage in any module:
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("session_started", session_id="abc123")
    logger.warning("retrieval_empty", query="robotics lab")
    logger.error("stt_failed", reason="connection timeout")
"""

import logging
import sys
from typing import Any

import structlog

from app.config import get_settings

# Maps string log level names to stdlib logging constants
_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_NOISY_LOGGER_LEVELS: dict[str, int] = {
    "pipecat": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "websockets": logging.WARNING,
    "asyncio": logging.WARNING,
}
_THIRD_PARTY_NOISE_CONFIGURED = False


def _configure_third_party_noise(level: int | None = None) -> None:
    """Reduce noisy stdlib and Loguru output from third-party libraries."""
    global _THIRD_PARTY_NOISE_CONFIGURED

    target_level = level if level is not None else logging.INFO
    for logger_name, logger_level in _NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(max(target_level, logger_level))

    try:
        from loguru import logger as loguru_logger  # type: ignore

        if (not _THIRD_PARTY_NOISE_CONFIGURED) or (level is not None):
            loguru_logger.remove()
            loguru_logger.add(sys.stderr, level="WARNING")
    except ImportError:
        pass

    _THIRD_PARTY_NOISE_CONFIGURED = True


def setup_logging() -> None:
    """
    Configure structlog for the application.
    Call this once at startup in main.py before anything else.

    - In DEBUG mode: pretty colored console output for development.
    - In INFO+ mode: clean key=value console output, easy to read on Pi terminal.
    """
    cfg = get_settings()
    level_str = cfg.log_level.upper()
    level = _LEVEL_MAP.get(level_str, logging.INFO)

    # Configure the stdlib root logger so third-party libraries
    # (deepgram, groq, etc.) also respect our log level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Shared processors applied to every log entry regardless of level.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    if level == logging.DEBUG:
        # Pretty output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Clean key=value output for Pi terminal and log files
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    _configure_third_party_noise(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a named structlog logger for a module.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A bound structlog logger.

    Example:
        logger = get_logger(__name__)
        logger.info("mic_ready", device_index=0, sample_rate=16000)
    """
    if not _THIRD_PARTY_NOISE_CONFIGURED:
        _configure_third_party_noise()
    return structlog.get_logger(name)
