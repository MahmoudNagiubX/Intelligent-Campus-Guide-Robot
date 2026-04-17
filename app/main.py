"""
Navigator - Campus Guide Robot
Main entrypoint. No business logic lives here.
This file bootstraps the application only.
"""

from app.utils.logging import setup_logging, get_logger
from app.config import get_settings


def main() -> None:
    """Application entry point."""
    # Logging must be the very first thing initialized.
    setup_logging()
    logger = get_logger(__name__)

    cfg = get_settings()

    logger.info(
        "navigator_starting",
        log_level=cfg.log_level,
        sample_rate=cfg.mic_sample_rate,
        wake_word=cfg.wake_word,
        db_path=cfg.sqlite_db_path,
        default_language=cfg.default_language,
        deepgram_key_set=cfg.has_deepgram_key,
        groq_key_set=cfg.has_groq_key,
    )

    logger.info("navigator_boot_ok")


if __name__ == "__main__":
    main()
