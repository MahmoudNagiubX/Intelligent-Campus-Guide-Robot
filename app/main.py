"""
Navigator - Campus Guide Robot
Main entrypoint. No business logic lives here.
This file bootstraps the application only.
"""

from app.utils.logging import setup_logging, get_logger
from app.config import get_settings
from app.storage import bootstrap_schema, get_table_counts


def main() -> None:
    """Application entry point."""
    setup_logging()
    logger = get_logger(__name__)

    cfg = get_settings()
    logger.info(
        "navigator_starting",
        log_level=cfg.log_level,
        wake_word=cfg.wake_word,
        db_path=cfg.sqlite_db_path,
        deepgram_key_set=cfg.has_deepgram_key,
        groq_key_set=cfg.has_groq_key,
    )

    # Bootstrap database schema
    bootstrap_schema()

    # Confirm table state
    counts = get_table_counts()
    logger.info("db_table_counts", **counts)

    logger.info("navigator_boot_ok")


if __name__ == "__main__":
    main()
