"""
Navigator - Campus Guide Robot
Main entrypoint.

This file bootstraps storage and then starts the live Pipecat runtime.
"""

from __future__ import annotations

import asyncio

from app.config import get_settings
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.routing.router import shutdown_router
from app.storage import bootstrap_schema, close_db, get_table_counts
from app.storage.sync_csv import sync_all_csvs
from app.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)

    try:
        cfg = get_settings()
        logger.info(
            "navigator_starting",
            wake_word=cfg.wake_word,
            db_path=cfg.sqlite_db_path,
            groq_key_set=cfg.has_groq_key,
        )

        bootstrap_schema()
        sync_all_csvs()

        counts = get_table_counts()
        logger.info("db_table_counts", **counts)
        logger.info("navigator_boot_ok")

        runtime = NavigatorPipecatRuntime()
        try:
            asyncio.run(runtime.run())
        except KeyboardInterrupt:
            logger.info("navigator_stopped_by_user")
    finally:
        shutdown_router()
        close_db()


if __name__ == "__main__":
    main()
