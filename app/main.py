"""
Navigator - Campus Guide Robot
Main entrypoint.

This file bootstraps storage and then starts the live Pipecat runtime.
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.routing.router import shutdown_router
from app.storage import bootstrap_schema, close_db, get_table_counts
from app.storage.sync_csv import sync_all_csvs
from scripts.health_check import run_health_checks
from app.utils.logging import get_logger, setup_logging


def _prewarm_groq(logger) -> None:
    """Warm the Groq HTTPS path without blocking startup on failure."""
    try:
        from app.routing.router import _get_groq

        groq = _get_groq()
        groq.complete_text(system_prompt="ping", user_message="ping", max_tokens=1)
        logger.info("navigator_groq_prewarm_ok")
    except Exception as exc:
        logger.warning("navigator_groq_prewarm_failed", error=str(exc))


async def main() -> None:
    setup_logging()
    logger = get_logger(__name__)

    try:
        cfg = get_settings()
        logger.info(
            "navigator_starting",
            wake_word=cfg.wake_word,
            db_path=cfg.sqlite_db_path,
            deepgram_key_set=cfg.has_deepgram_key,
            elevenlabs_key_set=cfg.has_elevenlabs_key,
            groq_key_set=cfg.has_groq_key,
            groq_model=cfg.groq_model,
            tts_voice_ar=cfg.edge_tts_voice_ar,
            tts_voice_en=cfg.edge_tts_voice_en,
        )

        bootstrap_schema()
        sync_all_csvs()

        counts = get_table_counts()
        logger.info("db_table_counts", **counts)

        if not run_health_checks():
            logger.error("navigator_startup_blocked", reason="health_check_failed")
            sys.exit(1)

        from app.stt.deepgram_client import load_arabic_keyterms_from_db, load_keyterms_from_db

        en_terms = load_keyterms_from_db()
        ar_terms = load_arabic_keyterms_from_db()
        logger.info("navigator_keyterms_loaded", english_count=len(en_terms), arabic_count=len(ar_terms))

        _prewarm_groq(logger)
        try:
            from app.tts.edge_tts_client import EdgeTTSClient

            await EdgeTTSClient().prewarm_fallback()
            logger.info("navigator_tts_fallback_prewarmed")
        except Exception as exc:
            logger.warning("navigator_tts_fallback_prewarm_failed", error=str(exc))

        logger.info("navigator_boot_ok")

        max_restarts = 50
        restart_count = 0
        restart_delay = 3.0

        while restart_count < max_restarts:
            runtime = NavigatorPipecatRuntime()
            try:
                logger.info("navigator_runtime_starting", attempt=restart_count + 1)
                await runtime.run()
                logger.info("navigator_runtime_clean_exit")
                break
            except KeyboardInterrupt:
                logger.info("navigator_stopped_by_user")
                break
            except Exception as exc:
                restart_count += 1
                logger.error(
                    "navigator_runtime_crashed",
                    error=str(exc),
                    restart_count=restart_count,
                    restarting_in_sec=restart_delay,
                    exc_info=True,
                )
                try:
                    await runtime.shutdown()
                except Exception as shutdown_exc:
                    logger.warning("navigator_runtime_shutdown_after_crash_failed", error=str(shutdown_exc))
                if restart_count >= max_restarts:
                    logger.error("navigator_max_restarts_exceeded", max_restarts=max_restarts)
                    sys.exit(1)
                await asyncio.sleep(restart_delay)
                logger.info("navigator_restarting")
    finally:
        get_logger(__name__).info("navigator_shutdown")
        shutdown_router()
        close_db()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
