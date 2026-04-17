"""
Navigator - Campus Guide Robot
Main entrypoint. No business logic lives here.
This file bootstraps the application only.
"""

from app.config import get_settings


def main() -> None:
    """Application entry point."""
    print("Navigator is starting up...")

    cfg = get_settings()
    print(f"  Log level      : {cfg.log_level}")
    print(f"  Sample rate    : {cfg.mic_sample_rate} Hz")
    print(f"  Wake word      : '{cfg.wake_word}'")
    print(f"  DB path        : {cfg.sqlite_db_path}")
    print(f"  Default lang   : {cfg.default_language}")
    print(f"  Deepgram key   : {'✅ set' if cfg.has_deepgram_key else '⚠️  not set'}")
    print(f"  Groq key       : {'✅ set' if cfg.has_groq_key else '⚠️  not set'}")

    print("\nNavigator config loaded OK ✅")


if __name__ == "__main__":
    main()
