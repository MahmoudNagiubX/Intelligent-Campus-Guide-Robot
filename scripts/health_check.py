"""
Navigator - System Health Check Script

Runs startup validation before the robot enters active mode.
Exit code 1 means startup should be blocked.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(name: str, fn) -> bool:
    try:
        result = bool(fn())
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}  {name}")
        return result
    except Exception as exc:
        print(f"  [FAIL]  {name} -- {exc}")
        return False


def _require_non_empty_file(path: Path) -> bool:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty file: {path}")
    return True


def _require_csv_directory(path: Path) -> bool:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {path}")
    return True


def run_health_checks() -> bool:
    print("\n" + "=" * 40)
    print("  Navigator - Pre-Flight Health Check")
    print("=" * 40 + "\n")

    results: list[bool] = []

    def check_config() -> bool:
        from app.config import get_settings

        cfg = get_settings()
        return bool(cfg.deepgram_api_key and cfg.groq_api_key)

    def check_sqlite() -> bool:
        from app.storage.db import get_db

        conn = get_db()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return len(tables) >= 8

    def check_csv_english_dir() -> bool:
        from app.config import get_settings

        cfg = get_settings()
        return _require_csv_directory(Path(cfg.csv_english_dir))

    def check_csv_arabic_dir() -> bool:
        from app.config import get_settings

        cfg = get_settings()
        return _require_csv_directory(Path(cfg.csv_arabic_dir))

    def check_internet() -> bool:
        socket.setdefaulttimeout(5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("8.8.8.8", 53))
        return True

    def check_deepgram_key_format() -> bool:
        from app.config import get_settings

        key = get_settings().deepgram_api_key
        return isinstance(key, str) and len(key) >= 20

    def check_groq_key_format() -> bool:
        from app.config import get_settings

        key = get_settings().groq_api_key
        return isinstance(key, str) and len(key) >= 20

    def check_prompts() -> bool:
        _require_non_empty_file(Path("prompts/campus_answer_prompt_en.txt"))
        _require_non_empty_file(Path("prompts/campus_answer_prompt_ar.txt"))
        _require_non_empty_file(Path("prompts/social_prompt.txt"))
        _require_non_empty_file(Path("prompts/router_prompt.txt"))
        return True

    def check_pyaudio() -> bool:
        import pyaudio  # noqa: F401

        return True

    def check_edge_tts() -> bool:
        import edge_tts  # noqa: F401

        return True

    results.append(check("Config loads (DEEPGRAM_API_KEY, GROQ_API_KEY present)", check_config))
    results.append(check("SQLite database opens and schema exists", check_sqlite))
    results.append(check("English CSV directory exists and contains CSV files", check_csv_english_dir))
    results.append(check("Arabic CSV directory exists and contains CSV files", check_csv_arabic_dir))
    results.append(check("Internet reachable (DNS test)", check_internet))
    results.append(check("Deepgram API key format looks valid", check_deepgram_key_format))
    results.append(check("Groq API key format looks valid", check_groq_key_format))
    results.append(check("Required prompt files exist and are non-empty", check_prompts))
    results.append(check("PyAudio installed (mic capture)", check_pyaudio))
    results.append(check("edge-tts installed (TTS output)", check_edge_tts))

    passed = sum(1 for item in results if item)
    failed = sum(1 for item in results if not item)
    total = len(results)

    print("\n" + "=" * 40)
    print(f"  Result: {passed}/{total} checks passed")
    if failed == 0:
        print("  [PASS] All checks passed. Navigator is ready to start.")
    else:
        print(f"  [FAIL] {failed} check(s) failed. Resolve the issues above before starting.")
    print("=" * 40 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = run_health_checks()
    sys.exit(0 if ok else 1)
