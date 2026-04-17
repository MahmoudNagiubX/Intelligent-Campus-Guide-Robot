"""
Navigator - System Health Check Script
Phase 10, Step 10.4

Runs all pre-flight checks before the robot enters active mode.
Can be run standalone or called from main.py during startup.

Usage:
    python scripts/health_check.py

Exit codes:
    0 - all checks passed
    1 - one or more checks failed (robot should not start)
"""

from __future__ import annotations

import sys
import socket
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(name: str, fn) -> bool:
    """Run one check. Print result. Return True if passed."""
    try:
        result = fn()
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}  {name}")
        return bool(result)
    except Exception as exc:
        print(f"  [FAIL]  {name} -- {exc}")
        return False


def run_health_checks() -> bool:
    """Run all checks. Returns True if all pass."""
    print("\n" + "=" * 40)
    print("  Navigator - Pre-Flight Health Check")
    print("=" * 40 + "\n")

    results = []

    # ── Config ────────────────────────────────────────────────────────────────
    def check_config():
        from app.config import get_settings
        cfg = get_settings()
        return bool(cfg.deepgram_api_key and cfg.groq_api_key)

    results.append(check("Config loads (DEEPGRAM_API_KEY, GROQ_API_KEY present)", check_config))

    # ── SQLite ────────────────────────────────────────────────────────────────
    def check_sqlite():
        from app.config import get_settings
        from app.storage.db import get_db
        conn = get_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        return len(tables) >= 4  # departments, locations, staff, aliases at minimum

    results.append(check("SQLite database opens and schema exists", check_sqlite))

    # ── CSV sync ──────────────────────────────────────────────────────────────
    def check_csv_dir():
        from app.config import get_settings
        cfg = get_settings()
        csv_dir = Path(cfg.csv_data_dir)
        csvs = list(csv_dir.glob("*.csv"))
        return len(csvs) >= 3  # at least departments, locations, staff

    results.append(check("CSV data directory has at least 3 files", check_csv_dir))

    # ── Internet connectivity ──────────────────────────────────────────────────
    def check_internet():
        socket.setdefaulttimeout(5)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True

    results.append(check("Internet reachable (DNS test)", check_internet))

    # ── Deepgram API key format ────────────────────────────────────────────────
    def check_deepgram_key_format():
        from app.config import get_settings
        key = get_settings().deepgram_api_key
        return isinstance(key, str) and len(key) >= 20

    results.append(check("Deepgram API key format looks valid", check_deepgram_key_format))

    # ── Groq API key format ────────────────────────────────────────────────────
    def check_groq_key_format():
        from app.config import get_settings
        key = get_settings().groq_api_key
        return isinstance(key, str) and len(key) >= 20

    results.append(check("Groq API key format looks valid", check_groq_key_format))

    # ── Prompts ────────────────────────────────────────────────────────────────
    def check_prompts():
        for name in ("campus_answer_prompt.txt", "social_prompt.txt", "router_prompt.txt"):
            path = Path("prompts") / name
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty: {name}")
        return True

    results.append(check("All required prompt files exist and are non-empty", check_prompts))

    # ── Optional: PyAudio ─────────────────────────────────────────────────────
    def check_pyaudio():
        import pyaudio  # noqa
        return True

    results.append(check("PyAudio installed (mic capture)", check_pyaudio))

    # ── Optional: edge-tts ────────────────────────────────────────────────────
    # -- Optional: edge-tts ----------------------------------------------------
    def check_edge_tts():
        import edge_tts  # noqa
        return True

    results.append(check("edge-tts installed (TTS output)", check_edge_tts))

    # -- Summary -----------------------------------------------------------
    passed  = sum(1 for r in results if r)
    failed  = sum(1 for r in results if not r)
    total   = len(results)

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
