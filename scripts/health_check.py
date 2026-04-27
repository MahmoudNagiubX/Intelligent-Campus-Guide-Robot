"""
Navigator - System Health Check Script

Runs startup validation before the robot enters active mode.
Exit code 1 means startup should be blocked.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def emit(message: str = "") -> None:
    sys.stdout.write(message + "\n")


_INSTALL_HINT = "  →  Run: python -m pip install -r requirements.txt"


def check(name: str, fn) -> bool:
    try:
        result = bool(fn())
        status = "[PASS]" if result else "[FAIL]"
        emit(f"  {status}  {name}")
        return result
    except ModuleNotFoundError as exc:
        emit(f"  [FAIL]  {name} -- {exc}")
        emit(_INSTALL_HINT)
        return False
    except Exception as exc:
        emit(f"  [FAIL]  {name} -- {exc}")
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
    emit("\n" + "=" * 40)
    emit("  ino - Pre-Flight Health Check")
    emit("=" * 40 + "\n")

    results: list[bool] = []

    def check_config() -> bool:
        from app.config import get_settings

        cfg = get_settings()
        if cfg.english_only_mode:
            return bool(cfg.deepgram_api_key and cfg.groq_api_key)
        return bool(cfg.deepgram_api_key and cfg.groq_api_key and cfg.elevenlabs_api_key)

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

    def check_elevenlabs_key() -> bool:
        from app.config import get_settings

        cfg = get_settings()
        if cfg.english_only_mode:
            return True
        key = cfg.elevenlabs_api_key
        return isinstance(key, str) and len(key) >= 20

    def check_wake_word_model() -> bool:
        import openwakeword
        from app.config import get_settings
        from app.wakeword.detector import WakeWordDetector

        detector = WakeWordDetector(mock=True)
        model_ref = detector._wake_word_model_ref
        if WakeWordDetector._is_model_path(model_ref):
            return Path(model_ref).exists()
        WakeWordDetector._validate_builtin_model_name(
            model_ref,
            set(openwakeword.MODELS.keys()),
            get_settings().wake_word,
        )
        return True

    def check_mic_accessible() -> bool:
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            return pa.get_device_count() > 0
        finally:
            pa.terminate()

    def check_speaker_accessible() -> bool:
        import sounddevice as sd

        devices = sd.query_devices()
        output_devices = [device for device in devices if device["max_output_channels"] > 0]
        return len(output_devices) > 0

    def check_prompts() -> bool:
        _require_non_empty_file(Path("prompts/campus_answer_prompt_en.txt"))
        _require_non_empty_file(Path("prompts/campus_answer_prompt_ar.txt"))
        _require_non_empty_file(Path("prompts/social_prompt.txt"))
        _require_non_empty_file(Path("prompts/router_prompt.txt"))
        _require_non_empty_file(Path("prompts/ecu_answer_prompt_en.txt"))
        _require_non_empty_file(Path("prompts/general_campus_prompt_en.txt"))
        _require_non_empty_file(Path("prompts/ecu_answer_prompt_ar.txt"))
        _require_non_empty_file(Path("prompts/general_campus_prompt_ar.txt"))
        return True

    def check_pyaudio() -> bool:
        import pyaudio  # noqa: F401

        return True

    def check_edge_tts() -> bool:
        import edge_tts  # noqa: F401

        return True

    def check_paho_mqtt() -> bool:
        import paho.mqtt.client  # noqa: F401

        return True

    results.append(check("Config loads with required API keys for selected language mode", check_config))
    results.append(check("SQLite database opens and schema exists", check_sqlite))
    results.append(check("English CSV directory exists and contains CSV files", check_csv_english_dir))
    results.append(check("Arabic CSV directory exists and contains CSV files", check_csv_arabic_dir))
    results.append(check("Internet reachable (DNS test)", check_internet))
    results.append(check("Deepgram API key format looks valid", check_deepgram_key_format))
    results.append(check("ElevenLabs API key set or skipped in English-only mode", check_elevenlabs_key))
    results.append(check("Groq API key format looks valid", check_groq_key_format))
    results.append(check("Required prompt files exist and are non-empty", check_prompts))
    results.append(check("Wake word model accessible", check_wake_word_model))
    results.append(check("Microphone device accessible", check_mic_accessible))
    results.append(check("Speaker/output device accessible", check_speaker_accessible))
    results.append(check("PyAudio installed (mic capture)", check_pyaudio))
    results.append(check("edge-tts installed (TTS output)", check_edge_tts))
    results.append(check("paho-mqtt installed (MQTT state publisher)", check_paho_mqtt))

    passed = sum(1 for item in results if item)
    failed = sum(1 for item in results if not item)
    total = len(results)

    emit("\n" + "=" * 40)
    emit(f"  Result: {passed}/{total} checks passed")
    if failed == 0:
        emit("  [PASS] All checks passed. ino is ready to start.")
    else:
        emit(f"  [FAIL] {failed} check(s) failed. Resolve the issues above before starting.")
    emit("=" * 40 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = run_health_checks()
    sys.exit(0 if ok else 1)
