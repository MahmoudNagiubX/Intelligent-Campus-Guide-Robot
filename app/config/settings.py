"""
Navigator - Configuration Settings
Loads all environment variables into a typed, validated Settings object.
Every module must read config from here — never from os.environ directly.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Navigator system.
    All values are loaded from the .env file or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Cloud API Keys ────────────────────────────────────────────────────────
    deepgram_api_key: str = Field(default="", description="Deepgram Nova-3 API key")
    groq_api_key: str = Field(default="", description="Groq LLM API key")

    # ── Audio ─────────────────────────────────────────────────────────────────
    mic_sample_rate: int = Field(default=16000, description="Microphone sample rate in Hz")
    mic_frame_size: int = Field(default=512, description="Audio frame size in samples")
    mic_channels: int = Field(default=1, description="Mono audio channel count")
    mic_device_index: int | None = Field(default=None, description="PyAudio device index. None = system default")

    # ── Wake Word ─────────────────────────────────────────────────────────────
    wake_word: str = Field(default="hey navigator", description="Wake phrase (English only for MVP)")
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Activation confidence threshold")

    # ── Session ───────────────────────────────────────────────────────────────
    session_timeout_sec: int = Field(default=15, gt=0, description="Seconds of silence before session closes")

    # ── Storage ───────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="./data/sqlite/navigator.db", description="Path to the SQLite database file")
    csv_data_dir: str = Field(default="./data/csv", description="Directory containing staff-editable CSV files")

    # ── TTS Voices ────────────────────────────────────────────────────────────
    edge_tts_voice_ar: str = Field(default="ar-EG-SalmaNeural", description="Arabic Egyptian TTS voice")
    edge_tts_voice_en: str = Field(default="en-US-JennyNeural", description="English TTS voice")
    default_language: str = Field(default="en", description="Fallback language code when detection is uncertain")

    # ── Router ────────────────────────────────────────────────────────────────
    router_confidence_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Minimum router confidence before asking for clarification"
    )

    # ── Action Bridge ─────────────────────────────────────────────────────────
    action_bridge_host: str = Field(default="localhost", description="Host for the hardware action bridge")
    action_bridge_port: int = Field(default=9090, gt=0, description="Port for the hardware action bridge")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def action_bridge_url(self) -> str:
        """Full base URL for the action bridge HTTP endpoint."""
        return f"http://{self.action_bridge_host}:{self.action_bridge_port}"

    @property
    def has_deepgram_key(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.
    Loaded once on first call, reused on every subsequent call.
    All modules should call get_settings() instead of constructing Settings directly.
    """
    return Settings()
