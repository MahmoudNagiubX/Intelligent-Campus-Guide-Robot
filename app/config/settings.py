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
        env_ignore_empty=True,
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
    speaker_device_index: int | None = Field(
        default=None,
        description="Playback output device index for sounddevice. None = system default output",
    )

    # ── Wake Word ─────────────────────────────────────────────────────────────
    wake_word: str = Field(default="hey jarvis", description="Wake phrase used for live wake-word activation")
    wake_word_model: str = Field(
        default="",
        description="openWakeWord model ID or .onnx/.tflite path. Blank derives from WAKE_WORD.",
    )
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Activation confidence threshold")

    # ── Session ───────────────────────────────────────────────────────────────
    session_timeout_sec: int = Field(default=10, gt=0, description="Seconds of silence before session closes")

    # ── Storage ───────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="./data/sqlite/navigator.db", description="Path to the SQLite database file")
    csv_data_dir: str = Field(default="./data/csv", description="Directory containing staff-editable CSV files")

    # ── TTS Voices ────────────────────────────────────────────────────────────
    edge_tts_voice_ar: str = Field(default="ar-EG-SalmaNeural", description="Arabic Egyptian TTS voice")
    edge_tts_voice_en: str = Field(default="en-US-JennyNeural", description="English TTS voice")
    edge_tts_rate: str = Field(default="-10%", description="Speech rate passed to edge-tts, for example -10% or +5%")
    default_language: str = Field(default="en", description="Fallback language code when detection is uncertain")

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model ID used for router and generator calls",
    )
    groq_timeout: float = Field(
        default=8.0, gt=0.0,
        description="Groq API request timeout in seconds",
    )
    groq_max_retries: int = Field(
        default=3, gt=0,
        description="Maximum retry attempts for Groq API calls before giving up",
    )
    groq_retry_backoff: float = Field(
        default=1.0, ge=0.0,
        description="Base delay in seconds for exponential backoff between retries",
    )

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
