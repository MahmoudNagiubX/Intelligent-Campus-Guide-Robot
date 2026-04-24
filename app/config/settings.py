"""
Navigator - Configuration Settings
Loads environment variables into a typed Settings object.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Navigator runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    # Deepgram (English specialist)
    deepgram_api_key: str = Field(default="", description="Deepgram Nova-3 API key")
    deepgram_model: str = Field(default="nova-3", description="Deepgram model ID")
    deepgram_language: str = Field(
        default="en",
        description='Deepgram language. Use "en" in dual STT mode; "multi" is only for standalone Deepgram.',
    )
    deepgram_keyterm_prompting_enabled: bool = Field(
        default=False,
        description="Opt in to sending Nova-3 keyterm hints to Deepgram.",
    )
    deepgram_endpointing_ms: int = Field(
        default=500,
        description="Deepgram endpointing silence duration in ms. 300 cuts too early.",
    )
    deepgram_utterance_end_ms: int = Field(
        default=1500,
        description="Deepgram utterance_end_ms — max wait before forced flush.",
    )
    english_only_mode: bool = Field(
        default=False,
        description=(
            "Disable Arabic STT, Arabic retrieval, and Arabic TTS paths entirely. "
            "Deepgram runs language='en' only. No ElevenLabs STT connection attempted."
        ),
    )

    # ElevenLabs (Arabic specialist)
    elevenlabs_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_STT_API_KEY",
            "ELEVENLABS_KEY",
            "ELEVEN_API_KEY",
        ),
        description="ElevenLabs Scribe v2 API key",
    )
    elevenlabs_model: str = Field(default="scribe_v2", description="ElevenLabs STT model ID")
    elevenlabs_keyterms_max: int = Field(default=100, description="Maximum ElevenLabs keyword hints")
    elevenlabs_partial_debug: bool = Field(default=False, description="Log ElevenLabs partial transcripts")

    # Dual STT arbitration
    stt_race_window_ms: int = Field(default=600, description="Normal dual-STT race window in milliseconds")
    stt_arabic_hold_max_ms: int = Field(default=2000, description="Max hold for phonetic Arabic Deepgram results")
    stt_confidence_margin: float = Field(default=0.10, description="Confidence margin used in STT tie-breaking")

    # Groq
    groq_api_key: str = Field(default="", description="Groq LLM API key")
    groq_model: str = Field(default="llama-3.1-8b-instant", description="Groq model ID")
    groq_timeout_sec: float = Field(default=8.0, gt=0.0, description="Groq request timeout in seconds")
    groq_max_retries: int = Field(default=3, gt=0, description="Maximum Groq retry attempts")
    groq_retry_backoff: float = Field(default=1.0, ge=0.0, description="Groq retry backoff in seconds")

    # TTS
    edge_tts_voice_en: str = Field(
        default="en-US-ChristopherNeural",
        description="English TTS voice. ChristopherNeural is the most stable male voice.",
    )
    edge_tts_voice_ar: str = Field(default="ar-EG-SalmaNeural", description="Arabic Egyptian TTS voice")
    edge_tts_rate: str = Field(
        default="-10%",
        description="Speech rate for English TTS. Separate from Arabic rate.",
    )
    edge_tts_rate_ar: str = Field(
        default="-3%",
        description="Speech rate for Arabic TTS. Arabic sounds more natural faster than English.",
    )
    elevenlabs_tts_arabic_enabled: bool = Field(
        default=False,
        description="Use ElevenLabs TTS for Arabic output instead of edge-tts.",
    )
    elevenlabs_tts_voice_ar: str = Field(default="", description="ElevenLabs voice ID for Arabic TTS")
    tts_fallback_phrase: str = Field(
        default="Sorry, I had a small audio issue. Please ask me again.",
        description="Spoken phrase when all TTS retries fail. Pre-synthesized at startup.",
    )

    @field_validator("edge_tts_voice_en", mode="before")
    @classmethod
    def _upgrade_deprecated_english_voice(cls, value: str | None) -> str:
        if not value or str(value).strip() in {"en-US-JennyNeural", "en-US-RyanNeural"}:
            return "en-US-ChristopherNeural"
        return str(value).strip()

    # Audio
    mic_sample_rate: int = Field(default=16000, description="Microphone sample rate in Hz")
    mic_channels: int = Field(default=1, description="Microphone channel count")
    mic_frame_ms: int = Field(default=30, description="Audio frame duration in milliseconds")
    mic_device_index: int = Field(default=-1, description="PyAudio input device index; -1 means default")
    speaker_device_index: int | None = Field(default=None, description="Output device index; None means default")
    playback_echo_suppress_ms: float = Field(
        default=1200.0,
        description="Milliseconds after playback ends to ignore mic input for echo suppression.",
    )

    # Wake word
    wake_word: str = Field(default="hey ino", description="Wake word phrase")
    wake_word_model: str = Field(default="", description="openWakeWord model ID or local model path")
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Wake confidence threshold")
    wake_cooldown_sec: float = Field(default=2.0, ge=0.0, description="Wake trigger cooldown in seconds")

    # VAD
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD speech threshold")
    vad_silence_ms: int = Field(
        default=900,
        gt=0,
        validation_alias=AliasChoices("VAD_END_OF_UTTERANCE_MS", "VAD_SILENCE_MS"),
        description="Silence before end of utterance",
    )

    # Session
    session_timeout_sec: int = Field(
        default=25,
        gt=0,
        description="Seconds of silence before session closes and returns to wake-word mode.",
    )
    session_timeout_speaking_paused: bool = Field(
        default=True,
        description="Pause the inactivity timer while the robot is speaking.",
    )
    max_session_turns: int = Field(default=10, gt=0, description="Maximum turns per active session")

    # Language detection
    default_language: str = Field(default="en", description="Fallback language code")
    lang_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Provider language threshold")

    # Storage
    sqlite_db_path: str = Field(default="data/sqlite/navigator.db", description="SQLite database path")
    csv_english_dir: str = Field(default="data/csv_english", description="English CSV data directory")
    csv_arabic_dir: str = Field(default="data/csv_arabic", description="Arabic CSV data directory")
    csv_data_dir: str = Field(default="data/csv", description="Deprecated legacy CSV directory")
    ecu_knowledge_path: str = Field(default="data/ecu_knowledge.json", description="Local ECU knowledge cache path")

    # Router/action/logging
    router_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    action_bridge_host: str = Field(default="localhost", description="Hardware action bridge host")
    action_bridge_port: int = Field(default=9090, gt=0, description="Hardware action bridge port")
    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def mic_frame_size(self) -> int:
        """Deprecated alias: frame size in samples derived from MIC_FRAME_MS."""
        return max(1, int(self.mic_sample_rate * (self.mic_frame_ms / 1000.0)))

    @property
    def vad_end_of_utterance_ms(self) -> int:
        """Deprecated alias for vad_silence_ms."""
        return self.vad_silence_ms

    @property
    def groq_timeout(self) -> float:
        """Deprecated alias for groq_timeout_sec."""
        return self.groq_timeout_sec

    @groq_timeout.setter
    def groq_timeout(self, value: float) -> None:
        self.groq_timeout_sec = value

    @property
    def deepgram_language_ar(self) -> str:
        """Deprecated alias retained for existing ElevenLabs Arabic language hints."""
        return "ar-EG"

    @property
    def action_bridge_url(self) -> str:
        """Full base URL for the action bridge HTTP endpoint."""
        return f"http://{self.action_bridge_host}:{self.action_bridge_port}"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_deepgram_key(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def has_elevenlabs_key(self) -> bool:
        return bool(self.elevenlabs_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance."""
    return Settings()
