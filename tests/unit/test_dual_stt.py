import time
from types import SimpleNamespace

import pytest

from app.stt.dual_stt_client import DualSTTClient, _looks_like_phonetic_arabic
from app.utils.contracts import TranscriptEvent


class FakeDeepgramStreamingClient:
    instances = []

    def __init__(
        self,
        *,
        on_partial=None,
        on_final=None,
        language="en",
        keyterms=None,
        mock=False,
        session_id=None,
    ):
        self.language = language
        self.keyterms = keyterms
        self.mock = mock
        self._session_id = session_id
        self.connected = False
        self.disconnected = False
        self.sent_audio = []
        self.finalized = False
        self.reset_count = 0
        self.injected = []
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_connected = None
        self._on_error = None
        FakeDeepgramStreamingClient.instances.append(self)

    @property
    def session_id(self):
        return self._session_id

    def set_callbacks(self, on_partial=None, on_final=None, on_connected=None, on_error=None):
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_connected = on_connected
        self._on_error = on_error

    def connect(self):
        self.connected = True
        if self._on_connected:
            self._on_connected()

    def disconnect(self):
        self.disconnected = True

    def send_audio(self, frame):
        self.sent_audio.append(frame)

    def finalize_turn(self):
        self.finalized = True

    def set_session_id(self, session_id):
        self._session_id = session_id

    def reset_turn(self):
        self.reset_count += 1

    def inject_mock_transcript(
        self,
        text,
        *,
        is_final=True,
        language=None,
        language_confidence=None,
    ):
        self.injected.append(
            {
                "text": text,
                "is_final": is_final,
                "language": language,
                "language_confidence": language_confidence,
            }
        )
        if self._on_final:
            self._on_final(
                TranscriptEvent(
                    text=text,
                    is_final=is_final,
                    language=language or self.language,
                    language_confidence=language_confidence,
                    confidence=0.95,
                    session_id=self._session_id,
                )
            )

    def fire_final(self, text, confidence=0.95, language=None):
        if self._on_final:
            self._on_final(
                TranscriptEvent(
                    text=text,
                    is_final=True,
                    language=language or self.language,
                    confidence=confidence,
                    session_id=self._session_id,
                )
            )

    def fire_error(self, reason="error", message="server rejected WebSocket connection: HTTP 403"):
        if self._on_error:
            self._on_error(reason, message)


class FakeElevenLabsArabicClient(FakeDeepgramStreamingClient):
    instances = []

    def __init__(
        self,
        *,
        on_partial=None,
        on_final=None,
        language="ar-EG",
        keyterms=None,
        mock=False,
        session_id=None,
    ):
        super().__init__(
            on_partial=on_partial,
            on_final=on_final,
            language=language,
            keyterms=keyterms,
            mock=mock,
            session_id=session_id,
        )
        FakeElevenLabsArabicClient.instances.append(self)


@pytest.fixture()
def dual_client(monkeypatch):
    monkeypatch.setattr("app.stt.dual_stt_client._elevenlabs_permanently_disabled", False)
    FakeDeepgramStreamingClient.instances = []
    FakeElevenLabsArabicClient.instances = []
    monkeypatch.setattr(
        "app.stt.dual_stt_client.DeepgramStreamingClient",
        FakeDeepgramStreamingClient,
    )
    monkeypatch.setattr(
        "app.stt.dual_stt_client.ElevenLabsArabicClient",
        FakeElevenLabsArabicClient,
    )
    client = DualSTTClient(mock=True, session_id="session-1")
    en_client = next(c for c in FakeDeepgramStreamingClient.instances if c.language == "en")
    ar_client = FakeElevenLabsArabicClient.instances[0]
    return client, en_client, ar_client


def wait_for(condition, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    assert condition()


def test_english_winner_selected_first(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the robotics lab", confidence=0.92)

    wait_for(lambda: len(finals) == 1)
    wait_for(lambda: ar_client.disconnected)
    assert finals[0].language == "en"
    assert finals[0].text == "where is the robotics lab"
    assert not en_client.disconnected


def test_phonetic_english_waits_for_arabic_whisper(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("wayn almaktab", confidence=0.92)
    time.sleep(0.3)
    assert finals == []

    ar_client.fire_final("فين معمل الروبوتات", confidence=0.91)

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "ar-EG"
    assert finals[0].text == "فين معمل الروبوتات"


def test_phonetic_english_commits_if_arabic_never_arrives(dual_client, monkeypatch):
    client, en_client, _ar_client = dual_client
    finals = []
    monkeypatch.setattr(
        "app.stt.dual_stt_client.get_settings",
        lambda: SimpleNamespace(stt_arabic_hold_max_ms=50),
    )
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("wayn almaktab", confidence=0.92)

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "en"
    assert finals[0].text == "wayn almaktab"


def test_real_english_with_stop_word_does_not_wait(dual_client):
    client, en_client, _ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where lab", confidence=0.92)

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "en"
    assert finals[0].text == "where lab"


def test_arabic_winner_selected_first(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    ar_client.fire_final("فين معمل الروبوتات", confidence=0.91)

    wait_for(lambda: len(finals) == 1)
    wait_for(lambda: en_client.disconnected)
    assert finals[0].language == "ar-EG"
    assert finals[0].text == "فين معمل الروبوتات"
    assert not ar_client.disconnected


def test_arabic_wins_on_arabic_text_when_tied(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("fin mamal robotat", confidence=0.9)
    ar_client.fire_final("فين معمل الروبوتات", confidence=0.9)

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "ar-EG"
    assert finals[0].text == "فين معمل الروبوتات"


def test_english_wins_on_english_text_when_tied(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the robotics lab", confidence=0.9)
    ar_client.fire_final("where is the robotics lab", confidence=0.9)

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "en"
    assert finals[0].text == "where is the robotics lab"


def test_low_confidence_result_does_not_win(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("uncertain words", confidence=0.49)
    time.sleep(0.25)

    assert finals == []
    assert not en_client.disconnected
    assert not ar_client.disconnected


def test_audio_routed_to_both_clients(dual_client):
    client, en_client, ar_client = dual_client

    client.send_audio(b"audio-frame")

    assert en_client.sent_audio == [b"audio-frame"]
    assert ar_client.sent_audio == [b"audio-frame"]


def test_connect_opens_both_clients(dual_client):
    client, en_client, ar_client = dual_client

    client.connect()

    assert en_client.connected
    assert ar_client.connected


def test_connect_is_idempotent_and_does_not_reset_winner_state(dual_client):
    client, en_client, _ar_client = dual_client
    client.connect()
    client._winner_language = "en"
    client._winner_forwarded = True

    client.connect()

    assert en_client.connected
    assert client._winner_language == "en"
    assert client._winner_forwarded is True


def test_arabic_connected_callback_resets_403_count(dual_client):
    client, en_client, ar_client = dual_client

    ar_client._on_connected = None
    client._el_consecutive_failures = 1
    client.connect()

    assert en_client.connected
    assert ar_client.connected
    assert client._el_consecutive_failures == 1

    client._handle_arabic_connected()

    assert client._el_consecutive_failures == 0


def test_elevenlabs_403_switches_deepgram_to_multi_after_threshold(dual_client):
    client, en_client, ar_client = dual_client

    ar_client.fire_error(message="server rejected WebSocket connection: HTTP 403")
    assert client._deepgram_fallback_to_multi is False
    ar_client.fire_error(message="server rejected WebSocket connection: HTTP 403")

    assert client._deepgram_fallback_to_multi is True
    assert ar_client._permanently_disabled is True
    assert en_client.disconnected is True
    assert client._deepgram_client.language == "multi"
    assert client._deepgram_client.connected is True


def test_deepgram_multi_preserves_arabic_language_after_fallback(dual_client):
    client, _en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)
    ar_client.fire_error(message="server rejected WebSocket connection: HTTP 403")
    ar_client.fire_error(message="server rejected WebSocket connection: HTTP 403")

    client._deepgram_client.fire_final("فين معمل الروبوتات", confidence=0.95, language="ar-EG")

    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "ar-EG"
    assert finals[0].source == "deepgram"


def test_disconnect_closes_both_clients(dual_client):
    client, en_client, ar_client = dual_client

    client.disconnect()

    assert en_client.disconnected
    assert ar_client.disconnected


def test_session_level_dedup_same_text(dual_client):
    client, en_client, _ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the lab", confidence=0.95)
    wait_for(lambda: len(finals) == 1)
    client._winner_forwarded = False
    en_client.fire_final("where is the lab", confidence=0.95)
    time.sleep(0.05)

    assert len(finals) == 1


def test_session_level_dedup_similar_text(dual_client):
    client, en_client, _ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the lab", confidence=0.95)
    wait_for(lambda: len(finals) == 1)
    client._winner_forwarded = False
    en_client.fire_final("where is the lab.", confidence=0.95)
    time.sleep(0.05)

    assert len(finals) == 1


def test_reset_turn_clears_dedup(dual_client):
    client, en_client, _ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the lab", confidence=0.95)
    wait_for(lambda: len(finals) == 1)
    client.reset_turn()
    en_client.fire_final("where is the lab", confidence=0.95)
    wait_for(lambda: len(finals) == 2)

    assert [event.text for event in finals] == ["where is the lab", "where is the lab"]


def test_mock_inject_en(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    client.inject_mock_transcript("hello", language="en")

    assert en_client.injected[0]["text"] == "hello"
    assert ar_client.injected == []
    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "en"


def test_mock_inject_ar(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    client.inject_mock_transcript("فين", language="ar-EG")

    assert ar_client.injected[0]["text"] == "فين"
    assert en_client.injected == []
    wait_for(lambda: len(finals) == 1)
    assert finals[0].language == "ar-EG"


def test_reset_clears_winner_state(dual_client):
    client, en_client, ar_client = dual_client
    finals = []
    client.set_callbacks(on_final=finals.append)

    en_client.fire_final("where is the lab", confidence=0.95)
    wait_for(lambda: len(finals) == 1)
    client.reset_turn()
    ar_client.disconnected = False
    en_client.disconnected = False

    ar_client.fire_final("فين المعمل", confidence=0.95)

    wait_for(lambda: len(finals) == 2)
    assert finals[0].language == "en"
    assert finals[1].language == "ar-EG"
    assert en_client.reset_count == 1
    assert ar_client.reset_count == 1


def test_looks_like_phonetic_arabic_heuristic():
    assert _looks_like_phonetic_arabic("wayn almaktab")
    assert _looks_like_phonetic_arabic("fin maktab robot")
    assert not _looks_like_phonetic_arabic("where is the lab")
    assert not _looks_like_phonetic_arabic("please take me to the robotics lab")
