import builtins

from app.stt.whisper_arabic_client import WhisperArabicSTTClient


def test_mock_connect_sets_connected():
    connected = []
    client = WhisperArabicSTTClient(mock=True)
    client.set_callbacks(on_connected=lambda: connected.append(True))

    client.connect()

    assert client._connected is True
    assert connected == [True]


def test_mock_inject_transcript_fires_on_final():
    finals = []
    client = WhisperArabicSTTClient(mock=True, session_id="session-1")
    client.set_callbacks(on_final=finals.append)

    client.inject_mock_transcript("فين معمل الروبوتات", language="ar-EG")

    assert len(finals) == 1
    assert finals[0].text == "فين معمل الروبوتات"
    assert finals[0].language == "ar-EG"
    assert finals[0].session_id == "session-1"
    assert finals[0].source == "whisper_mock"


def test_send_audio_appends_to_buffer():
    client = WhisperArabicSTTClient(mock=True)

    client.send_audio(b"abc")
    client.send_audio(b"def")

    assert bytes(client._audio_buffer) == b"abcdef"


def test_reset_clears_buffer():
    client = WhisperArabicSTTClient(mock=True)
    client.send_audio(b"abc")

    client.reset_turn()

    assert bytes(client._audio_buffer) == b""


def test_finalize_in_mock_does_not_transcribe(monkeypatch):
    called = []
    client = WhisperArabicSTTClient(mock=True)
    client.send_audio(b"abc")
    monkeypatch.setattr(client, "_transcribe", lambda _audio: called.append(True))

    client.finalize_turn()

    assert called == []
    assert bytes(client._audio_buffer) == b"abc"


def test_whisper_not_installed_calls_on_error(monkeypatch):
    errors = []
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("missing faster_whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(WhisperArabicSTTClient, "_model", None)
    monkeypatch.setattr(WhisperArabicSTTClient, "_model_name", None)
    client = WhisperArabicSTTClient(mock=False)
    client.set_callbacks(on_error=lambda reason, message: errors.append((reason, message)))

    client.connect()

    assert client._connected is False
    assert errors
    assert errors[0][0] == "whisper_not_installed"
