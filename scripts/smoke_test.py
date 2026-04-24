"""
Run a no-microphone Navigator smoke test.
Exit 0 means all mock utterances passed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.storage import bootstrap_schema, close_db
from app.storage.sync_csv import sync_all_csvs
from app.config import get_settings
from app.utils.contracts import SessionState


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


class _SmokeRouterGroq:
    def complete_json(self, *args, **kwargs) -> str:
        text = kwargs.get("user_message", "")
        lowered = text.lower()
        is_ar = any("\u0600" <= ch <= "\u06FF" for ch in text)
        social = any(word in lowered for word in ("how are you", "مساء", "ازيك", "كيف حالك"))
        nav = any(word in lowered for word in ("take me", "خدني", "وديني", "روحني"))
        if social:
            intent = "Social_Chat"
            target = ""
        else:
            intent = "Navigation_Request" if nav else "Campus_Query"
            target = "Registrar Office" if "تسجيل" in text else "Robotics Lab"
        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.9,
                "language": "ar" if is_ar else "en",
                "needs_retrieval": intent != "Social_Chat",
                "needs_navigation": nav,
                "target_entity": target,
                "target_type": "lab" if target == "Robotics Lab" else "department",
                "normalized_query": text,
                "clarification_needed": False,
                "clarification_question": "",
            },
            ensure_ascii=False,
        )


class _SmokeComposerGroq:
    def complete_text(self, *args, **kwargs) -> str:
        prompt = kwargs.get("system_prompt", "")
        user = kwargs.get("user_message", "")
        if any("\u0600" <= ch <= "\u06FF" for ch in prompt + user):
            return "المكان في المبنى C، أوضة 214."
        if "how are you" in user.lower():
            return "Doing great and ready to guide!"
        return "The place is in Building C, room 214."


async def _run_turn(runtime: NavigatorPipecatRuntime, text: str, language: str) -> tuple[bool, str]:
    before = len(runtime.tracer.events())
    runtime.trigger_wake_word()
    if not await runtime.wait_for_state(SessionState.LISTENING, timeout=2.0):
        return False, "did not enter LISTENING"
    runtime.vad.set_mock_speech(True)
    runtime.process_audio_frame(b"\x01" * 1024)
    await asyncio.sleep(0.05)
    runtime.vad.set_mock_speech(False)
    for _ in range(runtime.vad._end_of_utterance_frames):
        runtime.process_audio_frame(b"\x00" * 1024)
    await asyncio.sleep(0.05)
    runtime.inject_mock_transcript(text, language=language, language_confidence=0.98)
    if not await runtime.wait_for_state(SessionState.IDLE, timeout=4.0):
        return False, "did not return to IDLE"
    responses = [
        event.data.get("text", "")
        for event in runtime.tracer.events()[before:]
        if event.name == "response_generated"
    ]
    if not responses or not responses[-1]:
        return False, "empty response"
    if (
        language.startswith("ar")
        and not get_settings().english_only_mode
        and not any("\u0600" <= ch <= "\u06FF" for ch in responses[-1])
    ):
        return False, "expected Arabic response"
    return True, responses[-1]


async def main() -> int:
    import app.routing.router as router

    router._get_groq = lambda: _SmokeRouterGroq()  # type: ignore[assignment]
    bootstrap_schema()
    sync_all_csvs()

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=_SmokeComposerGroq()),
    )
    utterances = [
        ("where is the robotics lab", "en"),
        ("take me to building c", "en"),
        ("how are you", "en"),
        ("فين معمل الروبوتات", "ar-EG"),
        ("خدني لمكتب التسجيل", "ar-EG"),
        ("مساء الخير", "ar-EG"),
    ]
    await runtime.start()
    passed = 0
    try:
        for text, lang in utterances:
            ok, detail = await _run_turn(runtime, text, lang)
            if ok:
                passed += 1
                emit(f"[PASS] {text} -> {detail}")
            else:
                emit(f"[FAIL] {text} -> {detail}")
            await asyncio.sleep(2.1)
    finally:
        await runtime.shutdown()
        close_db()
    emit(f"{passed}/{len(utterances)} passed")
    return 0 if passed == len(utterances) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
