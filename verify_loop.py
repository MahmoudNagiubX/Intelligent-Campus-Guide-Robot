import asyncio
from app.utils.contracts import SessionState

async def main():
    from app.pipeline.pipecat_graph import NavigatorPipecatRuntime

    rt = NavigatorPipecatRuntime(mock=True)
    await rt.start()

    # Guarantee 1: English turn -> IDLE
    rt.trigger_wake_word()
    await rt.wait_for_state(SessionState.WAKE_DETECTED, timeout=2.0)
    rt.inject_mock_transcript('where is the robotics lab', language='en')
    await rt.wait_for_state(SessionState.IDLE, timeout=10.0)
    assert rt._session_manager.state == SessionState.IDLE
    print('PASS 1: English turn -> returned to IDLE')

    # Guarantee 2: Arabic turn -> IDLE
    rt.trigger_wake_word()
    await rt.wait_for_state(SessionState.WAKE_DETECTED, timeout=2.0)
    rt.inject_mock_transcript('فين معمل الروبوتات', language='ar')
    await rt.wait_for_state(SessionState.IDLE, timeout=10.0)
    assert rt._session_manager.state == SessionState.IDLE
    print('PASS 2: Arabic turn -> returned to IDLE')

    # Guarantee 3: wait for LISTENING (state moves past WAKE_DETECTED instantly)
    rt.trigger_wake_word()
    ok = await rt.wait_for_state(SessionState.LISTENING, timeout=2.0)
    assert ok, 'FAIL: second wake word did not reach LISTENING'
    print('PASS 3: second wake word accepted, reached LISTENING')

    await rt.shutdown()
    print()
    print('All guarantees verified. Safe to commit.')

asyncio.run(main())
