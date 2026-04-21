import asyncio, os
os.environ.setdefault('DEEPGRAM_API_KEY', open('.env').read().split('DEEPGRAM_API_KEY=')[1].split()[0])

async def test():
    import websockets, urllib.parse
    url = 'wss://api.deepgram.com/v1/listen?model=nova-3&language=en&encoding=linear16&sample_rate=16000&channels=1&interim_results=true&punctuate=true&smart_format=true'
    key = os.environ['DEEPGRAM_API_KEY']
    try:
        async with websockets.connect(url, additional_headers={'Authorization': f'Token {key}'}) as ws:
            print('CONNECTED - clean options work')
            await ws.close()
    except Exception as e:
        print(f'FAILED: {e}')

asyncio.run(test())
