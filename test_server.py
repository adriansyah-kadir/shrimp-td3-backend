import asyncio
import websockets
import json


async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print()
            print("State:", data["state"])
            print("Action:", data["action"])
            print()


asyncio.run(test_ws())
