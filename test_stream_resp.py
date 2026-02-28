import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/test")
async def test_endpoint():
    async def event_generator():
        for i in range(3):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

client = TestClient(app)

def run():
    response = client.get("/test")
    print(response.status_code)
    print(response.content)

run()
