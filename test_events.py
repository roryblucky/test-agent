import asyncio
from app.services.events import EventEmitter, EventType, StreamEvent

async def run_test():
    emitter = EventEmitter()
    
    async def produce():
        await emitter.emit_step_start("test")
        await emitter.emit_done({"foo": "bar"}) # This calls close()
        
    async def consume():
        events = []
        async for e in emitter:
            events.append(e)
            print("Received:", e.strip())
        print("Consumer finished normally with count:", len(events))
        
    await asyncio.gather(produce(), consume())

asyncio.run(run_test())
