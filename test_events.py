import asyncio

from app.services.events import EventEmitter


async def run_test() -> None:
    emitter = EventEmitter()

    async def produce() -> None:
        await emitter.emit_step_start("test")
        await emitter.emit_done({"foo": "bar"})  # This calls close()

    async def consume() -> None:
        events: list[str] = []
        async for e in emitter:
            events.append(e)
            print("Received:", e.strip())
        print("Consumer finished normally with count:", len(events))

    await asyncio.gather(produce(), consume())


asyncio.run(run_test())
