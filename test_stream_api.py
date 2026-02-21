import asyncio
from pydantic_ai import Agent

agent = Agent('test')

async def main():
    try:
        async with agent.run_stream("hello") as stream:
            result = await stream.get_output()
            print("DIR STREAM:")
            print([m for m in dir(stream) if not m.startswith('_')])
    except Exception as e:
        print("ERROR:", e)

asyncio.run(main())
