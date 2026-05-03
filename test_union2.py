import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

class CL(BaseModel):
    x: int

agent = Agent(TestModel(call_tools=['CL']), output_type=str | CL)

async def main():
    async with agent.run_stream('hello') as stream:
        try:
            async for out in stream.stream_output(debounce_by=None):
                print("out:", repr(out))
        except Exception as e:
            print("error:", e)

asyncio.run(main())
