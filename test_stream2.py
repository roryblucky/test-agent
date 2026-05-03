import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
import os
os.environ["OPENAI_API_KEY"] = "fake"

class CL(BaseModel):
    x: int

agent = Agent('test', output_type=str | CL)

async def main():
    async with agent.run_stream('hello') as stream:
        print("Methods:", [m for m in dir(stream) if not m.startswith('_')])

asyncio.run(main())
