import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel

class CL(BaseModel):
    x: int

agent = Agent('test', result_type=str | CL)

async def main():
    async with agent.run_stream('hello') as stream:
        print("is text?", stream.is_text if hasattr(stream, 'is_text') else "NO is_text")
        print("is structured?", stream.is_structured if hasattr(stream, 'is_structured') else "NO is_structured")

asyncio.run(main())
