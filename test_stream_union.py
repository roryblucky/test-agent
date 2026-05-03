import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

class CL(BaseModel):
    x: int

agent = Agent(TestModel(custom_result_text="hello world"), output_type=str | CL)

async def main():
    async with agent.run_stream('hello') as stream:
        try:
            async for text in stream.stream_text(delta=True):
                print(text)
        except Exception as e:
            print("stream_text error:", e)

        try:
            async for out in stream.stream_output(debounce_by=None):
                print(out)
        except Exception as e:
            print("stream_output error:", e)

asyncio.run(main())
