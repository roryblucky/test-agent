import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

class TestOutput(BaseModel):
    answer: str
    number: int

agent = Agent("test", output_type=TestOutput)

async def main():
    try:
        async with agent.run_stream("What is 2+2? Answer verbosely.") as stream:
            prev_answer = ""
            async for partial in stream.stream_structured():
                if partial.answer and len(partial.answer) > len(prev_answer):
                    chunk = partial.answer[len(prev_answer):]
                    print(f"CHUNK: '{chunk}'")
                    prev_answer = partial.answer
            
            final = await stream.get_output()
            print("FINAL:", final)
    except Exception as e:
        print("ERROR:", e)

asyncio.run(main())
