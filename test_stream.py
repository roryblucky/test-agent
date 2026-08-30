import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class CL(BaseModel):
    x: int


agent = Agent(TestModel(), output_type=str | CL)


async def main() -> None:
    async with agent.run_stream("hello") as stream:
        print("stream type:", type(stream).__name__)


asyncio.run(main())
