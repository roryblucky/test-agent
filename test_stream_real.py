import asyncio
from app.core.model_registry import ModelRegistry
from app.agents.coordinator import create_coordinator_agent, CoordinatorDeps
from app.api.dependencies import get_tenant_manager

async def test_coordinator_stream():
    registry = ModelRegistry()
    registry.register_google("gemini-1.5-flash-8b", "fast")
    registry.register_google("gemini-1.5-pro", "pro")
    
    agent = create_coordinator_agent(registry)
    deps = CoordinatorDeps(registry=registry, retriever=None, ranker=None)
    
    try:
        async with agent.run_stream("What is 1+1? Write 3 sentences.", deps=deps) as stream:
            prev = ""
            async for partial in stream.stream_structured():
                if partial.answer and len(partial.answer) > len(prev):
                    print("CHUNK:", partial.answer[len(prev):], end="", flush=True)
                    prev = partial.answer
            print()
            output = await stream.get_output()
            print("FINAL:", output)
            print("MESSAGES:", len(stream.new_messages()))
            print("USAGE:", stream.usage())
    except Exception as e:
        print("ERROR:", e)

asyncio.run(test_coordinator_stream())
