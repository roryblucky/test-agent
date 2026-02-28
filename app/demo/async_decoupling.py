"""Message Bus / Async Decoupling Demo.

This module demonstrates how to decouple the HTTP request from the slow
Orchestrator LLM execution using FastAPI BackgroundTasks and a Message Bus.
This is heavily inspired by enterprise architectures where the Web server
must return quickly and the LLM workloads run asynchronously.

Flow:
1. Client POSTs a query to `/chat`.
2. Server generates a `task_id`, schedules the Orchestrator in a BackgroundTask,
   and immediately returns `{"task_id": "..."}` (202 Accepted).
3. Client connects to `/chat/stream/{task_id}` via Server-Sent Events (SSE).
4. The background Orchestrator runs, sending events to an Event Bus (e.g., Redis PubSub).
5. The SSE endpoint listens to the Event Bus and forwards events to the client.

NOTE: This is a demonstration skeleton. In production, `InMemoryMessageBus`
should be replaced with Redis PubSub or Kafka.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# 1. Message Bus (Demo In-Memory Implementation)
# ---------------------------------------------------------------------------


class InMemoryMessageBus:
    """A simplistic in-memory pub-sub bus for routing events by task_id."""

    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, task_id: str, message: dict):
        """Publish an event to all subscribers of a task_id."""
        for queue in self._queues[task_id]:
            await queue.put(message)

    async def subscribe(self, task_id: str) -> asyncio.Queue:
        """Subscribe to events for a specific task_id."""
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[task_id].append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        if queue in self._queues[task_id]:
            self._queues[task_id].remove(queue)


bus = InMemoryMessageBus()

# ---------------------------------------------------------------------------
# 2. Mock Orchestrator that uses the Message Bus
# ---------------------------------------------------------------------------


class AsyncEventEmitter:
    """An EventEmitter that publishes to the message bus instead of direct yield."""

    def __init__(self, task_id: str, bus: InMemoryMessageBus):
        self.task_id = task_id
        self.bus = bus

    async def emit_step_start(self, step_name: str, payload: dict | None = None):
        msg = {"event": "step_start", "step": step_name, "data": payload or {}}
        await self.bus.publish(self.task_id, msg)

    async def emit_token(self, chunk: str):
        msg = {"event": "token", "data": {"chunk": chunk}}
        await self.bus.publish(self.task_id, msg)

    async def emit_step_completed(self, step_name: str, payload: dict | None = None):
        msg = {"event": "step_completed", "step": step_name, "data": payload or {}}
        await self.bus.publish(self.task_id, msg)


async def run_orchestrator_background(task_id: str, query: str):
    """The heavy background job representing the dynamic agent loop."""
    emitter = AsyncEventEmitter(task_id, bus)

    # Simulate pre-step
    await emitter.emit_step_start("moderation")
    await asyncio.sleep(0.5)
    await emitter.emit_step_completed("moderation", {"is_flagged": False})

    # Simulate router agent LLM generation
    await emitter.emit_step_start("router_agent_core")
    await asyncio.sleep(1.0)

    words = ["Here", " is", " the", " asynchronously", " generated", " answer."]
    for word in words:
        await emitter.emit_token(word)
        await asyncio.sleep(0.2)

    await emitter.emit_step_completed("router_agent_core", {"model": "pro"})

    # Send a final 'done' signal so the SSE stream can close gracefully
    await bus.publish(task_id, {"event": "done"})


# ---------------------------------------------------------------------------
# 3. FastAPI Endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/demo/async", tags=["Async Demo"])


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    task_id: str


@router.post("/chat", response_model=ChatResponse, status_code=202)
async def submit_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Submit a query and return a task_id immediately."""
    task_id = str(uuid.uuid4())

    # Offload the heavy agentic loop to the background
    background_tasks.add_task(run_orchestrator_background, task_id, req.query)

    return ChatResponse(task_id=task_id)


@router.get("/chat/stream/{task_id}")
async def stream_chat_events(task_id: str):
    """Clients connect here via SSE to listen for their task's events."""
    queue = await bus.subscribe(task_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                msg = await queue.get()
                if msg.get("event") == "done":
                    break
                yield json.dumps(msg)
        finally:
            bus.unsubscribe(task_id, queue)

    return EventSourceResponse(event_generator())
