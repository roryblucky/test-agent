"""Unit tests for streaming event helpers."""

import json

import pytest

from app.services.events import EventEmitter


@pytest.mark.asyncio
async def test_event_emitter_supports_phase7_events() -> None:
    """Phase 7 events serialize through the existing SSE protocol."""
    emitter = EventEmitter()

    await emitter.emit_progress("working", {"step": "planner"})
    await emitter.emit_tool_observation({"tool_name": "search", "status": "success"})
    await emitter.emit_answer_delta("approved answer")
    await emitter.close()

    events = []
    async for line in emitter:
        payload = line.removeprefix("data: ").strip()
        events.append(json.loads(payload))

    assert events == [
        {
            "type": "progress",
            "data": {"message": "working", "data": {"step": "planner"}},
        },
        {
            "type": "tool_observation",
            "data": {"tool_name": "search", "status": "success"},
        },
        {"type": "answer_delta", "data": "approved answer"},
    ]
