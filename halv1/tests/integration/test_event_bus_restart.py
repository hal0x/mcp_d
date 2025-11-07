"""Integration test for worker restart on handler deadlock."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from events.models import MessageReceived
from services.event_bus import AsyncEventBus


@pytest.mark.asyncio
async def test_worker_restarts_after_handler_timeout() -> None:
    bus: AsyncEventBus[MessageReceived] = AsyncEventBus(
        workers_per_topic=1, handler_timeout=0.1, max_retries=0
    )
    processed = asyncio.Event()
    calls = 0

    async def handler(event: MessageReceived) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)  # exceed timeout to simulate deadlock
        else:
            processed.set()

    bus.subscribe("messages", handler)
    first = MessageReceived(
        id="1", timestamp=datetime.now(UTC), chat_id=1, message_id=1, text="a"
    )
    second = MessageReceived(
        id="2", timestamp=datetime.now(UTC), chat_id=1, message_id=2, text="b"
    )
    await bus.publish("messages", first)
    await bus.publish("messages", second)

    await asyncio.wait_for(processed.wait(), timeout=5)
    await bus.graceful_shutdown()

    assert calls >= 2
