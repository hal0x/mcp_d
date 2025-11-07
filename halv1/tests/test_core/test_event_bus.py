import asyncio
from datetime import UTC, datetime

from events.models import ErrorOccurred, MessageReceived
from services.event_bus import AsyncEventBus


def _event(event_id: str = "1") -> MessageReceived:
    return MessageReceived(
        id=event_id,
        timestamp=datetime.now(UTC),
        chat_id=1,
        message_id=1,
        text="hi",
    )


def test_pub_sub() -> None:
    async def main() -> list[MessageReceived]:
        bus: AsyncEventBus[MessageReceived] = AsyncEventBus(workers_per_topic=1)
        received: list[MessageReceived] = []

        async def handler(event: MessageReceived) -> None:
            received.append(event)

        bus.subscribe("messages", handler)
        event = _event("a")
        await bus.publish("messages", event)
        await bus.join("messages")
        await bus.graceful_shutdown()
        return received

    received = asyncio.run(main())
    assert len(received) == 1
    assert received[0].text == "hi"


def test_retry() -> None:
    async def main() -> int:
        bus: AsyncEventBus[MessageReceived] = AsyncEventBus(
            workers_per_topic=1, max_retries=2, retry_base_delay=0.0
        )
        attempts = 0

        async def handler(event: MessageReceived) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("boom")

        bus.subscribe("messages", handler)
        await bus.publish("messages", _event("b"))
        await bus.join("messages")
        await bus.graceful_shutdown()
        return attempts

    attempts = asyncio.run(main())
    assert attempts == 3


def test_deduplication() -> None:
    async def main() -> list[MessageReceived]:
        bus: AsyncEventBus[MessageReceived] = AsyncEventBus(workers_per_topic=1)
        received: list[MessageReceived] = []

        async def handler(event: MessageReceived) -> None:
            received.append(event)

        bus.subscribe("messages", handler)
        ev = _event("dup")
        await bus.publish("messages", ev)
        await bus.publish("messages", ev)
        await bus.join("messages")
        await bus.graceful_shutdown()
        return received

    received = asyncio.run(main())
    assert len(received) == 1


def test_dead_letter_published_on_handler_failure() -> None:
    async def main() -> list[ErrorOccurred]:
        bus: AsyncEventBus[MessageReceived] = AsyncEventBus(
            workers_per_topic=1, max_retries=0, retry_base_delay=0.0
        )
        errors: list[ErrorOccurred] = []

        async def failing_handler(event: MessageReceived) -> None:
            raise RuntimeError("boom")

        async def error_sink(event: ErrorOccurred) -> None:
            errors.append(event)

        bus.subscribe("errors", error_sink)
        bus.subscribe("messages", failing_handler)

        await bus.publish("messages", _event("err"))
        await bus.join("messages")
        await bus.join("errors")
        await bus.graceful_shutdown()
        return errors

    captured = asyncio.run(main())
    assert captured, "Expected ErrorOccurred to be emitted"
    err = captured[0]
    assert err.origin == "event_bus:messages"
    assert err.context and err.context.get("source_topic") == "messages"
