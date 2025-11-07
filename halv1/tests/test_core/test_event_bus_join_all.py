import asyncio
from datetime import UTC, datetime

from events.models import MessageReceived
from services.event_bus import AsyncEventBus


def _event(event_id: str) -> MessageReceived:
    return MessageReceived(
        id=event_id,
        timestamp=datetime.now(UTC),
        chat_id=1,
        message_id=1,
        text="hi",
    )


def test_join_waits_for_all_topics() -> None:
    async def main() -> tuple[list[MessageReceived], list[MessageReceived]]:
        bus: AsyncEventBus[MessageReceived] = AsyncEventBus(workers_per_topic=1)
        received_a: list[MessageReceived] = []
        received_b: list[MessageReceived] = []

        async def handler_a(event: MessageReceived) -> None:
            received_a.append(event)

        async def handler_b(event: MessageReceived) -> None:
            received_b.append(event)

        bus.subscribe("topic_a", handler_a)
        bus.subscribe("topic_b", handler_b)

        await bus.publish("topic_a", _event("a"))
        await bus.publish("topic_b", _event("b"))

        await bus.join()
        result = (list(received_a), list(received_b))
        await bus.graceful_shutdown()
        return result

    received_a, received_b = asyncio.run(main())
    assert len(received_a) == 1
    assert len(received_b) == 1
