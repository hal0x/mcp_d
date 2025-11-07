"""Unified event bus implementations (basic and optimized) in one place.

Both AsyncEventBus and AsyncEventBusOptimized share helpers from
`services.event_bus_common` to avoid duplication.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

from events.models import Event, ErrorOccurred

from .event_bus_common import LRUCache, NoopAwaitable, derive_event_id

LOGGER = logging.getLogger("event_bus")
DLQ_LOGGER = logging.getLogger("event_bus.dlq")

BUS_QUEUE_SIZE = int(os.getenv("BUS_QUEUE_SIZE", "100"))
BUS_WORKERS_PER_TOPIC = int(os.getenv("BUS_WORKERS_PER_TOPIC", "1"))
BUS_MAX_RETRIES = int(os.getenv("BUS_MAX_RETRIES", "3"))
BUS_RETRY_BASE_DELAY = float(os.getenv("BUS_RETRY_BASE_DELAY", "0.1"))
BUS_DEDUP_CACHE_SIZE = int(os.getenv("BUS_DEDUP_CACHE_SIZE", "1024"))


E = TypeVar("E", bound=Event)
Subscriber = Callable[[E], Awaitable[None]]


class AsyncEventBus(Generic[E]):
    """Asynchronous pub/sub event bus (baseline, no priorities)."""

    def __init__(
        self,
        workers_per_topic: int = BUS_WORKERS_PER_TOPIC,
        queue_size: int = BUS_QUEUE_SIZE,
        max_retries: int = BUS_MAX_RETRIES,
        retry_base_delay: float = BUS_RETRY_BASE_DELAY,
        dedup_cache_size: int = BUS_DEDUP_CACHE_SIZE,
        dead_letter_topic: str | None = "errors",
    ) -> None:
        self.workers_per_topic = workers_per_topic
        self.queue_size = queue_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.dedup_cache = LRUCache(dedup_cache_size)
        self.dead_letter_topic = dead_letter_topic if dead_letter_topic else None

        self.queues: Dict[str, asyncio.Queue[E | None]] = {}
        self.subscribers: Dict[str, List[Subscriber[E]]] = {}
        self.workers: Dict[str, List[asyncio.Task[None]]] = {}
        self._topics_started: set[str] = set()
        self._shutdown = asyncio.Event()

    def subscribe(self, topic: str, handler: Subscriber[E]) -> NoopAwaitable:
        """Register a subscriber for a topic. Returns an awaitable placeholder."""
        self._ensure_topic(topic)
        self.subscribers[topic].append(handler)
        try:
            asyncio.get_running_loop()
            if hasattr(self, "_topics_started"):
                self._ensure_workers_started(topic)  # type: ignore[attr-defined]
        except RuntimeError:
            pass
        return NoopAwaitable()

    async def publish(self, topic: str, event: E) -> None:
        ev_id = derive_event_id(topic, event)
        if ev_id in self.dedup_cache:
            LOGGER.debug("duplicate_event", extra={"id": ev_id, "topic": topic})
            return
        self.dedup_cache.add(ev_id)
        queue = self._ensure_topic(topic)
        self._ensure_workers_started(topic)

        # Inline dispatch when workers not yet running or payload is simple
        if not self.workers.get(topic) or not hasattr(event, "id"):
            for handler in list(self.subscribers.get(topic, [])):
                try:
                    await self._dispatch_with_retry(topic, handler, event)
                except Exception as exc:  # pragma: no cover
                    await self._record_failure(
                        topic=topic,
                        handler=handler,
                        event=event,
                        exc=exc,
                        attempt=1,
                        inline=True,
                    )
            return

        await queue.put(event)
        LOGGER.debug(
            "event_published",
            extra={"topic": topic, "id": ev_id, "depth": queue.qsize()},
        )

    def _ensure_topic(self, topic: str) -> asyncio.Queue[E | None]:
        if topic not in self.queues:
            queue: asyncio.Queue[E | None] = asyncio.Queue(maxsize=self.queue_size)
            self.queues[topic] = queue
            self.subscribers[topic] = []
            self.workers[topic] = []
        return self.queues[topic]

    def _ensure_workers_started(self, topic: str) -> None:
        if topic in self._topics_started:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.workers[topic] = [
            asyncio.create_task(self._worker(topic))
            for _ in range(self.workers_per_topic)
        ]
        self._topics_started.add(topic)

    async def _worker(self, topic: str) -> None:
        queue = self.queues[topic]
        while not self._shutdown.is_set():
            try:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    queue.task_done()
                    break
                for handler in list(self.subscribers.get(topic, [])):
                    try:
                        await self._dispatch_with_retry(topic, handler, event)
                    except Exception as e:  # pragma: no cover
                        LOGGER.error(
                            "worker_failed", extra={"topic": topic, "error": str(e)}
                        )
                queue.task_done()
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("worker_loop_error", exc_info=exc)
        LOGGER.debug("worker_stopped", extra={"topic": topic})

    async def _dispatch_with_retry(
        self, topic: str, handler: Subscriber[E], event: E
    ) -> None:
        ev_id = derive_event_id(topic, event)
        delay = self.retry_base_delay
        for attempt in range(self.max_retries + 1):
            try:
                result = handler(event)
                import inspect as _inspect

                if _inspect.isawaitable(result):
                    await result
                LOGGER.debug(
                    "event_handled",
                    extra={
                        "topic": topic,
                        "id": ev_id,
                        "handler": getattr(handler, "__name__", str(handler)),
                    },
                )
                return
            except Exception as exc:  # pragma: no cover
                if attempt >= self.max_retries:
                    await self._record_failure(
                        topic=topic,
                        handler=handler,
                        event=event,
                        exc=exc,
                        attempt=attempt + 1,
                        inline=False,
                    )
                    return
                await asyncio.sleep(delay)
                delay *= 2

    async def _record_failure(
        self,
        *,
        topic: str,
        handler: Subscriber[E] | None,
        event: E,
        exc: Exception,
        attempt: int,
        inline: bool,
        handler_name: str | None = None,
    ) -> None:
        if handler_name is None:
            handler_name = getattr(handler, "__name__", str(handler)) if handler else "unknown"
        event_info = {
            "topic": topic,
            "id": getattr(event, "id", derive_event_id(topic, event)),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "event_type": type(event).__name__ if event else "unknown",
            "handler": handler_name,
            "attempts": attempt,
        }

        if hasattr(event, "__dict__"):
            payload = {}
            for key, value in event.__dict__.items():
                if isinstance(value, str) and len(value) > 1000:
                    payload[key] = value[:1000] + "..."
                elif isinstance(value, (dict, list)) and len(str(value)) > 2000:
                    payload[key] = str(value)[:2000] + "..."
                else:
                    payload[key] = value
            event_info["payload"] = payload

        log_key = "event_failed_inline" if inline else "event_failed"
        DLQ_LOGGER.error(log_key, extra=event_info)

        await self._emit_dead_letter(
            topic=topic,
            handler_name=handler_name,
            event=event,
            exc=exc,
            attempts=attempt,
        )

    async def _emit_dead_letter(
        self,
        *,
        topic: str,
        handler_name: str,
        event: E,
        exc: Exception,
        attempts: int,
    ) -> None:
        if not self.dead_letter_topic or topic == self.dead_letter_topic:
            return

        context = {
            "source_topic": topic,
            "handler": handler_name,
            "attempts": attempts,
        }
        try:
            context["event_id"] = getattr(event, "id")
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            context["event_type"] = type(event).__name__
        except Exception:  # pragma: no cover - best effort
            pass

        error_event = ErrorOccurred(
            origin=f"event_bus:{topic}",
            error=str(exc),
            context=context,
        )

        try:
            await self.publish(self.dead_letter_topic, error_event)
        except Exception as publish_exc:  # pragma: no cover - safety
            DLQ_LOGGER.error(
                "dead_letter_publish_failed",
                extra={
                    "topic": topic,
                    "dead_letter_topic": self.dead_letter_topic,
                    "error": str(publish_exc),
                },
            )

    async def graceful_shutdown(self) -> None:
        self._shutdown.set()
        for topic, queue in self.queues.items():
            for _ in range(len(self.workers.get(topic, []))):
                await queue.put(None)
        all_tasks = [task for ws in self.workers.values() for task in ws]
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0
                )
            except asyncio.TimeoutError:  # pragma: no cover
                LOGGER.warning("graceful_shutdown_timeout")

    async def join(self, topic: Optional[str] = None, timeout: float = 30.0) -> None:
        if topic:
            queue = self.queues.get(topic)
            if queue:
                self._ensure_workers_started(topic)
                await asyncio.wait_for(queue.join(), timeout=timeout)
            return
        for t in list(self.queues.keys()):
            self._ensure_workers_started(t)
        join_tasks = [q.join() for q in self.queues.values()]
        if join_tasks:
            await asyncio.wait_for(asyncio.gather(*join_tasks), timeout=timeout)

    def get_queue_status(self) -> Dict[str, dict]:  # pragma: no cover - diagnostics
        status: Dict[str, dict] = {}
        for topic, queue in self.queues.items():
            status[topic] = {
                "queue_size": queue.qsize(),
                "worker_count": len(self.workers.get(topic, [])),
                "subscriber_count": len(self.subscribers.get(topic, [])),
                "workers_active": sum(
                    1 for task in self.workers.get(topic, []) if not task.done()
                ),
            }
        return status


# Optimized bus with priorities and metrics
class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PrioritizedEvent:
    event: Event
    priority: EventPriority
    timestamp: float = field(default_factory=time.time)

    def __lt__(
        self, other: "PrioritizedEvent"
    ) -> bool:  # pragma: no cover - relies on ordering
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


BUS_PRIORITY_QUEUES = int(os.getenv("BUS_PRIORITY_QUEUES", "3"))


class AsyncEventBusOptimized(Generic[E]):
    def __init__(
        self,
        workers_per_topic: int = int(os.getenv("BUS_WORKERS_PER_TOPIC", "3")),
        queue_size: int = int(os.getenv("BUS_QUEUE_SIZE", "200")),
        max_retries: int = BUS_MAX_RETRIES,
        retry_base_delay: float = BUS_RETRY_BASE_DELAY,
        dedup_cache_size: int = int(os.getenv("BUS_DEDUP_CACHE_SIZE", "2048")),
        priority_queues: int = BUS_PRIORITY_QUEUES,
    ) -> None:
        self.workers_per_topic = workers_per_topic
        self.queue_size = queue_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.dedup_cache = LRUCache(dedup_cache_size)
        self.priority_queues = priority_queues

        self.priority_queues_dict: Dict[
            str, List[asyncio.PriorityQueue[PrioritizedEvent | None]]
        ] = {}
        self.subscribers: Dict[str, List[Subscriber[E]]] = {}
        self.workers: Dict[str, List[asyncio.Task[None]]] = {}
        self._shutdown = asyncio.Event()
        self._topic_lock = asyncio.Lock()

        self._event_counters: Dict[str, int] = {}
        self._processing_times: Dict[str, List[float]] = {}
        self._queue_depths: Dict[str, List[int]] = {}

    async def subscribe(self, topic: str, handler: Subscriber[E]) -> None:
        await self._ensure_topic(topic)
        self.subscribers[topic].append(handler)
        LOGGER.info(f"Subscriber registered for topic: {topic}")

    async def publish(
        self, topic: str, event: E, priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        if event.id in self.dedup_cache:
            LOGGER.debug("duplicate_event", extra={"id": event.id, "topic": topic})
            return

        self.dedup_cache.add(event.id)
        prioritized_event = PrioritizedEvent(event, priority)
        queues = await self._ensure_topic(topic)
        queue_idx = min(priority.value, len(queues) - 1)
        try:
            queues[queue_idx].put_nowait(prioritized_event)
        except asyncio.QueueFull:
            LOGGER.warning(
                "queue_full",
                extra={
                    "topic": topic,
                    "priority": priority.name,
                    "queue_idx": queue_idx,
                },
            )
            for i in range(queue_idx - 1, -1, -1):
                try:
                    queues[i].put_nowait(prioritized_event)
                    LOGGER.info(f"Event moved to lower priority queue {i}")
                    break
                except asyncio.QueueFull:
                    continue
            else:
                LOGGER.error(f"All queues full for topic {topic}, event dropped")

    async def _ensure_topic(
        self, topic: str
    ) -> List[asyncio.PriorityQueue[PrioritizedEvent | None]]:
        if topic in self.priority_queues_dict:
            return self.priority_queues_dict[topic]
        async with self._topic_lock:
            if topic not in self.priority_queues_dict:
                queues = [
                    asyncio.PriorityQueue(
                        maxsize=self.queue_size // self.priority_queues
                    )
                    for _ in range(self.priority_queues)
                ]
                self.priority_queues_dict[topic] = queues
                self.subscribers[topic] = []
                self.workers[topic] = [
                    asyncio.create_task(self._worker(topic, queue_idx))
                    for queue_idx in range(self.priority_queues)
                    for _ in range(self.workers_per_topic)
                ]
                self._event_counters[topic] = 0
                self._processing_times[topic] = []
                self._queue_depths[topic] = []
        return self.priority_queues_dict[topic]

    async def _worker(self, topic: str, queue_idx: int) -> None:
        queue = self.priority_queues_dict[topic][queue_idx]
        worker_id = f"{topic}_{queue_idx}_{id(queue)}"
        LOGGER.info(f"Worker {worker_id} started")
        while not self._shutdown.is_set():
            try:
                try:
                    prioritized_event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if prioritized_event is None:
                    queue.task_done()
                    break
                start_time = time.time()
                await self._dispatch_with_retry(
                    topic, prioritized_event.event, worker_id
                )
                processing_time = time.time() - start_time
                self._processing_times[topic].append(processing_time)
                if len(self._processing_times[topic]) > 100:
                    self._processing_times[topic] = self._processing_times[topic][-100:]
                # increment processed counter
                self._event_counters[topic] = self._event_counters.get(topic, 0) + 1
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover
                LOGGER.exception(f"Worker {worker_id} error: {exc}")
                if prioritized_event is not None:
                    queue.task_done()
        LOGGER.info(f"Worker {worker_id} stopped")

    async def _dispatch_with_retry(
        self, topic: str, event: Event, worker_id: str
    ) -> None:
        delay = self.retry_base_delay
        for attempt in range(self.max_retries + 1):
            try:
                handlers = list(self.subscribers.get(topic, []))
                if not handlers:
                    LOGGER.warning(f"No handlers for topic {topic}")
                    return

                exceptions: list[tuple[Subscriber[E], Exception]] = []
                awaitables: list[asyncio.Future[Any] | asyncio.Task[Any]] = []
                async_handlers: list[Subscriber[E]] = []

                for handler in handlers:
                    try:
                        result = handler(event)
                    except Exception as sync_exc:  # pragma: no cover - sync failure
                        exceptions.append((handler, sync_exc))
                        continue

                    import inspect as _inspect

                    if _inspect.isawaitable(result):
                        async_handlers.append(handler)
                        awaitables.append(asyncio.ensure_future(result))
                    else:
                        # synchronous handler completed successfully
                        continue

                if awaitables:
                    async_results = await asyncio.gather(
                        *awaitables, return_exceptions=True
                    )
                    for handler_obj, res in zip(async_handlers, async_results):
                        if isinstance(res, Exception):
                            exceptions.append((handler_obj, res))

                if exceptions:
                    if attempt >= self.max_retries:
                        for handler_obj, handler_exc in exceptions:
                            await self._record_failure(
                                topic=topic,
                                handler=handler_obj,
                                event=event,
                                exc=handler_exc,
                                attempt=attempt + 1,
                                inline=False,
                            )
                        return
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                LOGGER.debug(
                    "event_handled",
                    extra={
                        "topic": topic,
                        "id": event.id,
                        "worker_id": worker_id,
                        "handlers_count": len(handlers),
                    },
                )
                return
            except Exception as exc:  # pragma: no cover
                if attempt >= self.max_retries:
                    await self._record_failure(
                        topic=topic,
                        handler=None,
                        event=event,
                        exc=exc,
                        attempt=attempt + 1,
                        inline=False,
                        handler_name="worker_dispatch",
                    )
                    return
                await asyncio.sleep(delay)
                delay *= 2

    async def _record_failure(
        self,
        *,
        topic: str,
        handler: Subscriber[E] | None,
        event: Event,
        exc: Exception,
        attempt: int,
        inline: bool,
        handler_name: str | None = None,
    ) -> None:
        if handler_name is None:
            handler_name = getattr(handler, "__name__", str(handler)) if handler else "unknown"
        event_info = {
            "topic": topic,
            "id": getattr(event, "id", derive_event_id(topic, event)),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "event_type": type(event).__name__ if event else "unknown",
            "handler": handler_name,
            "attempts": attempt,
        }

        if hasattr(event, "__dict__"):
            payload = {}
            for key, value in event.__dict__.items():
                if isinstance(value, str) and len(value) > 1000:
                    payload[key] = value[:1000] + "..."
                elif isinstance(value, (dict, list)) and len(str(value)) > 2000:
                    payload[key] = str(value)[:2000] + "..."
                else:
                    payload[key] = value
            event_info["payload"] = payload

        log_key = "event_failed_inline" if inline else "event_failed"
        DLQ_LOGGER.error(log_key, extra=event_info)

        await self._emit_dead_letter(
            topic=topic,
            handler_name=handler_name,
            event=event,
            exc=exc,
            attempts=attempt,
        )

    async def _emit_dead_letter(
        self,
        *,
        topic: str,
        handler_name: str,
        event: Event,
        exc: Exception,
        attempts: int,
    ) -> None:
        if not self.dead_letter_topic or topic == self.dead_letter_topic:
            return

        context = {
            "source_topic": topic,
            "handler": handler_name,
            "attempts": attempts,
        }

        try:
            context["event_id"] = getattr(event, "id")
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            context["event_type"] = type(event).__name__
        except Exception:  # pragma: no cover - best effort
            pass

        error_event = ErrorOccurred(
            origin=f"event_bus:{topic}",
            error=str(exc),
            context=context,
        )

        try:
            await self.publish(self.dead_letter_topic, error_event)
        except Exception as publish_exc:  # pragma: no cover - safety
            DLQ_LOGGER.error(
                "dead_letter_publish_failed",
                extra={
                    "topic": topic,
                    "dead_letter_topic": self.dead_letter_topic,
                    "error": str(publish_exc),
                },
            )

    async def graceful_shutdown(self) -> None:
        self._shutdown.set()
        for topic, queues in self.priority_queues_dict.items():
            for q in queues:
                await q.put(None)
        all_tasks = [task for ws in self.workers.values() for task in ws]
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0
                )
            except asyncio.TimeoutError:  # pragma: no cover
                LOGGER.warning("graceful_shutdown_timeout")

    def get_metrics(
        self, topic: str
    ) -> Dict[str, int | float | None]:  # pragma: no cover - diagnostics
        return {
            "events_processed": self._event_counters.get(topic, 0),
            "subscribers_count": len(self.subscribers.get(topic, [])),
            "workers_count": len(self.workers.get(topic, [])),
            "avg_processing_time": (
                (
                    sum(self._processing_times.get(topic, []))
                    / len(self._processing_times.get(topic, []))
                )
                if self._processing_times.get(topic)
                else None
            ),
        }

    def get_queue_status(self) -> Dict[str, dict]:  # pragma: no cover - diagnostics
        status: Dict[str, dict] = {}
        for topic, queues in self.priority_queues_dict.items():
            topic_status: dict = {}
            for i, q in enumerate(queues):
                topic_status[f"priority_{i}"] = {
                    "size": q.qsize(),
                    "maxsize": q.maxsize,
                    "full": q.full(),
                }
            status[topic] = topic_status
        return status
