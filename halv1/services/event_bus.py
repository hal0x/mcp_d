from __future__ import annotations

import asyncio
import logging
import os
import time
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
BUS_HANDLER_TIMEOUT = float(os.getenv("BUS_HANDLER_TIMEOUT", "0"))


E = TypeVar("E", bound=Event)

Subscriber = Callable[[E], Awaitable[None]]


class AsyncEventBus(Generic[E]):
    """Asynchronous pub/sub event bus."""

    def __init__(
        self,
        workers_per_topic: int = BUS_WORKERS_PER_TOPIC,
        queue_size: int = BUS_QUEUE_SIZE,
        max_retries: int = BUS_MAX_RETRIES,
        retry_base_delay: float = BUS_RETRY_BASE_DELAY,
        dedup_cache_size: int = BUS_DEDUP_CACHE_SIZE,
        handler_timeout: float = BUS_HANDLER_TIMEOUT,
        dead_letter_topic: str | None = "errors",
    ) -> None:
        self.workers_per_topic = workers_per_topic
        self.queue_size = queue_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.dedup_cache = LRUCache(dedup_cache_size)
        self.handler_timeout = handler_timeout
        self.dead_letter_topic = dead_letter_topic if dead_letter_topic else None

        self.queues: Dict[str, asyncio.Queue[E | None]] = {}
        self.subscribers: Dict[str, List[Subscriber[E]]] = {}
        self.workers: Dict[str, List[asyncio.Task[None]]] = {}
        self._topics_started: set[str] = set()
        self._shutdown = asyncio.Event()
        self._max_queue_depth: Dict[str, int] = {}
        self._max_wait_time: Dict[str, float] = {}

    def subscribe(self, topic: str, handler: Subscriber[E]) -> NoopAwaitable:
        """Register a subscriber for a topic.

        Returns an awaitable placeholder so callers may use either
        ``bus.subscribe(...)`` or ``await bus.subscribe(...)``.
        """
        self._ensure_topic(topic)
        self.subscribers[topic].append(handler)
        # If we're within a running event loop, start workers immediately
        try:
            asyncio.get_running_loop()
            # Start workers lazily but as early as possible
            if hasattr(self, "_topics_started"):
                self._ensure_workers_started(topic)
        except RuntimeError:
            # No running loop at subscription time; workers will start on publish/join
            pass
        return NoopAwaitable()

    async def publish(self, topic: str, event: E) -> None:
        """Publish event to topic with deduplication.

        Accepts any object; if it lacks an ``id`` attribute, a stable textual key
        is derived for deduplication purposes.
        """
        # Derive a key even for non-Event payloads (tests may publish strings)
        ev_id = derive_event_id(topic, event)

        if ev_id in self.dedup_cache:
            LOGGER.debug("duplicate_event", extra={"id": ev_id, "topic": topic})
            return
        self.dedup_cache.add(ev_id)
        queue = self._ensure_topic(topic)
        self._ensure_workers_started(topic)

        # Fallback inline dispatch when workers are not yet running, or for simple
        # payloads without an `id` attribute to improve responsiveness in tests.
        # This avoids deadlocks when subscriptions happen before an asyncio loop
        # is running (common in tests that instantiate the bus during setup).
        if not self.workers.get(topic) or not hasattr(event, "id"):
            for handler in list(self.subscribers.get(topic, [])):
                try:
                    await self._dispatch_with_retry(topic, handler, event)
                except Exception as exc:  # pragma: no cover - safety
                    await self._record_failure(
                        topic=topic,
                        handler=handler,
                        event=event,
                        exc=exc,
                        attempt=1,
                        inline=True,
                    )
            return

        start = time.monotonic()
        await queue.put(event)  # backpressure via bounded queue
        wait_time = time.monotonic() - start
        depth = queue.qsize()
        self._max_wait_time[topic] = max(self._max_wait_time.get(topic, 0.0), wait_time)
        self._max_queue_depth[topic] = max(self._max_queue_depth.get(topic, 0), depth)
        if self.handler_timeout and wait_time > self.handler_timeout:
            LOGGER.warning(
                "publish_wait_exceeded",
                extra={"topic": topic, "wait": wait_time, "depth": depth},
            )
        try:
            ev_log_id = getattr(event, "id", ev_id)
        except Exception:  # pragma: no cover
            ev_log_id = ev_id
        LOGGER.debug(
            "event_published",
            extra={
                "topic": topic,
                "id": ev_log_id,
                "depth": depth,
                "max_depth": self._max_queue_depth[topic],
                "wait": wait_time,
            },
        )

    def _ensure_topic(self, topic: str) -> asyncio.Queue[E | None]:
        if topic not in self.queues:
            queue: asyncio.Queue[E | None] = asyncio.Queue(maxsize=self.queue_size)
            self.queues[topic] = queue
            self.subscribers[topic] = []
            # Start workers lazily when a running loop is available
            self.workers[topic] = []
            self._max_queue_depth[topic] = 0
            self._max_wait_time[topic] = 0.0
        queue = self.queues[topic]
        depth = queue.qsize()
        self._max_queue_depth[topic] = max(self._max_queue_depth.get(topic, 0), depth)
        if depth > self.queue_size * 0.8:
            LOGGER.warning(
                "queue_depth_high",
                extra={"topic": topic, "depth": depth, "maxsize": self.queue_size},
            )
        return queue

    def _ensure_workers_started(self, topic: str) -> None:
        """Start worker tasks for a topic if not yet started and a loop is running."""
        if topic in self._topics_started:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; will attempt later from an async context
            return
        self.workers[topic] = [
            asyncio.create_task(self._worker(topic))
            for _ in range(self.workers_per_topic)
        ]
        self._topics_started.add(topic)

    async def _worker(self, topic: str) -> None:
        queue = self.queues[topic]
        worker_id = f"{topic}_worker_{id(queue)}"
        LOGGER.debug(f"Worker {worker_id} started for topic {topic}")

        while not self._shutdown.is_set():
            event: E | None = None
            try:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    LOGGER.debug(f"Worker {worker_id} received shutdown signal")
                    queue.task_done()
                    break
                LOGGER.debug(
                    f"Worker {worker_id} processing event {getattr(event, 'id', '?')}"
                )
                for handler in list(self.subscribers.get(topic, [])):
                    await self._dispatch_with_retry(topic, handler, event)
                queue.task_done()
                LOGGER.debug(
                    f"Worker {worker_id} completed event {getattr(event, 'id', '?')}"
                )
            except asyncio.CancelledError:  # pragma: no cover - cancellation
                LOGGER.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as exc:  # pragma: no cover - unexpected
                LOGGER.exception(
                    "worker_loop_error", extra={"topic": topic, "worker": worker_id}
                )
                if event is not None:
                    queue.task_done()
                if isinstance(exc, asyncio.TimeoutError):
                    self._restart_worker(topic, asyncio.current_task())
                return
        LOGGER.debug(f"Worker {worker_id} shutting down")

    def _restart_worker(
        self, topic: str, failed_task: Optional[asyncio.Task[None]]
    ) -> None:
        if self._shutdown.is_set():
            return
        workers = self.workers.get(topic, [])
        if failed_task in workers:
            workers.remove(failed_task)
        new_task = asyncio.create_task(self._worker(topic))
        workers.append(new_task)
        LOGGER.warning("worker_restarted", extra={"topic": topic})

    async def _dispatch_with_retry(
        self, topic: str, handler: Subscriber[E], event: E
    ) -> None:
        ev_id = getattr(event, "id", None)
        if ev_id is None:
            try:
                ev_id = f"{topic}:{hash(str(event))}"
            except Exception:  # pragma: no cover
                ev_id = topic
        delay = self.retry_base_delay
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False
        handler_name = getattr(handler, "__name__", repr(handler))
        for attempt in range(self.max_retries + 1):
            try:
                result = handler(event)
                import inspect as _inspect

                if _inspect.isawaitable(result):
                    if has_loop and self.handler_timeout > 0:
                        await asyncio.wait_for(result, timeout=self.handler_timeout)
                    else:
                        await result
                LOGGER.debug(
                    "event_handled",
                    extra={"topic": topic, "id": ev_id, "handler": handler_name},
                )
                return
            except asyncio.TimeoutError:
                LOGGER.error(
                    "handler_timeout",
                    extra={"topic": topic, "id": ev_id, "handler": handler_name},
                )
                raise
            except Exception as exc:  # pragma: no cover - logging
                if attempt >= self.max_retries or not has_loop:
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
        handler_name: str | None = None,
        event: E,
        exc: Exception,
        attempt: int,
        inline: bool,
    ) -> None:
        try:
            ev_id = getattr(event, "id")
        except Exception:  # pragma: no cover - derive safety
            ev_id = derive_event_id(topic, event)

        if handler_name is None:
            handler_name = getattr(handler, "__name__", str(handler)) if handler else "unknown"

        event_info: Dict[str, Any] = {
            "topic": topic,
            "id": ev_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "event_type": type(event).__name__ if event else "unknown",
            "handler": handler_name,
            "attempts": attempt,
        }

        if hasattr(event, "__dict__"):
            payload: Dict[str, Any] = {}
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

        context: Dict[str, Any] = {
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
        """Stop all workers after processing queued events."""
        LOGGER.info("Starting graceful shutdown")
        self._shutdown.set()

        # Отправляем сигнал остановки во все очереди
        for topic, queue in self.queues.items():
            worker_count = len(self.workers.get(topic, []))
            LOGGER.info(
                f"Sending shutdown signal to {worker_count} workers for topic {topic}"
            )
            for _ in range(worker_count):
                await queue.put(None)

        # Ждем завершения всех воркеров с таймаутом
        all_tasks = [task for ws in self.workers.values() for task in ws]
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0
                )
                LOGGER.info("All workers completed successfully")
            except asyncio.TimeoutError:
                LOGGER.warning("Some workers did not complete within timeout")

        LOGGER.info("Graceful shutdown completed")

    def get_queue_status(self) -> Dict[str, Any]:
        """Получает статус всех очередей для диагностики."""
        status: Dict[str, Any] = {}
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

    def get_worker_status(self) -> Dict[str, Any]:
        """Получает статус всех воркеров для диагностики."""
        status: Dict[str, Any] = {}
        for topic, workers in self.workers.items():
            status[topic] = []
            for i, task in enumerate(workers):
                status[topic].append(
                    {
                        "worker_id": f"{topic}_worker_{i}",
                        "done": task.done(),
                        "cancelled": task.cancelled(),
                        "exception": (
                            str(task.exception())
                            if task.done() and task.exception()
                            else None
                        ),
                    }
                )
        return status

    async def join(self, topic: Optional[str] = None, timeout: float = 30.0) -> None:
        """Wait for all queued events to finish with diagnostics and guard rails."""

        async def _wait_queue(name: str, queue: asyncio.Queue[E | None]) -> None:
            self._ensure_workers_started(name)
            start = time.monotonic()
            remaining = timeout
            check_interval = min(timeout, 5.0) if timeout else 5.0
            while True:
                wait_for = check_interval if remaining is None else min(check_interval, remaining)
                try:
                    await asyncio.wait_for(queue.join(), timeout=wait_for)
                    LOGGER.info(
                        "queue_join_completed",
                        extra={
                            "topic": name,
                            "elapsed": time.monotonic() - start,
                            "max_depth": self._max_queue_depth.get(name, 0),
                        },
                    )
                    return
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - start
                    unfinished = getattr(queue, "_unfinished_tasks", None)
                    LOGGER.warning(
                        "queue_join_waiting",
                        extra={
                            "topic": name,
                            "elapsed": elapsed,
                            "pending": queue.qsize(),
                            "unfinished_tasks": unfinished,
                        },
                    )
                    if timeout and elapsed >= timeout:
                        self._log_topic_diagnostic(name)
                        self._force_drain_queue(name)
                        raise
                    if remaining is not None:
                        remaining = max(timeout - elapsed, 0.0)

        if topic:
            queue = self.queues.get(topic)
            if queue:
                try:
                    await _wait_queue(topic, queue)
                except asyncio.TimeoutError:
                    LOGGER.warning(f"Timeout waiting for topic {topic}")
                    return
            else:
                LOGGER.warning(f"Topic {topic} not found")
            return

        # ensure workers are started for all topics before waiting
        for t in list(self.queues.keys()):
            self._ensure_workers_started(t)

        queues = list(self.queues.items())
        if not queues:
            return

        tasks = [asyncio.create_task(_wait_queue(name, q)) for name, q in queues]
        try:
            await asyncio.gather(*tasks)
            LOGGER.info("All topics completed successfully")
        except asyncio.TimeoutError:
            LOGGER.warning("Timeout waiting for all topics")
            self._log_all_topics_diagnostic()
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    def _force_drain_queue(self, topic: str) -> None:
        queue = self.queues.get(topic)
        if not queue:
            return
        drained = 0
        try:
            while True:
                queue.get_nowait()
                queue.task_done()
                drained += 1
        except asyncio.QueueEmpty:
            pass
        unfinished = getattr(queue, "_unfinished_tasks", 0)
        if unfinished:
            for _ in range(unfinished):
                try:
                    queue.task_done()
                except ValueError:
                    break
        LOGGER.error(
            "queue_force_drained",
            extra={
                "topic": topic,
                "drained": drained,
                "remaining": queue.qsize(),
                "unfinished_after": getattr(queue, "_unfinished_tasks", None),
            },
        )

    def _log_topic_diagnostic(self, topic: str) -> None:
        """Логирует диагностику для конкретного топика при таймауте."""
        queue = self.queues.get(topic)
        if queue:
            LOGGER.error(f"Topic {topic} diagnostic:")
            LOGGER.error(f"  Queue size: {queue.qsize()}")
            LOGGER.error(f"  Workers: {len(self.workers.get(topic, []))}")
            LOGGER.error(f"  Subscribers: {len(self.subscribers.get(topic, []))}")

            # Проверяем состояние воркеров
            workers = self.workers.get(topic, [])
            active_workers = sum(1 for task in workers if not task.done())
            LOGGER.error(f"  Active workers: {active_workers}/{len(workers)}")

    def _log_all_topics_diagnostic(self) -> None:
        """Логирует диагностику для всех топиков при таймауте."""
        LOGGER.error("All topics diagnostic:")
        for topic, queue in self.queues.items():
            LOGGER.error(f"  Topic {topic}:")
            LOGGER.error(f"    Queue size: {queue.qsize()}")
            LOGGER.error(f"    Workers: {len(self.workers.get(topic, []))}")
            LOGGER.error(f"    Subscribers: {len(self.subscribers.get(topic, []))}")

            # Проверяем состояние воркеров
            workers = self.workers.get(topic, [])
            active_workers = sum(1 for task in workers if not task.done())
            LOGGER.error(f"    Active workers: {active_workers}/{len(workers)}")
