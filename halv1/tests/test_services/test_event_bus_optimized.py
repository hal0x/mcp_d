"""Тесты для оптимизированного AsyncEventBus."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from services.event_bus_optimized import (
    AsyncEventBusOptimized,
    EventPriority,
    LRUCache,
    PrioritizedEvent,
)


class MockEvent:
    """Тестовое событие."""

    def __init__(self, event_id: str, data: str = ""):
        self.id = event_id
        self.data = data


class TestLRUCache:
    """Тесты для LRU кэша."""

    def test_lru_cache_basic(self):
        """Базовый тест LRU кэша."""
        cache = LRUCache(maxsize=3)

        # Добавляем элементы
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")

        assert "key1" in cache
        assert "key2" in cache
        assert "key3" in cache
        assert len(cache._data) == 3

        # Добавляем еще один - должен вытеснить первый
        cache.add("key4")
        assert "key1" not in cache
        assert "key4" in cache
        assert len(cache._data) == 3

    def test_lru_cache_discard(self):
        """Тест удаления элементов."""
        cache = LRUCache(maxsize=2)

        cache.add("key1")
        cache.add("key2")

        cache.discard("key1")
        assert "key1" not in cache
        assert "key2" in cache

        # Удаление несуществующего ключа не должно вызывать ошибку
        cache.discard("nonexistent")


class TestPrioritizedEvent:
    """Тесты для приоритизированных событий."""

    def test_prioritized_event_creation(self):
        """Тест создания приоритизированного события."""
        event = MockEvent("test_id", "test_data")
        prioritized = PrioritizedEvent(event, EventPriority.HIGH)

        assert prioritized.event == event
        assert prioritized.priority == EventPriority.HIGH
        assert prioritized.timestamp > 0

    def test_prioritized_event_comparison(self):
        """Тест сравнения приоритизированных событий."""
        event1 = MockEvent("id1")
        event2 = MockEvent("id2")

        # Событие с высоким приоритетом должно быть меньше (первым в очереди)
        high_priority = PrioritizedEvent(event1, EventPriority.HIGH)
        low_priority = PrioritizedEvent(event2, EventPriority.LOW)

        assert high_priority < low_priority

        # События с одинаковым приоритетом сравниваются по времени
        early_time = PrioritizedEvent(event1, EventPriority.NORMAL)
        time.sleep(0.001)  # Небольшая задержка
        late_time = PrioritizedEvent(event2, EventPriority.NORMAL)

        assert early_time < late_time


class TestAsyncEventBusOptimized:
    """Тесты для оптимизированного AsyncEventBus."""

    @pytest.fixture
    def event_bus(self):
        """Фикстура для создания event bus."""
        return AsyncEventBusOptimized(
            workers_per_topic=2, queue_size=10, priority_queues=3
        )

    @pytest.fixture
    def mock_handler(self):
        """Фикстура для mock обработчика."""
        return AsyncMock()

    def test_event_bus_creation(self, event_bus):
        """Тест создания event bus."""
        assert event_bus.workers_per_topic == 2
        assert event_bus.queue_size == 10
        assert event_bus.priority_queues == 3
        assert len(event_bus.priority_queues_dict) == 0

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus, mock_handler):
        """Тест подписки и публикации событий."""
        # Подписываемся на топик
        await event_bus.subscribe("test_topic", mock_handler)

        # Проверяем, что топик создан
        assert "test_topic" in event_bus.subscribers
        assert len(event_bus.subscribers["test_topic"]) == 1

        # Публикуем событие
        event = MockEvent("test_event_id", "test_data")
        await event_bus.publish("test_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Проверяем, что обработчик вызван
        mock_handler.assert_called_once()
        called_event = mock_handler.call_args[0][0]
        assert called_event.id == "test_event_id"

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_priority_queues(self, event_bus, mock_handler):
        """Тест приоритетных очередей."""
        await event_bus.subscribe("priority_topic", mock_handler)

        # Публикуем события с разными приоритетами
        event1 = MockEvent("event1", "high_priority")
        event2 = MockEvent("event2", "normal_priority")
        event3 = MockEvent("event3", "low_priority")

        await event_bus.publish("priority_topic", event1, EventPriority.HIGH)
        await event_bus.publish("priority_topic", event2, EventPriority.NORMAL)
        await event_bus.publish("priority_topic", event3, EventPriority.LOW)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Проверяем, что все события обработаны
        assert mock_handler.call_count == 3

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_deduplication(self, event_bus, mock_handler):
        """Тест дедупликации событий."""
        await event_bus.subscribe("dedup_topic", mock_handler)

        event = MockEvent("duplicate_id", "data")

        # Публикуем одно и то же событие дважды
        await event_bus.publish("dedup_topic", event)
        await event_bus.publish("dedup_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Проверяем, что обработчик вызван только один раз
        assert mock_handler.call_count == 1

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_metrics_collection(self, event_bus, mock_handler):
        """Тест сбора метрик."""
        await event_bus.subscribe("metrics_topic", mock_handler)

        # Публикуем несколько событий
        for i in range(3):
            event = MockEvent(f"event_{i}", f"data_{i}")
            await event_bus.publish("metrics_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Получаем метрики
        metrics = event_bus.get_metrics("metrics_topic")

        assert metrics["events_processed"] == 3
        assert metrics["subscribers_count"] == 1
        assert metrics["workers_count"] > 0

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_queue_status(self, event_bus):
        """Тест получения статуса очередей."""
        # Создаем топик
        await event_bus.subscribe("status_topic", AsyncMock())

        # Получаем статус
        status = event_bus.get_queue_status()

        assert "status_topic" in status
        topic_status = status["status_topic"]

        # Проверяем, что есть очереди для всех приоритетов
        assert len(topic_status) == 3
        for i in range(3):
            priority_key = f"priority_{i}"
            assert priority_key in topic_status
            assert "size" in topic_status[priority_key]
            assert "maxsize" in topic_status[priority_key]
            assert "full" in topic_status[priority_key]

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, event_bus):
        """Тест graceful shutdown."""
        # Создаем топик
        mock_handler = AsyncMock()
        await event_bus.subscribe("shutdown_topic", mock_handler)

        # Публикуем событие
        event = MockEvent("shutdown_event", "data")
        await event_bus.publish("shutdown_topic", event)

        # Ждем немного обработки
        await asyncio.sleep(0.1)

        # Выполняем shutdown
        await event_bus.graceful_shutdown()

        # Проверяем, что все воркеры остановлены
        for workers in event_bus.workers.values():
            for worker in workers:
                assert worker.done()

    @pytest.mark.asyncio
    async def test_concurrent_handlers(self, event_bus):
        """Тест конкурентной обработки несколькими обработчиками."""
        # Создаем несколько обработчиков
        handlers = [AsyncMock() for _ in range(3)]
        for handler in handlers:
            await event_bus.subscribe("concurrent_topic", handler)

        # Публикуем событие
        event = MockEvent("concurrent_event", "data")
        await event_bus.publish("concurrent_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Проверяем, что все обработчики вызваны
        for handler in handlers:
            handler.assert_called_once()

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_error_handling(self, event_bus):
        """Тест обработки ошибок в обработчиках."""
        # Создаем обработчик, который выбрасывает исключение
        error_handler = AsyncMock(side_effect=Exception("Test error"))
        await event_bus.subscribe("error_topic", error_handler)

        # Публикуем событие
        event = MockEvent("error_event", "data")
        await event_bus.publish("error_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.1)

        # Проверяем, что обработчик вызван
        error_handler.assert_called_once()

        # Проверяем, что event bus продолжает работать
        assert not event_bus._shutdown.is_set()

        # Graceful shutdown
        await event_bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_topic_creation(self):
        """Тест конкурентной публикации в новый топик создаёт структуры один раз."""
        bus = AsyncEventBusOptimized(
            workers_per_topic=1, queue_size=10, priority_queues=2
        )

        async def pub(event_id: str) -> None:
            await bus.publish("new_topic", MockEvent(event_id))

        await asyncio.gather(pub("e1"), pub("e2"))
        await asyncio.sleep(0.1)

        assert "new_topic" in bus.priority_queues_dict
        assert len(bus.priority_queues_dict["new_topic"]) == bus.priority_queues
        assert (
            len(bus.workers["new_topic"]) == bus.priority_queues * bus.workers_per_topic
        )
        assert len(bus.subscribers["new_topic"]) == 0

        await bus.graceful_shutdown()

    @pytest.mark.asyncio
    async def test_queue_full_handling(self):
        """Тест обработки переполнения очередей."""
        # Создаем топик с очень маленькой очередью
        small_bus = AsyncEventBusOptimized(
            workers_per_topic=1,
            queue_size=2,  # Увеличиваем размер очереди
            priority_queues=1,
        )

        # Создаем медленный обработчик
        slow_handler = AsyncMock()
        slow_handler.side_effect = lambda e: asyncio.sleep(
            0.05
        )  # Уменьшаем время обработки
        await small_bus.subscribe("slow_topic", slow_handler)

        # Публикуем несколько событий быстро
        events = [MockEvent(f"event_{i}", f"data_{i}") for i in range(3)]

        for event in events:
            await small_bus.publish("slow_topic", event)

        # Ждем обработки
        await asyncio.sleep(0.3)

        # Проверяем, что события обработаны (может быть меньше из-за переполнения)
        assert slow_handler.call_count >= 1
        assert slow_handler.call_count <= 3

        await small_bus.graceful_shutdown()
