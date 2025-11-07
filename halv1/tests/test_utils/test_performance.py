"""Тесты для модуля производительности."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from utils.performance import (
    PerformanceMetrics,
    PerformanceProfiler,
    measure_context,
    measure_context_async,
    measure_time,
    measure_time_async,
    profiler,
)


def test_performance_metrics():
    """Тест базовых метрик производительности."""
    metrics = PerformanceMetrics("test_operation")
    
    # Добавляем измерения
    metrics.add_measurement(0.1)
    metrics.add_measurement(0.2)
    metrics.add_measurement(0.3)
    
    assert metrics.total_calls == 3
    assert abs(metrics.total_time - 0.6) < 0.001
    assert metrics.min_time == 0.1
    assert metrics.max_time == 0.3
    assert abs(metrics.avg_time - 0.2) < 0.001
    
    summary = metrics.get_summary()
    assert summary["operation"] == "test_operation"
    assert summary["total_calls"] == 3
    assert summary["avg_time_ms"] == 200.0


def test_performance_profiler():
    """Тест профилировщика производительности."""
    profiler_instance = PerformanceProfiler()
    
    # Добавляем измерения
    profiler_instance.add_measurement("op1", 0.1)
    profiler_instance.add_measurement("op1", 0.2)
    profiler_instance.add_measurement("op2", 0.3)
    
    all_metrics = profiler_instance.get_all_metrics()
    assert len(all_metrics) == 2
    assert "op1" in all_metrics
    assert "op2" in all_metrics
    
    slowest = profiler_instance.get_slowest_operations(2)
    assert len(slowest) == 2
    assert slowest[0]["operation"] == "op2"  # Самый медленный


def test_measure_time_decorator():
    """Тест декоратора для синхронных функций."""
    @measure_time("test_function")
    def test_func():
        time.sleep(0.01)  # Небольшая задержка
        return "result"
    
    result = test_func()
    assert result == "result"
    
    # Проверяем, что метрики были добавлены
    metrics = profiler.get_metrics("test_function")
    assert metrics.total_calls > 0


@pytest.mark.asyncio
async def test_measure_time_async_decorator():
    """Тест декоратора для асинхронных функций."""
    @measure_time_async("test_async_function")
    async def test_async_func():
        await asyncio.sleep(0.01)  # Небольшая задержка
        return "async_result"
    
    result = await test_async_func()
    assert result == "async_result"
    
    # Проверяем, что метрики были добавлены
    metrics = profiler.get_metrics("test_async_function")
    assert metrics.total_calls > 0


def test_measure_context():
    """Тест контекст-менеджера для синхронного кода."""
    with measure_context("test_context"):
        time.sleep(0.01)
    
    # Проверяем, что метрики были добавлены
    metrics = profiler.get_metrics("test_context")
    assert metrics.total_calls > 0


@pytest.mark.asyncio
async def test_measure_context_async():
    """Тест асинхронного контекст-менеджера."""
    async with measure_context_async("test_async_context"):
        await asyncio.sleep(0.01)
    
    # Проверяем, что метрики были добавлены
    metrics = profiler.get_metrics("test_async_context")
    assert metrics.total_calls > 0


def test_profiler_singleton():
    """Тест, что глобальный профилировщик работает как синглтон."""
    # Очищаем предыдущие метрики
    profiler._metrics.clear()
    
    with measure_context("singleton_test"):
        time.sleep(0.01)
    
    # Проверяем, что метрики добавились в глобальный профилировщик
    assert "singleton_test" in profiler._metrics


@pytest.mark.asyncio
async def test_performance_integration():
    """Интеграционный тест производительности."""
    # Очищаем предыдущие метрики
    profiler._metrics.clear()
    
    # Выполняем несколько операций
    @measure_time("sync_op")
    def sync_operation():
        time.sleep(0.01)
        return "sync_done"
    
    @measure_time_async("async_op")
    async def async_operation():
        await asyncio.sleep(0.01)
        return "async_done"
    
    # Синхронная операция
    sync_result = sync_operation()
    assert sync_result == "sync_done"
    
    # Асинхронная операция
    async_result = await async_operation()
    assert async_result == "async_done"
    
    # Контекст-менеджеры
    with measure_context("context_op"):
        time.sleep(0.01)
    
    async with measure_context_async("async_context_op"):
        await asyncio.sleep(0.01)
    
    # Проверяем, что все метрики собраны
    all_metrics = profiler.get_all_metrics()
    expected_operations = {"sync_op", "async_op", "context_op", "async_context_op"}
    assert set(all_metrics.keys()) == expected_operations
    
    # Проверяем, что все операции выполнились
    for op_name in expected_operations:
        metrics = profiler.get_metrics(op_name)
        assert metrics.total_calls == 1
        assert metrics.total_time > 0
