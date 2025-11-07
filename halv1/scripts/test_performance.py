#!/usr/bin/env python3
"""Скрипт для тестирования системы метрик производительности."""

import asyncio
import logging
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.performance import (
    measure_context,
    measure_context_async,
    measure_time,
    measure_time_async,
    log_performance_summary,
    get_performance_summary,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)


@measure_time("sync_function")
def sync_function():
    """Синхронная функция для тестирования."""
    logger.info("Выполняется синхронная функция")
    import time
    time.sleep(0.1)  # Имитируем работу
    return "sync_result"


@measure_time_async("async_function")
async def async_function():
    """Асинхронная функция для тестирования."""
    logger.info("Выполняется асинхронная функция")
    await asyncio.sleep(0.1)  # Имитируем асинхронную работу
    return "async_result"


def test_sync_operations():
    """Тест синхронных операций."""
    logger.info("=== Тестирование синхронных операций ===")
    
    # Тест декоратора
    result = sync_function()
    logger.info(f"Результат: {result}")
    
    # Тест контекст-менеджера
    with measure_context("sync_context"):
        logger.info("Внутри синхронного контекста")
        import time
        time.sleep(0.05)
    
    logger.info("Синхронные операции завершены")


async def test_async_operations():
    """Тест асинхронных операций."""
    logger.info("=== Тестирование асинхронных операций ===")
    
    # Тест декоратора
    result = await async_function()
    logger.info(f"Результат: {result}")
    
    # Тест контекст-менеджера
    async with measure_context_async("async_context"):
        logger.info("Внутри асинхронного контекста")
        await asyncio.sleep(0.05)
    
    logger.info("Асинхронные операции завершены")


def test_nested_operations():
    """Тест вложенных операций."""
    logger.info("=== Тестирование вложенных операций ===")
    
    @measure_time("outer_function")
    def outer_function():
        logger.info("Внешняя функция")
        
        with measure_context("inner_context"):
            logger.info("Внутренний контекст")
            import time
            time.sleep(0.02)
        
        return "nested_result"
    
    result = outer_function()
    logger.info(f"Результат: {result}")


async def test_mixed_operations():
    """Тест смешанных операций."""
    logger.info("=== Тестирование смешанных операций ===")
    
    @measure_time_async("mixed_function")
    async def mixed_function():
        logger.info("Смешанная функция")
        
        # Синхронная операция внутри асинхронной
        with measure_context("sync_in_async"):
            logger.info("Синхронная операция в асинхронной функции")
            import time
            time.sleep(0.02)
        
        # Асинхронная операция
        async with measure_context_async("async_in_async"):
            logger.info("Асинхронная операция в асинхронной функции")
            await asyncio.sleep(0.02)
        
        return "mixed_result"
    
    result = await mixed_function()
    logger.info(f"Результат: {result}")


async def main():
    """Основная функция тестирования."""
    logger.info("Запуск тестирования системы метрик производительности")
    
    try:
        # Тестируем синхронные операции
        test_sync_operations()
        
        # Тестируем асинхронные операции
        await test_async_operations()
        
        # Тестируем вложенные операции
        test_nested_operations()
        
        # Тестируем смешанные операции
        await test_mixed_operations()
        
        # Выводим сводку производительности
        logger.info("=== Сводка производительности ===")
        summary = get_performance_summary()
        
        logger.info(f"Время работы: {summary['uptime']:.2f}s")
        logger.info(f"Количество операций: {len(summary['operations'])}")
        
        for op_name, metrics in summary['operations'].items():
            logger.info(
                f"  {op_name}: "
                f"вызовов={metrics['total_calls']}, "
                f"среднее={metrics['avg_time_ms']}ms, "
                f"p95={metrics['p95_time_ms']}ms"
            )
        
        # Выводим самые медленные операции
        logger.info("=== Топ-5 самых медленных операций ===")
        for i, op in enumerate(summary['slowest_operations'][:5], 1):
            logger.info(
                f"  {i}. {op['operation']}: "
                f"среднее={op['avg_time_ms']}ms, "
                f"вызовов={op['total_calls']}"
            )
        
        logger.info("Тестирование завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
