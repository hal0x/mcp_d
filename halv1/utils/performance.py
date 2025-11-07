"""Модуль для сбора метрик производительности."""

from __future__ import annotations

import time
import functools
import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class PerformanceMetrics:
    """Метрики производительности для операции."""
    operation: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: deque = field(default_factory=lambda: deque(maxlen=100))  # Последние 100 измерений
    
    @property
    def avg_time(self) -> float:
        """Среднее время выполнения."""
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def median_time(self) -> float:
        """Медианное время выполнения."""
        return statistics.median(self.times) if self.times else 0.0
    
    @property
    def p95_time(self) -> float:
        """95-й процентиль времени выполнения."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index]
    
    def add_measurement(self, execution_time: float) -> None:
        """Добавить измерение времени."""
        self.total_calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.times.append(execution_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить сводку метрик."""
        return {
            "operation": self.operation,
            "total_calls": self.total_calls,
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "median_time_ms": round(self.median_time * 1000, 2),
            "p95_time_ms": round(self.p95_time * 1000, 2),
            "min_time_ms": round(self.min_time * 1000, 2),
            "max_time_ms": round(self.max_time * 1000, 2),
            "total_time_ms": round(self.total_time * 1000, 2)
        }

class PerformanceProfiler:
    """Профилировщик производительности."""
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._start_time = time.time()
    
    def get_metrics(self, operation: str) -> PerformanceMetrics:
        """Получить метрики для операции."""
        if operation not in self._metrics:
            self._metrics[operation] = PerformanceMetrics(operation)
        return self._metrics[operation]
    
    def add_measurement(self, operation: str, execution_time: float) -> None:
        """Добавить измерение времени для операции."""
        metrics = self.get_metrics(operation)
        metrics.add_measurement(execution_time)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Получить все метрики."""
        return {op: metrics.get_summary() for op, metrics in self._metrics.items()}
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить самые медленные операции."""
        all_metrics = self.get_all_metrics()
        sorted_ops = sorted(
            all_metrics.values(),
            key=lambda x: x["avg_time_ms"],
            reverse=True
        )
        return sorted_ops[:limit]
    
    def log_summary(self) -> None:
        """Вывести сводку метрик в лог."""
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Uptime: {time.time() - self._start_time:.2f}s")
        
        slowest = self.get_slowest_operations(5)
        if slowest:
            logger.info("Top 5 slowest operations:")
            for op in slowest:
                logger.info(
                    f"  {op['operation']}: "
                    f"avg={op['avg_time_ms']}ms, "
                    f"calls={op['total_calls']}, "
                    f"p95={op['p95_time_ms']}ms"
                )
        
        total_ops = sum(m.total_calls for m in self._metrics.values())
        total_time = sum(m.total_time for m in self._metrics.values())
        logger.info(f"Total operations: {total_ops}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info("==========================")

# Глобальный экземпляр профилировщика
profiler = PerformanceProfiler()

def measure_time(operation: str):
    """Декоратор для измерения времени выполнения синхронных функций."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                profiler.add_measurement(operation, execution_time)
                logger.debug(f"{operation} took {execution_time*1000:.2f}ms")
        return wrapper
    return decorator

def measure_time_async(operation: str):
    """Декоратор для измерения времени выполнения асинхронных функций."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                profiler.add_measurement(operation, execution_time)
                logger.debug(f"{operation} took {execution_time*1000:.2f}ms")
        return wrapper
    return decorator

@contextmanager
def measure_context(operation: str):
    """Контекст-менеджер для измерения времени выполнения блока кода."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        execution_time = time.perf_counter() - start_time
        profiler.add_measurement(operation, execution_time)
        logger.debug(f"{operation} took {execution_time*1000:.2f}ms")

@asynccontextmanager
async def measure_context_async(operation: str):
    """Асинхронный контекст-менеджер для измерения времени."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        execution_time = time.perf_counter() - start_time
        profiler.add_measurement(operation, execution_time)
        logger.debug(f"{operation} took {execution_time*1000:.2f}ms")

def get_performance_summary() -> Dict[str, Any]:
    """Получить сводку производительности."""
    return {
        "uptime": time.time() - profiler._start_time,
        "operations": profiler.get_all_metrics(),
        "slowest_operations": profiler.get_slowest_operations(10)
    }

def log_performance_summary() -> None:
    """Вывести сводку производительности в лог."""
    profiler.log_summary()
