"""Система мониторинга и алертов для HAL AI-агента."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from utils.performance import PerformanceProfiler, get_performance_summary

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Уровни алертов."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Алерт для отправки в Telegram."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_telegram_message(self) -> str:
        """Преобразует алерт в сообщение для Telegram."""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }[self.level]
        
        return f"{emoji} **{self.title}**\n\n{self.message}\n\n_Время: {self.timestamp.strftime('%H:%M:%S')}_"


@dataclass
class MetricThreshold:
    """Пороговое значение для метрики."""
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    check_function: Callable[[float], bool] = lambda x: x > 0


class MonitoringService:
    """Сервис мониторинга производительности и алертов."""
    
    def __init__(
        self,
        summary_chat_id: int,
        telegram_bot=None,
        check_interval: int = 60,  # секунды
    ):
        self.summary_chat_id = summary_chat_id
        self.telegram_bot = telegram_bot
        self.check_interval = check_interval
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Профилировщик производительности
        self.profiler = PerformanceProfiler()
        
        # Пороговые значения для алертов
        self.thresholds = {
            "avg_time_ms": MetricThreshold(
                metric_name="avg_time_ms",
                warning_threshold=1000,  # 1 секунда
                error_threshold=5000,    # 5 секунд
                critical_threshold=10000 # 10 секунд
            ),
            "p95_time_ms": MetricThreshold(
                metric_name="p95_time_ms",
                warning_threshold=2000,  # 2 секунды
                error_threshold=8000,    # 8 секунд
                critical_threshold=15000 # 15 секунд
            ),
            "total_calls": MetricThreshold(
                metric_name="total_calls",
                warning_threshold=100,
                error_threshold=500,
                critical_threshold=1000
            )
        }
        
        # История алертов для предотвращения спама
        self.alert_history: Dict[str, datetime] = {}
        self.alert_cooldown = 300  # 5 минут между одинаковыми алертами
    
    async def start(self) -> None:
        """Запускает мониторинг."""
        if self.is_running:
            logger.warning("Мониторинг уже запущен")
            return
        
        self.is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("🔍 Мониторинг производительности запущен")
    
    async def stop(self) -> None:
        """Останавливает мониторинг."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Мониторинг производительности остановлен")
    
    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга."""
        while self.is_running:
            try:
                await self._check_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Ошибка в цикле мониторинга: {exc}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_metrics(self) -> None:
        """Проверяет метрики и генерирует алерты."""
        try:
            # Получаем текущие метрики
            summary = get_performance_summary()
            
            # Проверяем каждую операцию
            for operation, metrics in summary.get("operations", {}).items():
                await self._check_operation_metrics(operation, metrics)
            
            # Проверяем общие метрики системы
            await self._check_system_metrics(summary)
            
        except Exception as exc:
            logger.error(f"Ошибка при проверке метрик: {exc}")
    
    async def _check_operation_metrics(self, operation: str, metrics: Dict[str, Any]) -> None:
        """Проверяет метрики конкретной операции."""
        for threshold_name, threshold in self.thresholds.items():
            if threshold_name not in metrics:
                continue
            
            value = metrics[threshold_name]
            alert_level = self._get_alert_level(value, threshold)
            
            if alert_level:
                await self._send_alert(Alert(
                    level=alert_level,
                    title=f"Метрика {operation}",
                    message=f"Значение {threshold_name}: {value:.2f} (порог: {threshold.warning_threshold})",
                    metadata={
                        "operation": operation,
                        "metric": threshold_name,
                        "value": value,
                        "threshold": threshold.warning_threshold
                    }
                ))
    
    async def _check_system_metrics(self, summary: Dict[str, Any]) -> None:
        """Проверяет общие метрики системы."""
        uptime = summary.get("uptime", 0)
        
        # Проверяем время работы
        if uptime > 3600:  # Больше часа
            await self._send_alert(Alert(
                level=AlertLevel.INFO,
                title="Система работает стабильно",
                message=f"Время работы: {uptime/3600:.1f} часов",
                metadata={"uptime": uptime}
            ))
    
    def _get_alert_level(self, value: float, threshold: MetricThreshold) -> Optional[AlertLevel]:
        """Определяет уровень алерта на основе значения и порога."""
        if value >= threshold.critical_threshold:
            return AlertLevel.CRITICAL
        elif value >= threshold.error_threshold:
            return AlertLevel.ERROR
        elif value >= threshold.warning_threshold:
            return AlertLevel.WARNING
        return None
    
    async def _send_alert(self, alert: Alert) -> None:
        """Отправляет алерт в Telegram."""
        # Проверяем cooldown для предотвращения спама
        alert_key = f"{alert.level.value}_{alert.title}"
        now = datetime.now()
        
        if alert_key in self.alert_history:
            last_sent = self.alert_history[alert_key]
            if now - last_sent < timedelta(seconds=self.alert_cooldown):
                logger.debug(f"Алерт {alert_key} пропущен из-за cooldown")
                return
        
        # Отправляем алерт
        if self.telegram_bot:
            try:
                message = alert.to_telegram_message()
                await self.telegram_bot.send_message(
                    chat_id=self.summary_chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
                self.alert_history[alert_key] = now
                logger.info(f"Алерт отправлен: {alert.title}")
            except Exception as exc:
                logger.error(f"Ошибка отправки алерта: {exc}")
        else:
            logger.warning(f"Telegram бот недоступен, алерт не отправлен: {alert.title}")
    
    async def send_manual_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Отправляет ручной алерт."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        await self._send_alert(alert)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получает сводку метрик."""
        return get_performance_summary()
    
    def get_alert_history(self) -> Dict[str, datetime]:
        """Получает историю алертов."""
        return self.alert_history.copy()


# Глобальный экземпляр для использования в других модулях
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> Optional[MonitoringService]:
    """Получает глобальный экземпляр сервиса мониторинга."""
    return _monitoring_service


def set_monitoring_service(service: MonitoringService) -> None:
    """Устанавливает глобальный экземпляр сервиса мониторинга."""
    global _monitoring_service
    _monitoring_service = service


async def send_alert(
    level: AlertLevel,
    title: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Отправляет алерт через глобальный сервис мониторинга."""
    service = get_monitoring_service()
    if service:
        await service.send_manual_alert(level, title, message, metadata)
    else:
        logger.warning(f"Сервис мониторинга недоступен, алерт не отправлен: {title}")
