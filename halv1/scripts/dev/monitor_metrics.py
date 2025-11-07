#!/usr/bin/env python3
"""Скрипт для мониторинга метрик производительности в реальном времени."""

import asyncio
import time
import logging
from pathlib import Path
import sys
from typing import Dict, Any

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from utils.performance import get_metrics

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsMonitor:
    """Монитор метрик производительности."""
    
    def __init__(self):
        self.thresholds = {
            "vector_index_embed": 1000,  # 1 секунда
            "vector_index_add": 2000,    # 2 секунды
            "vector_index_search": 1000, # 1 секунда
        }
        self.alert_count = 0
        
    def check_metrics(self) -> Dict[str, Any]:
        """Проверяет текущие метрики и возвращает статистику."""
        metrics = get_metrics()
        alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                avg_time = metrics[metric_name].get("avgtimems", 0)
                if avg_time > threshold:
                    alerts.append({
                        "metric": metric_name,
                        "value": avg_time,
                        "threshold": threshold,
                        "excess": avg_time - threshold
                    })
        
        return {
            "timestamp": time.time(),
            "metrics": metrics,
            "alerts": alerts,
            "alert_count": len(alerts)
        }
    
    def format_alert(self, alert: Dict[str, Any]) -> str:
        """Форматирует предупреждение о превышении порога."""
        return (
            f"🚨 Метрика {alert['metric']}\n"
            f"Значение avgtimems: {alert['value']:.2f} (порог: {alert['threshold']})\n"
            f"Превышение: {alert['excess']:.2f}мс\n"
            f"Время: {time.strftime('%H:%M:%S')}"
        )
    
    async def monitor_loop(self, interval: int = 10):
        """Основной цикл мониторинга."""
        logger.info("🔍 Запуск мониторинга метрик производительности")
        logger.info(f"📊 Интервал проверки: {interval} секунд")
        logger.info(f"⚠️ Пороги: {self.thresholds}")
        logger.info("-" * 50)
        
        while True:
            try:
                stats = self.check_metrics()
                
                if stats["alerts"]:
                    self.alert_count += 1
                    logger.warning(f"⚠️ Обнаружено {len(stats['alerts'])} превышений порогов")
                    
                    for alert in stats["alerts"]:
                        logger.warning(self.format_alert(alert))
                        
                    # Дополнительная диагностика
                    await self.diagnose_performance_issues(stats["alerts"])
                else:
                    logger.info("✅ Все метрики в норме")
                
                # Показываем текущие значения ключевых метрик
                for metric_name in self.thresholds.keys():
                    if metric_name in stats["metrics"]:
                        avg_time = stats["metrics"][metric_name].get("avgtimems", 0)
                        logger.info(f"📈 {metric_name}: {avg_time:.2f}мс")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Мониторинг остановлен пользователем")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка мониторинга: {e}")
                await asyncio.sleep(interval)
    
    async def diagnose_performance_issues(self, alerts: list):
        """Диагностирует проблемы производительности."""
        logger.info("🔍 Диагностика проблем производительности...")
        
        for alert in alerts:
            metric = alert["metric"]
            
            if metric == "vector_index_embed":
                logger.warning("💡 Рекомендации для vector_index_embed:")
                logger.warning("   - Проверьте доступность сервера embeddings")
                logger.warning("   - Убедитесь, что модель загружена")
                logger.warning("   - Рассмотрите использование кеширования")
                logger.warning("   - Проверьте размер обрабатываемого текста")
            
            elif metric == "vector_index_add":
                logger.warning("💡 Рекомендации для vector_index_add:")
                logger.warning("   - Используйте batch-обработку для множественных добавлений")
                logger.warning("   - Проверьте размер индекса")
                logger.warning("   - Рассмотрите асинхронную обработку")
            
            elif metric == "vector_index_search":
                logger.warning("💡 Рекомендации для vector_index_search:")
                logger.warning("   - Оптимизируйте размер top_k")
                logger.warning("   - Проверьте состояние FAISS индекса")
                logger.warning("   - Рассмотрите кеширование результатов поиска")

async def main():
    """Главная функция."""
    monitor = MetricsMonitor()
    
    print("🚀 Мониторинг метрик производительности")
    print("=" * 50)
    print("Нажмите Ctrl+C для остановки")
    print()
    
    try:
        await monitor.monitor_loop(interval=5)  # Проверка каждые 5 секунд
    except KeyboardInterrupt:
        print("\n👋 Мониторинг завершен")

if __name__ == "__main__":
    asyncio.run(main())
