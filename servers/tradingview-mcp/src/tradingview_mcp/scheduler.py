"""Планировщик для автоматического запуска Professional Scanners."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ScannerScheduler:
    """Планировщик для автоматического запуска сканеров по расписанию."""
    
    def __init__(self, config_path: str = "/app/configs/scanner_scheduler.yaml"):
        """Инициализация планировщика.
        
        Args:
            config_path: Путь к файлу конфигурации планировщика
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.is_running = False
        self.last_run_time: Optional[datetime] = None
        self.task: Optional[asyncio.Task] = None
        
    def load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию планировщика."""
        try:
            # Пытаемся найти конфигурацию относительно модуля
            if not Path(self.config_path).is_absolute():
                module_dir = Path(__file__).parent
                config_file = module_dir / self.config_path
            else:
                config_file = Path(self.config_path)
                
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return {
                    "enabled": False,
                    "interval_minutes": 15,
                    "scanners": []
                }
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            logger.info(f"Loaded scheduler config: {self.config}")
            return self.config
            
        except Exception as exc:
            logger.error(f"Failed to load scheduler config: {exc}")
            return {
                "enabled": False,
                "interval_minutes": 15,
                "scanners": []
            }
    
    async def get_symbols_for_scanning(self) -> List[str]:
        """Получает список символов для сканирования.
        
        Использует существующие инструменты для поиска подходящих кандидатов.
        """
        try:
            # Импортируем функции из server.py
            from tradingview_mcp.server import top_gainers_batch, smart_scanner
            
            # Получаем топ-20 символов с разных бирж
            symbols = set()
            
            # Добавляем топ-гайнеры с разных бирж используя batch функцию
            exchanges = ["BINANCE", "KUCOIN", "BYBIT"]
            batch_requests = [
                {"exchange": exchange, "timeframe": "15m", "limit": 10}
                for exchange in exchanges
            ]
            
            try:
                batch_results = top_gainers_batch(batch_requests)
                for result in batch_results:
                    if "error" not in result:
                        exchange = result.get("exchange", "")
                        for item in result.get("results", []):
                            symbol = item.get("symbol", "").replace(f"{exchange}:", "")
                            if symbol:
                                symbols.add(symbol)
            except Exception as exc:
                logger.warning(f"Failed to get top gainers batch: {exc}")
            
            # Добавляем результаты smart_scanner
            try:
                smart_results = smart_scanner(exchange="BINANCE", timeframe="1h", limit=15)
                for item in smart_results:
                    symbol = item.get("symbol", "").replace("BINANCE:", "")
                    if symbol:
                        symbols.add(symbol)
            except Exception as exc:
                logger.warning(f"Failed to get smart scanner results: {exc}")
            
            # Преобразуем в список и ограничиваем до 30 символов
            symbol_list = list(symbols)[:30]
            logger.info(f"Selected {len(symbol_list)} symbols for scanning: {symbol_list[:5]}...")
            
            return symbol_list
            
        except Exception as exc:
            logger.error(f"Failed to get symbols for scanning: {exc}")
            # Fallback к базовому списку
            return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
    
    async def run_scanner(self, scanner_name: str, symbols: List[str], profile: str) -> None:
        """Запускает конкретный сканер и сохраняет результат."""
        try:
            logger.info(f"Running {scanner_name} with {len(symbols)} symbols, profile: {profile}")
            
            # Импортируем функции сканеров из server.py
            from tradingview_mcp.server import (
                pro_momentum_scan,
                pro_mean_revert_scan,
                pro_breakout_scan,
                pro_volume_profile_scan
            )
            
            # Выбираем нужную функцию сканера
            scanner_functions = {
                "pro_momentum_scan": pro_momentum_scan,
                "pro_mean_revert_scan": pro_mean_revert_scan,
                "pro_breakout_scan": pro_breakout_scan,
                "pro_volume_profile_scan": pro_volume_profile_scan,
            }
            
            if scanner_name not in scanner_functions:
                logger.error(f"Unknown scanner: {scanner_name}")
                return
            
            scanner_func = scanner_functions[scanner_name]
            
            # Запускаем сканер
            start_time = time.time()
            result = await scanner_func(symbols, profile)
            duration = time.time() - start_time
            
            logger.info(f"{scanner_name} completed in {duration:.2f}s, found {len(result)} signals")
            
            # Сохраняем результаты в Redis
            try:
                from tradingview_mcp.server import _save_scanner_result_redis
                request_id = await _save_scanner_result_redis(scanner_name, symbols, profile, result)
                logger.info(f"Saved {scanner_name} results to Redis with ID: {request_id}")
            except Exception as exc:
                logger.error(f"Failed to save {scanner_name} results to Redis: {exc}")
            
        except Exception as exc:
            logger.error(f"Failed to run {scanner_name}: {exc}", exc_info=True)
    
    async def run_scheduled_scan(self) -> None:
        """Выполняет запланированное сканирование."""
        try:
            logger.info("Starting scheduled scan")
            self.last_run_time = datetime.utcnow()
            
            # Загружаем конфигурацию
            config = self.load_config()
            logger.info(f"Loaded config: {config}")
            
            if not config.get("enabled", False):
                logger.info("Scheduler is disabled in config")
                return
            
            # Получаем список символов
            symbols = await self.get_symbols_for_scanning()
            if not symbols:
                logger.warning("No symbols available for scanning")
                return
            
            # Запускаем включенные сканеры
            scanners = config.get("scanners", [])
            logger.info(f"Found {len(scanners)} scanners in config")
            
            for scanner_config in scanners:
                logger.info(f"Processing scanner config: {scanner_config}")
                
                if not scanner_config.get("enabled", False):
                    logger.info(f"Scanner {scanner_config.get('name', 'unknown')} is disabled, skipping")
                    continue
                
                scanner_name = scanner_config.get("name")
                profile = scanner_config.get("profile", "balanced")
                
                if not scanner_name:
                    logger.warning("Scanner config missing name")
                    continue
                
                logger.info(f"Running scanner: {scanner_name} with profile: {profile}")
                await self.run_scanner(scanner_name, symbols, profile)
                logger.info(f"Scanner {scanner_name} completed")
            
            logger.info("Scheduled scan completed successfully")
            
        except Exception as exc:
            logger.error(f"Scheduled scan failed: {exc}", exc_info=True)
    
    async def start(self) -> None:
        """Запускает планировщик."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        config = self.load_config()
        if not config.get("enabled", False):
            logger.info("Scheduler is disabled, not starting")
            return
        
        self.is_running = True
        interval_minutes = config.get("interval_minutes", 15)
        interval_seconds = interval_minutes * 60
        
        logger.info(f"Starting scanner scheduler with {interval_minutes} minute interval")
        
        # Запускаем периодическую задачу (включая первую проверку)
        self.task = asyncio.create_task(self._run_periodic())
        logger.info("Scheduler task created and running in background")
    
    async def _run_periodic(self) -> None:
        """Выполняет периодические запуски сканирования."""
        config = self.load_config()
        interval_minutes = config.get("interval_minutes", 15)
        interval_seconds = interval_minutes * 60
        
        while self.is_running:
            try:
                # Выполняем сканирование сразу
                await self.run_scheduled_scan()
                
                # Ждем до следующего запуска
                if self.is_running:
                    await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Periodic scan failed: {exc}")
                # При ошибке ждем немного перед повтором
                if self.is_running:
                    await asyncio.sleep(60)  # 1 минута при ошибке
    
    async def stop(self) -> None:
        """Останавливает планировщик."""
        if not self.is_running:
            logger.info("Scheduler is not running")
            return
        
        logger.info("Stopping scanner scheduler")
        self.is_running = False
        
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scanner scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус планировщика."""
        return {
            "is_running": self.is_running,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "config": self.config,
            "task_running": self.task is not None and not self.task.done() if self.task else False
        }


# Глобальный экземпляр планировщика
scheduler = ScannerScheduler()
