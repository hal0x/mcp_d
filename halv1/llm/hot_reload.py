"""Система горячей перезагрузки промтов."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class PromptsFileHandler(FileSystemEventHandler):
    """Обработчик изменений файла промтов."""
    
    def __init__(self, reload_callback: Callable[[], None], debounce_seconds: float = 1.0):
        self.reload_callback = reload_callback
        self.debounce_seconds = debounce_seconds
        self.last_modified = 0
        self._reload_task: Optional[asyncio.Task] = None
    
    def on_modified(self, event):
        """Обрабатывает изменение файла."""
        if event.is_directory:
            return
        
        # Проверяем, что это файл промтов
        if not event.src_path.endswith(('prompts.yaml', 'prompts.yml')):
            return
        
        current_time = time.time()
        
        # Debounce: игнорируем изменения слишком часто
        if current_time - self.last_modified < self.debounce_seconds:
            return
        
        self.last_modified = current_time
        logger.info(f"Обнаружено изменение файла промтов: {event.src_path}")
        
        # Запускаем перезагрузку асинхронно
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()
        
        self._reload_task = asyncio.create_task(self._debounced_reload())
    
    async def _debounced_reload(self):
        """Перезагружает промты с задержкой."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            logger.info("Выполняем горячую перезагрузку промтов...")
            self.reload_callback()
            logger.info("✅ Горячая перезагрузка промтов завершена")
        except asyncio.CancelledError:
            logger.debug("Перезагрузка промтов отменена")
        except Exception as exc:
            logger.error(f"Ошибка при горячей перезагрузке промтов: {exc}")


class HotReloadManager:
    """Менеджер горячей перезагрузки промтов."""
    
    def __init__(
        self,
        prompts_file_path: str,
        prompt_manager,
        debounce_seconds: float = 1.0
    ):
        self.prompts_file_path = Path(prompts_file_path)
        self.prompt_manager = prompt_manager
        self.debounce_seconds = debounce_seconds
        self.observer: Optional[Observer] = None
        self.is_running = False
    
    def start(self) -> None:
        """Запускает мониторинг изменений файла."""
        if self.is_running:
            logger.warning("Горячая перезагрузка уже запущена")
            return
        
        if not self.prompts_file_path.exists():
            logger.error(f"Файл промтов не найден: {self.prompts_file_path}")
            return
        
        # Создаем обработчик событий
        handler = PromptsFileHandler(
            reload_callback=self._reload_prompts,
            debounce_seconds=self.debounce_seconds
        )
        
        # Создаем наблюдатель
        self.observer = Observer()
        self.observer.schedule(
            handler,
            path=str(self.prompts_file_path.parent),
            recursive=False
        )
        
        # Запускаем наблюдение
        self.observer.start()
        self.is_running = True
        
        logger.info(f"🔥 Горячая перезагрузка промтов запущена для {self.prompts_file_path}")
    
    def stop(self) -> None:
        """Останавливает мониторинг изменений файла."""
        if not self.is_running or self.observer is None:
            return
        
        self.observer.stop()
        self.observer.join()
        self.is_running = False
        
        logger.info("🛑 Горячая перезагрузка промтов остановлена")
    
    def _reload_prompts(self) -> None:
        """Перезагружает промты."""
        try:
            # Валидируем файл перед перезагрузкой
            if self._validate_prompts_file():
                self.prompt_manager.reload_config()
                logger.info("✅ Промты успешно перезагружены")
            else:
                logger.error("❌ Файл промтов содержит ошибки, перезагрузка отменена")
        except Exception as exc:
            logger.error(f"❌ Ошибка при перезагрузке промтов: {exc}")
    
    def _validate_prompts_file(self) -> bool:
        """Валидирует файл промтов перед перезагрузкой."""
        try:
            import yaml
            
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            
            return True
        except Exception as exc:
            logger.error(f"Ошибка валидации файла промтов: {exc}")
            return False
    
    def __enter__(self):
        """Контекстный менеджер для автоматического запуска/остановки."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер для автоматического запуска/остановки."""
        self.stop()


async def start_hot_reload(
    prompts_file_path: str,
    prompt_manager,
    debounce_seconds: float = 1.0
) -> HotReloadManager:
    """Запускает горячую перезагрузку промтов."""
    manager = HotReloadManager(
        prompts_file_path=prompts_file_path,
        prompt_manager=prompt_manager,
        debounce_seconds=debounce_seconds
    )
    
    manager.start()
    return manager


def create_hot_reload_manager(
    prompts_file_path: str,
    prompt_manager,
    debounce_seconds: float = 1.0
) -> HotReloadManager:
    """Создает менеджер горячей перезагрузки промтов."""
    return HotReloadManager(
        prompts_file_path=prompts_file_path,
        prompt_manager=prompt_manager,
        debounce_seconds=debounce_seconds
    )
