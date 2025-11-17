#!/usr/bin/env python3
"""
Фоновый сервис автоматической индексации чатов.

Периодически проверяет директорию input на наличие новых сообщений,
извлекает их в chats и автоматически запускает индексацию.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import chromadb
from zoneinfo import ZoneInfo

from ..config import get_settings
from ..utils.message_extractor import MessageExtractor
from ..utils.naming import slugify

logger = logging.getLogger(__name__)


class BackgroundIndexingService:
    """Сервис фоновой индексации чатов."""

    def __init__(
        self,
        input_path: str = "input",
        chats_path: str = "chats",
        chroma_path: str = "./chroma_db",
        check_interval: int = 60,
    ):
        """
        Инициализация сервиса фоновой индексации.

        Args:
            input_path: Путь к директории input с новыми сообщениями
            chats_path: Путь к директории chats для сохранения сообщений
            chroma_path: Путь к ChromaDB для хранения состояния
            check_interval: Интервал проверки в секундах (по умолчанию 60)
        """
        self.input_path = Path(input_path)
        self.chats_path = Path(chats_path)
        self.chroma_path = Path(chroma_path)
        self.check_interval = check_interval

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._progress_collection: Optional[chromadb.Collection] = None
        self._message_extractor: Optional[MessageExtractor] = None

        # Callback для запуска индексации (будет установлен из MCP сервера)
        self._index_chat_callback: Optional[callable] = None

    def set_index_chat_callback(self, callback: callable):
        """Установить callback для запуска индексации чата."""
        self._index_chat_callback = callback

    def _initialize_chromadb(self):
        """Инициализация ChromaDB для хранения состояния."""
        try:
            if not self.chroma_path.exists():
                self.chroma_path.mkdir(parents=True, exist_ok=True)

            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))

            # Получаем или создаем коллекцию для прогресса
            try:
                self._progress_collection = self._chroma_client.get_collection(
                    "indexing_progress"
                )
            except Exception:
                self._progress_collection = self._chroma_client.create_collection(
                    name="indexing_progress",
                    metadata={
                        "description": "Отслеживание прогресса индексации и состояния фонового процесса"
                    },
                )

            logger.info("ChromaDB инициализирован для фоновой индексации")
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}", exc_info=True)
            raise

    def _get_last_check_time(self) -> Optional[datetime]:
        """Получить время последней проверки input директории из ChromaDB."""
        if not self._progress_collection:
            return None

        try:
            result = self._progress_collection.get(
                ids=["background_indexing_state"], include=["metadatas"]
            )

            if result.get("ids") and len(result["ids"]) > 0:
                metadata = result.get("metadatas", [{}])[0]
                last_check_str = metadata.get("last_input_check_time")
                if last_check_str:
                    from ..utils.datetime_utils import parse_datetime_utc

                    return parse_datetime_utc(last_check_str, use_zoneinfo=True)
        except Exception as e:
            logger.debug(f"Не удалось получить время последней проверки: {e}")

        return None

    def _save_last_check_time(self, check_time: datetime):
        """Сохранить время последней проверки input директории в ChromaDB."""
        if not self._progress_collection:
            return

        try:
            metadata = {
                "last_input_check_time": check_time.isoformat(),
                "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            }

            # Используем пустой эмбеддинг (не нужен для метаданных)
            dummy_embedding = [0.0] * 1024

            self._progress_collection.upsert(
                ids=["background_indexing_state"],
                documents=["Background indexing state"],
                embeddings=[dummy_embedding],
                metadatas=[metadata],
            )

            logger.debug(f"Сохранено время последней проверки: {check_time.isoformat()}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении времени проверки: {e}")

    def _get_updated_chats(self, last_check_time: Optional[datetime]) -> Set[str]:
        """
        Получить список чатов с обновленными файлами в input директории.

        Args:
            last_check_time: Время последней проверки (None для первой проверки)

        Returns:
            Множество названий чатов с обновлениями
        """
        updated_chats = set()

        if not self.input_path.exists():
            logger.debug(f"Директория input не найдена: {self.input_path}")
            return updated_chats

        # Проходим по всем чатам в input
        for chat_dir in self.input_path.iterdir():
            if not chat_dir.is_dir():
                continue

            chat_name = chat_dir.name

            # Проверяем все JSON файлы в директории чата
            for json_file in chat_dir.glob("*.json"):
                try:
                    file_mtime = datetime.fromtimestamp(
                        json_file.stat().st_mtime, tz=timezone.utc
                    )

                    # Если файл был изменен после последней проверки
                    if last_check_time is None or file_mtime > last_check_time:
                        updated_chats.add(chat_name)
                        logger.debug(
                            f"Найден обновленный чат: {chat_name} (файл {json_file.name} изменен {file_mtime})"
                        )
                        break  # Достаточно одного обновленного файла
                except Exception as e:
                    logger.warning(f"Ошибка при проверке файла {json_file}: {e}")
                    continue

        return updated_chats

    async def _extract_new_messages(self, chat_filter: Optional[str] = None) -> Dict[str, int]:
        """
        Извлечь новые сообщения из input в chats через MessageExtractor.

        Args:
            chat_filter: Фильтр по названию чата (опционально)

        Returns:
            Статистика извлечения
        """
        if not self._message_extractor:
            self._message_extractor = MessageExtractor(
                input_dir=str(self.input_path), chats_dir=str(self.chats_path)
            )

        # Извлекаем сообщения (без фильтрации по дате, так как проверяем по времени файла)
        stats = self._message_extractor.extract_all_messages(
            dry_run=False, filter_by_date=False, chat_filter=chat_filter
        )

        return stats

    async def _index_updated_chats(self, updated_chats: Set[str]):
        """
        Запустить индексацию для обновленных чатов.

        Args:
            updated_chats: Множество названий чатов для индексации
        """
        if not self._index_chat_callback:
            logger.warning("Callback для индексации не установлен, пропускаем индексацию")
            return

        for chat_name in updated_chats:
            try:
                logger.info(f"Запуск индексации для обновленного чата: {chat_name}")

                # Используем callback для запуска индексации
                # Создаем request объект
                from ..mcp.schema import IndexChatRequest

                request = IndexChatRequest(
                    chat=chat_name,
                    force_full=False,  # Инкрементальная индексация
                    recent_days=0,  # Индексируем все новые сообщения
                    progress=False,
                )

                # Запускаем индексацию через callback
                response = await self._index_chat_callback(request)
                
                if response and hasattr(response, 'status'):
                    logger.info(f"Индексация для чата {chat_name} запущена (job_id: {getattr(response, 'job_id', 'N/A')})")
                else:
                    logger.info(f"Индексация для чата {chat_name} запущена")
            except Exception as e:
                logger.error(
                    f"Ошибка при запуске индексации для чата {chat_name}: {e}",
                    exc_info=True,
                )
                # Продолжаем обработку других чатов

    async def _check_and_process(self):
        """Основная логика проверки и обработки новых сообщений."""
        try:
            logger.debug("Начало проверки input директории")

            # Получаем время последней проверки
            last_check_time = self._get_last_check_time()

            # Находим обновленные чаты
            updated_chats = self._get_updated_chats(last_check_time)

            if not updated_chats:
                logger.debug("Нет обновленных чатов")
            else:
                logger.info(f"Найдено обновленных чатов: {len(updated_chats)}: {list(updated_chats)}")

                # Извлекаем новые сообщения для всех обновленных чатов
                stats = await self._extract_new_messages()
                logger.info(
                    f"Извлечено сообщений: {stats.get('messages_copied', 0)}, "
                    f"дубликатов пропущено: {stats.get('duplicates_skipped', 0)}"
                )

                # Запускаем индексацию для обновленных чатов только если были извлечены новые сообщения
                if stats.get('messages_copied', 0) > 0:
                    await self._index_updated_chats(updated_chats)
                else:
                    logger.debug("Нет новых сообщений для индексации")

            # Сохраняем время текущей проверки
            current_check_time = datetime.now(ZoneInfo("UTC"))
            self._save_last_check_time(current_check_time)

            logger.debug("Проверка input директории завершена")

        except Exception as e:
            logger.error(f"Ошибка при проверке и обработке: {e}", exc_info=True)
            # Продолжаем работу даже при ошибках

    async def _run_loop(self):
        """Основной цикл фонового процесса."""
        logger.info(
            f"Фоновая индексация запущена (интервал: {self.check_interval} секунд)"
        )

        # Инициализируем ChromaDB
        try:
            self._initialize_chromadb()
        except Exception as e:
            logger.error(f"Не удалось инициализировать ChromaDB: {e}")
            return

        while self._running:
            try:
                await self._check_and_process()
            except Exception as e:
                logger.error(f"Ошибка в цикле фоновой индексации: {e}", exc_info=True)

            # Ждем до следующей проверки
            if self._running:
                await asyncio.sleep(self.check_interval)

        logger.info("Фоновая индексация остановлена")

    def start(self):
        """Запустить фоновый процесс индексации."""
        if self._running:
            logger.warning("Фоновая индексация уже запущена")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Фоновая индексация запущена")

    def stop(self):
        """Остановить фоновый процесс индексации."""
        if not self._running:
            logger.warning("Фоновая индексация не запущена")
            return

        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Остановка фоновой индексации...")

    async def stop_async(self):
        """Асинхронная остановка фонового процесса."""
        self.stop()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def is_running(self) -> bool:
        """Проверить, запущен ли фоновый процесс."""
        return self._running

    def get_status(self) -> Dict[str, any]:
        """Получить статус фонового процесса."""
        last_check_time = self._get_last_check_time()

        return {
            "running": self._running,
            "check_interval": self.check_interval,
            "last_check_time": last_check_time.isoformat() if last_check_time else None,
            "input_path": str(self.input_path),
            "chats_path": str(self.chats_path),
        }

