"""Утилита для извлечения новых сообщений из Telegram дампов."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .datetime_utils import parse_datetime_flexible
from .deduplication import get_message_hash

logger = logging.getLogger(__name__)


class MessageExtractor:
    """Класс для извлечения новых сообщений с расширенной функциональностью."""

    def __init__(self, input_dir: str = "input", chats_dir: str = "chats"):
        """
        Инициализация экстрактора сообщений.

        Args:
            input_dir: Директория с исходными данными (input)
            chats_dir: Директория для сохранения обработанных чатов
        """
        self.input_dir = Path(input_dir)
        self.chats_dir = Path(chats_dir)
        current_year = datetime.now().year
        self.cutoff_date = datetime(current_year, 1, 1, tzinfo=timezone.utc)
        self.stats = {
            "total_chats": 0,
            "processed_chats": 0,
            "skipped_chats": 0,
            "total_messages_input": 0,
            "total_messages_output": 0,
            "messages_copied": 0,
            "messages_filtered_by_date": 0,
            "duplicates_skipped": 0,
            "errors": 0,
            "files_processed": 0,
            "files_skipped": 0,
        }
        self.existing_messages_cache = {}

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Парсинг даты из различных форматов.

        Args:
            date_str: Строка с датой в различных форматах

        Returns:
            Объект datetime или None, если не удалось распарсить
        """
        return parse_datetime_flexible(date_str)

    def get_message_hash(self, message: Dict) -> str:
        """
        Получение хеша сообщения для дедупликации.

        Использует общую функцию из модуля дедупликации.

        Args:
            message: Словарь с данными сообщения

        Returns:
            MD5 хеш сообщения
        """
        return get_message_hash(message)

    def load_existing_messages(self, chat_dir: Path) -> Tuple[Set[str], Set[str]]:
        """
        Загрузка существующих сообщений для дедупликации.

        Args:
            chat_dir: Директория чата с существующими сообщениями

        Returns:
            Кортеж (множество ID сообщений, множество хешей сообщений)
        """
        existing_ids = set()
        existing_hashes = set()

        if chat_dir not in self.existing_messages_cache:
            for json_file in chat_dir.glob("*.json"):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        for line in f:
                            try:
                                message = json.loads(line.strip())
                                if isinstance(message, dict) and "id" in message:
                                    existing_ids.add(str(message["id"]))
                                    # Добавляем хеш для дополнительной дедупликации
                                    msg_hash = self.get_message_hash(message)
                                    existing_hashes.add(msg_hash)
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

            self.existing_messages_cache[chat_dir] = (existing_ids, existing_hashes)

        return self.existing_messages_cache[chat_dir]

    def filter_messages(
        self,
        messages: List[Dict],
        existing_ids: Set[str],
        existing_hashes: Set[str],
        filter_by_date: bool = True,
    ) -> List[Dict]:
        """
        Фильтрация сообщений по дате и дубликатам.

        Args:
            messages: Список сообщений для фильтрации
            existing_ids: Множество существующих ID сообщений
            existing_hashes: Множество существующих хешей сообщений
            filter_by_date: Включить ли фильтрацию по дате

        Returns:
            Отфильтрованный список сообщений
        """
        filtered = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            # Проверка дубликатов по ID
            if "id" in message and str(message["id"]) in existing_ids:
                self.stats["duplicates_skipped"] += 1
                continue

            # Проверка дубликатов по хешу
            msg_hash = self.get_message_hash(message)
            if msg_hash in existing_hashes:
                self.stats["duplicates_skipped"] += 1
                continue

            # Фильтрация по дате
            if filter_by_date:
                # Поддерживаем оба формата даты: "date" и "date_utc"
                date_field = message.get("date") or message.get("date_utc")
                if date_field:
                    msg_date = self.parse_date(date_field)
                    if msg_date and msg_date < self.cutoff_date:
                        self.stats["messages_filtered_by_date"] += 1
                        continue

            filtered.append(message)
            self.stats["messages_copied"] += 1

        return filtered

    def extract_chat_messages(
        self,
        input_chat_dir: Path,
        chats_chat_dir: Path,
        dry_run: bool = False,
        filter_by_date: bool = True,
    ) -> Dict[str, int]:
        """
        Извлечение сообщений для одного чата.

        Args:
            input_chat_dir: Директория чата в input
            chats_chat_dir: Директория чата в chats
            dry_run: Режим тестирования без сохранения
            filter_by_date: Включить ли фильтрацию по дате

        Returns:
            Словарь со статистикой обработки чата
        """
        chat_stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "messages_copied": 0,
            "messages_filtered_by_date": 0,
            "duplicates_skipped": 0,
            "errors": 0,
        }

        if not input_chat_dir.exists():
            return chat_stats

        # Загружаем существующие сообщения для дедупликации
        existing_ids, existing_hashes = self.load_existing_messages(chats_chat_dir)

        # Обрабатываем все JSON файлы в директории чата
        for json_file in input_chat_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    messages = []
                    for line in f:
                        try:
                            message = json.loads(line.strip())
                            messages.append(message)
                        except json.JSONDecodeError:
                            continue

                # Фильтруем сообщения
                filtered_messages = self.filter_messages(
                    messages, existing_ids, existing_hashes, filter_by_date
                )

                if filtered_messages:
                    if not dry_run:
                        # Создаем директорию если не существует
                        chats_chat_dir.mkdir(parents=True, exist_ok=True)

                        # Записываем новые сообщения
                        output_file = chats_chat_dir / json_file.name
                        with open(output_file, "a", encoding="utf-8") as f:
                            for message in filtered_messages:
                                f.write(json.dumps(message, ensure_ascii=False) + "\n")

                    chat_stats["files_processed"] += 1
                    chat_stats["messages_copied"] += len(filtered_messages)
                else:
                    chat_stats["files_skipped"] += 1

            except Exception as e:
                logger.error(f"Ошибка обработки файла {json_file}: {e}")
                chat_stats["errors"] += 1

        return chat_stats

    def extract_all_messages(
        self,
        dry_run: bool = False,
        filter_by_date: bool = True,
        chat_filter: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Извлечение сообщений для всех чатов.

        Args:
            dry_run: Режим тестирования без сохранения
            filter_by_date: Включить ли фильтрацию по дате
            chat_filter: Фильтр по названию чата (опционально)

        Returns:
            Словарь со статистикой обработки
        """
        if not self.input_dir.exists():
            logger.error(f"Директория {self.input_dir} не найдена")
            return self.stats

        # Получаем список всех чатов
        chat_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        self.stats["total_chats"] = len(chat_dirs)

        for chat_dir in chat_dirs:
            chat_name = chat_dir.name

            # Фильтрация по названию чата
            if chat_filter and chat_filter.lower() not in chat_name.lower():
                self.stats["skipped_chats"] += 1
                continue

            chats_chat_dir = self.chats_dir / chat_name

            logger.info(f"Обработка чата: {chat_name}")

            # Извлекаем сообщения для чата
            chat_stats = self.extract_chat_messages(
                chat_dir, chats_chat_dir, dry_run, filter_by_date
            )

            # Обновляем общую статистику
            for key, value in chat_stats.items():
                self.stats[key] += value

            self.stats["processed_chats"] += 1

            logger.info(
                f"Чата {chat_name}: {chat_stats['messages_copied']} сообщений скопировано"
            )

        return self.stats

    def print_stats(self):
        """Вывод статистики извлечения."""
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА ИЗВЛЕЧЕНИЯ СООБЩЕНИЙ")
        print("=" * 60)
        print(f"📁 Всего чатов: {self.stats['total_chats']}")
        print(f"✅ Обработано чатов: {self.stats['processed_chats']}")
        print(f"⏭️  Пропущено чатов: {self.stats['skipped_chats']}")
        print(f"📄 Обработано файлов: {self.stats['files_processed']}")
        print(f"⏭️  Пропущено файлов: {self.stats['files_skipped']}")
        print(f"📨 Всего сообщений на входе: {self.stats['total_messages_input']}")
        print(f"📤 Сообщений скопировано: {self.stats['messages_copied']}")
        print(f"📅 Отфильтровано по дате: {self.stats['messages_filtered_by_date']}")
        print(f"🔄 Пропущено дубликатов: {self.stats['duplicates_skipped']}")
        print(f"❌ Ошибок: {self.stats['errors']}")
        print("=" * 60)

