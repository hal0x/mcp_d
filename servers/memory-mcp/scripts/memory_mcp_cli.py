#!/usr/bin/env python3
"""
Главный скрипт для работы с Telegram дампами
Объединяет все функции: индексацию, саммаризацию и поиск
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# Добавляем src в PYTHONPATH для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импортируем только доступные модули
try:
    from memory_mcp.core.ollama_client import OllamaEmbeddingClient

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ OllamaClient недоступен")

try:
    from memory_mcp.analysis.instruction_manager import InstructionManager

    INSTRUCTION_MANAGER_AVAILABLE = True
except ImportError:
    INSTRUCTION_MANAGER_AVAILABLE = False
    print("⚠️ InstructionManager недоступен")

# Импортируем MCP сервер из scripts
try:
    from mcp_server import TelegramDumpMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ TelegramDumpMCP недоступен")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageExtractor:
    """Класс для извлечения новых сообщений с расширенной функциональностью."""

    def __init__(self, input_dir: str = "input", chats_dir: str = "chats"):
        self.input_dir = Path(input_dir)
        self.chats_dir = Path(chats_dir)

        # Определяем начало текущего года для фильтрации
        current_year = datetime.now().year
        self.cutoff_date = datetime(current_year, 1, 1, tzinfo=timezone.utc)

        # Статистика обработки
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

        # Кэш для быстрой проверки существующих сообщений
        self.existing_messages_cache = {}

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсит дату из строки в различных форматах."""
        if not date_str:
            return None

        try:
            # Заменяем Z на +00:00 для UTC
            if date_str.endswith("Z"):
                date_str = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(date_str)
        except ValueError:
            # Пробуем другие форматы
            try:
                return datetime.fromisoformat(date_str.replace("Z", ""))
            except ValueError:
                logger.warning(f"Не удалось распарсить дату: {date_str}")
                return None

    def get_message_hash(self, message: Dict) -> str:
        """Создает хэш сообщения для дополнительной проверки дубликатов."""
        # Используем ID, дату и первые 100 символов текста
        text_preview = str(message.get("text", ""))[:100]
        hash_input = (
            f"{message.get('id', '')}_{message.get('date_utc', '')}_{text_preview}"
        )
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    def load_existing_messages(self, chat_dir: Path) -> Tuple[Set[str], Set[str]]:
        """Загружает существующие ID и хэши сообщений из рабочей директории."""
        existing_ids = set()
        existing_hashes = set()

        if chat_dir.name in self.existing_messages_cache:
            return self.existing_messages_cache[chat_dir.name]

        # Ищем JSON файлы в чате
        json_files = []
        for pattern in ["unknown.json", "result.json", "messages.json"]:
            json_file = chat_dir / pattern
            if json_file.exists():
                json_files.append(json_file)

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            message = json.loads(line)
                            if "id" in message:
                                existing_ids.add(str(message["id"]))
                                existing_hashes.add(self.get_message_hash(message))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ошибка JSON в {json_file}:{line_num}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Ошибка чтения файла {json_file}: {e}")

        # Кэшируем результат
        self.existing_messages_cache[chat_dir.name] = (existing_ids, existing_hashes)
        return existing_ids, existing_hashes

    def filter_messages(
        self,
        messages: List[Dict],
        existing_ids: Set[str],
        existing_hashes: Set[str],
        filter_by_date: bool = True,
    ) -> List[Dict]:
        """Фильтрует сообщения по дате и дубликатам."""
        filtered_messages = []

        for message in messages:
            # Проверяем наличие обязательных полей
            if "id" not in message:
                logger.warning("Сообщение без ID пропущено")
                continue

            message_id = str(message["id"])
            message_hash = self.get_message_hash(message)

            # Проверяем на дубликаты по ID
            if message_id in existing_ids:
                self.stats["duplicates_skipped"] += 1
                continue

            # Проверяем на дубликаты по хэшу (дополнительная защита)
            if message_hash in existing_hashes:
                self.stats["duplicates_skipped"] += 1
                continue

            # Проверяем дату если включена фильтрация
            if filter_by_date and "date_utc" in message:
                msg_date = self.parse_date(message["date_utc"])
                if msg_date and msg_date < self.cutoff_date:
                    self.stats["messages_filtered_by_date"] += 1
                    continue

            # Добавляем сообщение
            filtered_messages.append(message)
            existing_ids.add(message_id)
            existing_hashes.add(message_hash)

        return filtered_messages

    def extract_chat_messages(
        self,
        input_chat_dir: Path,
        chats_chat_dir: Path,
        dry_run: bool = False,
        filter_by_date: bool = True,
    ) -> Dict[str, int]:
        """Извлекает сообщения из одного чата."""
        chat_stats = {
            "messages_input": 0,
            "messages_output": 0,
            "messages_copied": 0,
            "errors": 0,
            "files_processed": 0,
            "files_skipped": 0,
        }

        # Ищем JSON файлы в input чате
        input_json_files = []
        for pattern in ["unknown.json", "result.json", "messages.json", "*.json"]:
            if pattern == "*.json":
                input_json_files.extend(input_chat_dir.glob(pattern))
            else:
                json_file = input_chat_dir / pattern
                if json_file.exists():
                    input_json_files.append(json_file)

        if not input_json_files:
            logger.warning(f"Нет JSON файлов в чате: {input_chat_dir}")
            chat_stats["files_skipped"] = 1
            return chat_stats

        # Загружаем существующие сообщения
        existing_ids, existing_hashes = self.load_existing_messages(chats_chat_dir)

        # Создаем директорию в chats если не существует
        if not dry_run:
            chats_chat_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Обрабатываем чат: {input_chat_dir.name}")
        logger.info(f"Найдено {len(existing_ids)} существующих сообщений")
        logger.info(f"Найдено {len(input_json_files)} файлов для обработки")

        # Обрабатываем каждый JSON файл
        for input_json_file in input_json_files:
            try:
                chat_stats["files_processed"] += 1

                # Читаем сообщения из input
                messages = []
                with open(input_json_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            message = json.loads(line)
                            messages.append(message)
                            chat_stats["messages_input"] += 1
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Ошибка JSON в {input_json_file}:{line_num}: {e}"
                            )
                            chat_stats["errors"] += 1

                # Фильтруем сообщения
                filtered_messages = self.filter_messages(
                    messages, existing_ids, existing_hashes, filter_by_date
                )

                if filtered_messages:
                    # Определяем имя выходного файла
                    output_file = chats_chat_dir / input_json_file.name

                    if not dry_run:
                        # Записываем новые сообщения
                        with open(output_file, "a", encoding="utf-8") as f:
                            for message in filtered_messages:
                                json.dump(message, f, ensure_ascii=False)
                                f.write("\n")

                    chat_stats["messages_copied"] += len(filtered_messages)
                    logger.info(
                        f"Скопировано {len(filtered_messages)} новых сообщений в {output_file}"
                    )
                else:
                    logger.info(f"Нет новых сообщений в файле {input_json_file.name}")

            except Exception as e:
                logger.error(f"Ошибка обработки файла {input_json_file}: {e}")
                chat_stats["errors"] += 1

        chat_stats["messages_output"] = chat_stats["messages_copied"]
        return chat_stats

    def extract_all_messages(
        self,
        dry_run: bool = False,
        filter_by_date: bool = True,
        chat_filter: Optional[str] = None,
    ) -> Dict[str, int]:
        """Извлекает новые сообщения из всех чатов."""
        if not self.input_dir.exists():
            logger.error(f"Директория {self.input_dir} не существует")
            return self.stats

        # Создаем директорию chats если не существует
        if not dry_run:
            self.chats_dir.mkdir(parents=True, exist_ok=True)

        # Получаем список чатов для обработки
        if chat_filter:
            input_chat_dirs = (
                [self.input_dir / chat_filter]
                if (self.input_dir / chat_filter).exists()
                else []
            )
        else:
            input_chat_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]

        self.stats["total_chats"] = len(input_chat_dirs)

        logger.info(f"Найдено {len(input_chat_dirs)} чатов в input для обработки")
        if filter_by_date:
            logger.info(
                f"Фильтр по дате: сообщения с {self.cutoff_date.strftime('%Y-%m-%d')} и новее"
            )
        else:
            logger.info("Фильтрация по дате отключена")

        if dry_run:
            logger.info("РЕЖИМ ТЕСТИРОВАНИЯ - изменения не будут сохранены")

        for input_chat_dir in input_chat_dirs:
            try:
                chats_chat_dir = self.chats_dir / input_chat_dir.name
                chat_stats = self.extract_chat_messages(
                    input_chat_dir, chats_chat_dir, dry_run, filter_by_date
                )

                # Обновляем общую статистику
                self.stats["processed_chats"] += 1
                self.stats["total_messages_input"] += chat_stats["messages_input"]
                self.stats["total_messages_output"] += chat_stats["messages_output"]
                self.stats["messages_copied"] += chat_stats["messages_copied"]
                self.stats["errors"] += chat_stats["errors"]
                self.stats["files_processed"] += chat_stats["files_processed"]
                self.stats["files_skipped"] += chat_stats["files_skipped"]

            except Exception as e:
                logger.error(f"Ошибка обработки чата {input_chat_dir}: {e}")
                self.stats["errors"] += 1
                self.stats["skipped_chats"] += 1

        return self.stats

    def print_stats(self):
        """Выводит подробную статистику обработки."""
        print("\n" + "=" * 80)
        print("СТАТИСТИКА ИЗВЛЕЧЕНИЯ НОВЫХ СООБЩЕНИЙ")
        print("=" * 80)
        print(
            f"Обработано чатов: {self.stats['processed_chats']}/{self.stats['total_chats']}"
        )
        print(f"Пропущено чатов: {self.stats['skipped_chats']}")
        print(f"Обработано файлов: {self.stats['files_processed']}")
        print(f"Пропущено файлов: {self.stats['files_skipped']}")
        print(f"Всего сообщений в input: {self.stats['total_messages_input']}")
        print(f"Извлечено новых сообщений: {self.stats['messages_copied']}")
        print(f"Отфильтровано по дате: {self.stats['messages_filtered_by_date']}")
        print(f"Пропущено дубликатов: {self.stats['duplicates_skipped']}")
        print(f"Ошибок: {self.stats['errors']}")

        if self.stats["total_messages_input"] > 0:
            copy_percent = (
                self.stats["messages_copied"] / self.stats["total_messages_input"]
            ) * 100
            print(f"Процент извлеченных: {copy_percent:.2f}%")


class MessageDeduplicator:
    """Класс для удаления дубликатов сообщений по полю 'id'."""

    def __init__(self, chats_dir: str = "chats"):
        self.chats_dir = Path(chats_dir)
        self.stats = {
            "total_chats": 0,
            "processed_chats": 0,
            "total_messages": 0,
            "duplicates_removed": 0,
            "unique_messages": 0,
            "errors": 0,
        }

    def deduplicate_chat(self, chat_dir: Path) -> Dict[str, int]:
        """Удаляет дубликаты в одном чате."""
        chat_stats = {
            "messages_before": 0,
            "messages_after": 0,
            "duplicates_removed": 0,
            "errors": 0,
        }

        # Ищем JSON файлы в чате (unknown.json или result.json)
        json_files = []
        for pattern in ["unknown.json", "result.json"]:
            json_file = chat_dir / pattern
            if json_file.exists():
                json_files.append(json_file)

        if not json_files:
            logger.warning(f"Нет JSON файлов в чате: {chat_dir}")
            return chat_stats

        for json_file in json_files:
            try:
                # Читаем все сообщения
                messages = []
                with open(json_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            message = json.loads(line)
                            messages.append(message)
                            chat_stats["messages_before"] += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ошибка JSON в {json_file}:{line_num}: {e}")
                            chat_stats["errors"] += 1

                # Удаляем дубликаты по ID
                seen_ids = set()
                unique_messages = []
                duplicates_count = 0

                for message in messages:
                    message_id = str(message.get("id", ""))
                    if message_id and message_id not in seen_ids:
                        seen_ids.add(message_id)
                        unique_messages.append(message)
                    else:
                        duplicates_count += 1

                chat_stats["duplicates_removed"] += duplicates_count
                chat_stats["messages_after"] += len(unique_messages)

                # Перезаписываем файл без дубликатов
                if duplicates_count > 0:
                    with open(json_file, "w", encoding="utf-8") as f:
                        for message in unique_messages:
                            json.dump(message, f, ensure_ascii=False)
                            f.write("\n")

                    logger.info(
                        f"Удалено {duplicates_count} дубликатов в {json_file.name}"
                    )

            except Exception as e:
                logger.error(f"Ошибка обработки файла {json_file}: {e}")
                chat_stats["errors"] += 1

        return chat_stats

    def deduplicate_all_chats(self) -> Dict[str, int]:
        """Удаляет дубликаты во всех чатах."""
        if not self.chats_dir.exists():
            logger.error(f"Директория {self.chats_dir} не существует")
            return self.stats

        chat_dirs = [d for d in self.chats_dir.iterdir() if d.is_dir()]
        self.stats["total_chats"] = len(chat_dirs)

        logger.info(f"Найдено {len(chat_dirs)} чатов для обработки")

        for chat_dir in chat_dirs:
            try:
                chat_stats = self.deduplicate_chat(chat_dir)

                # Обновляем общую статистику
                self.stats["processed_chats"] += 1
                self.stats["total_messages"] += chat_stats["messages_before"]
                self.stats["duplicates_removed"] += chat_stats["duplicates_removed"]
                self.stats["unique_messages"] += chat_stats["messages_after"]
                self.stats["errors"] += chat_stats["errors"]

            except Exception as e:
                logger.error(f"Ошибка обработки чата {chat_dir}: {e}")
                self.stats["errors"] += 1

        return self.stats

    def print_stats(self):
        """Выводит статистику дедупликации."""
        print("\n" + "=" * 70)
        print("СТАТИСТИКА УДАЛЕНИЯ ДУБЛИКАТОВ")
        print("=" * 70)
        print(
            f"Обработано чатов: {self.stats['processed_chats']}/{self.stats['total_chats']}"
        )
        print(f"Всего сообщений: {self.stats['total_messages']}")
        print(f"Уникальных сообщений: {self.stats['unique_messages']}")
        print(f"Удалено дубликатов: {self.stats['duplicates_removed']}")
        print(f"Ошибок: {self.stats['errors']}")

        if self.stats["total_messages"] > 0:
            duplicate_percent = (
                self.stats["duplicates_removed"] / self.stats["total_messages"]
            ) * 100
            print(f"Процент дубликатов: {duplicate_percent:.2f}%")

        print("=" * 70)


class ProcessManager:
    """Класс для управления процессами индексации."""

    @staticmethod
    def kill_processes_by_name(pattern: str) -> int:
        """Остановить процессы по имени"""
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

            lines = result.stdout.split("\n")
            killed_count = 0

            for line in lines:
                if pattern in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"✅ Остановлен процесс {pid}: {line[:80]}...")
                            killed_count += 1
                        except (OSError, ValueError) as e:
                            print(f"❌ Не удалось остановить процесс {pid}: {e}")

            return killed_count

        except Exception as e:
            print(f"❌ Ошибка при поиске процессов: {e}")
            return 0

    @staticmethod
    def stop_ollama():
        """Остановить Ollama сервер"""
        print("\n🛑 Остановка Ollama сервера...")

        # Попробуем остановить через ollama stop
        try:
            result = subprocess.run(
                ["ollama", "stop"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("✅ Ollama сервер остановлен через ollama stop")
            else:
                print(f"⚠️  ollama stop завершился с кодом {result.returncode}")
        except subprocess.TimeoutExpired:
            print("⚠️  ollama stop превысил время ожидания")
        except FileNotFoundError:
            print("⚠️  Команда ollama не найдена")
        except Exception as e:
            print(f"⚠️  Ошибка при остановке Ollama: {e}")

        # Принудительно остановим процессы ollama
        ollama_count = ProcessManager.kill_processes_by_name("ollama")
        if ollama_count > 0:
            print(f"✅ Остановлено {ollama_count} процессов Ollama")
        else:
            print("ℹ️  Процессы Ollama не найдены")

    @staticmethod
    def stop_indexing_processes():
        """Остановить все процессы индексации"""
        print("\n🛑 Остановка процессов индексации...")

        patterns = [
            "python.*index",
            "python.*memory_mcp",
            "python.*process_and_index",
            "python.*summarize",
            "python.*markdown",
        ]

        total_killed = 0
        for pattern in patterns:
            count = ProcessManager.kill_processes_by_name(pattern)
            total_killed += count

        if total_killed > 0:
            print(f"✅ Остановлено {total_killed} процессов индексации")
        else:
            print("ℹ️  Процессы индексации не найдены")

    @staticmethod
    def check_remaining_processes():
        """Проверить оставшиеся процессы"""
        print("\n🔍 Проверка оставшихся процессов...")

        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

            lines = result.stdout.split("\n")
            remaining = []

            for line in lines:
                if (
                    any(
                        keyword in line.lower()
                        for keyword in ["index", "memory_mcp", "ollama", "summarize"]
                    )
                    and "grep" not in line
                ):
                    remaining.append(line)

            if remaining:
                print("⚠️  Найдены оставшиеся процессы:")
                for line in remaining:
                    print(f"   {line[:100]}...")
            else:
                print("✅ Все процессы индексации остановлены")

        except Exception as e:
            print(f"❌ Ошибка при проверке процессов: {e}")

    @staticmethod
    def stop_all_indexing():
        """Остановить все процессы индексации"""
        print("🛑 ОСТАНОВКА ВСЕХ ПРОЦЕССОВ ИНДЕКСАЦИИ")
        print("=" * 50)

        # Остановить процессы индексации
        ProcessManager.stop_indexing_processes()

        # Остановить Ollama
        ProcessManager.stop_ollama()

        # Проверить результат
        ProcessManager.check_remaining_processes()

        print("\n" + "=" * 50)
        print("✅ Остановка завершена!")
        print("\n💡 Для полной остановки Ollama также выполните:")
        print("   pkill -f ollama")
        print("\n💡 Для проверки статуса:")
        print("   ps aux | grep -E '(index|memory_mcp|ollama)' | grep -v grep")


class TelegramDumpManager:
    """Главный менеджер для работы с Telegram дампами"""

    def __init__(self):
        # Инициализируем только доступные компоненты
        if MCP_AVAILABLE:
            self.mcp = TelegramDumpMCP()
        else:
            self.mcp = None

        if OLLAMA_AVAILABLE:
            self.ollama_client = OllamaEmbeddingClient()
        else:
            self.ollama_client = None

        if INSTRUCTION_MANAGER_AVAILABLE:
            self.instruction_manager = InstructionManager()
        else:
            self.instruction_manager = None

        # Инициализируем новые компоненты
        self.message_extractor = MessageExtractor()
        self.message_deduplicator = MessageDeduplicator()

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        if self.ollama_client:
            await self.ollama_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        if self.ollama_client:
            await self.ollama_client.__aexit__(exc_type, exc_val, exc_tb)

    async def check_system(self) -> bool:
        """Проверка системы"""
        print("🔧 Проверка системы...")

        # Проверяем Ollama
        if self.ollama_client:
            if not await self.ollama_client.check_model_availability():
                print("❌ Ollama недоступен или модель не найдена")
                print("Убедитесь, что Ollama запущен и модель установлена")
                return False
            print("✅ Ollama доступен")
        else:
            print("⚠️ OllamaClient недоступен")

        # Проверяем ChromaDB
        if self.mcp:
            try:
                self.mcp.collection = self.mcp.chroma_client.get_collection(
                    "telegram_messages"
                )
                count = self.mcp.collection.count()
                print(f"✅ ChromaDB доступен (сообщений: {count})")
            except:
                print("⚠️ ChromaDB коллекция не найдена")
        else:
            print("⚠️ MCP сервер недоступен")

        return True

    def set_summarization_instruction(
        self,
        *,
        chat: Optional[str] = None,
        mode: Optional[str] = None,
        instruction: Optional[str] = None,
        clear: bool = False,
    ) -> None:
        """Управление специальными инструкциями саммаризации для MCP и CLI."""
        if not self.instruction_manager:
            print("⚠️ InstructionManager недоступен")
            return

        target_count = sum(1 for value in (chat, mode) if value)
        if target_count != 1:
            raise ValueError("Нужно указать ровно один параметр: chat или mode")

        if clear:
            if chat:
                self.instruction_manager.clear_chat_instruction(chat)
            else:
                self.instruction_manager.clear_mode_instruction(mode)
            return

        if instruction is None or not instruction.strip():
            raise ValueError("Не передан текст инструкции")

        if chat:
            self.instruction_manager.set_chat_instruction(chat, instruction)
        else:
            self.instruction_manager.set_mode_instruction(mode, instruction)

    async def get_stats(self) -> None:
        """Получение статистики"""
        print("📊 Статистика системы...")

        # Статистика по сообщениям
        if self.mcp:
            try:
                self.mcp.collection = self.mcp.chroma_client.get_collection(
                    "telegram_messages"
                )
                message_count = self.mcp.collection.count()
                print(f"📚 Сообщений в индексе: {message_count}")
            except:
                print("📚 Сообщений в индексе: 0")
        else:
            print("📚 MCP сервер недоступен")

        # Статистика по файлам
        chats_path = Path("chats")
        if chats_path.exists():
            json_files = list(chats_path.glob("**/*.json"))
            print(f"📁 JSON файлов: {len(json_files)}")
        else:
            print("📁 Директория chats не найдена")

        md_files = (
            list(Path("summaries").glob("*.md")) if Path("summaries").exists() else []
        )
        print(f"📄 MD файлов: {len(md_files)}")

    async def extract_messages(
        self,
        dry_run: bool = False,
        filter_by_date: bool = True,
        chat_filter: Optional[str] = None,
        input_dir: str = "input",
        chats_dir: str = "chats",
    ) -> None:
        """Извлечение новых сообщений из input в chats"""
        print("📥 Извлечение новых сообщений...")

        # Обновляем пути в экстракторе
        self.message_extractor.input_dir = Path(input_dir)
        self.message_extractor.chats_dir = Path(chats_dir)

        # Выполняем извлечение
        self.message_extractor.extract_all_messages(
            dry_run=dry_run, filter_by_date=filter_by_date, chat_filter=chat_filter
        )

        # Выводим статистику
        self.message_extractor.print_stats()

    async def deduplicate_messages(self, chats_dir: str = "chats") -> None:
        """Удаление дубликатов сообщений"""
        print("🧹 Удаление дубликатов сообщений...")

        # Обновляем путь в дедупликаторе
        self.message_deduplicator.chats_dir = Path(chats_dir)

        # Выполняем дедупликацию
        self.message_deduplicator.deduplicate_all_chats()

        # Выводим статистику
        self.message_deduplicator.print_stats()

    async def stop_indexing(self) -> None:
        """Остановка всех процессов индексации"""
        print("🛑 Остановка процессов индексации...")
        ProcessManager.stop_all_indexing()


async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Telegram Dump Manager")
    parser.add_argument(
        "command",
        choices=[
            "check",
            "stats",
            "extract-messages",
            "deduplicate",
            "stop-indexing",
        ],
        help="Команда для выполнения",
    )
    parser.add_argument("--query", "-q", help="Поисковый запрос")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Лимит результатов")
    parser.add_argument(
        "--max-files", type=int, help="Максимальное количество файлов для обработки"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Размер батча для индексации"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Режим тестирования без изменений"
    )
    parser.add_argument(
        "--no-date-filter", action="store_true", help="Отключить фильтрацию по дате"
    )
    parser.add_argument("--chat", help="Обработать только указанный чат")
    parser.add_argument("--input-dir", default="input", help="Путь к директории input")
    parser.add_argument("--chats-dir", default="chats", help="Путь к директории chats")

    args = parser.parse_args()

    async with TelegramDumpManager() as manager:
        if args.command == "check":
            await manager.check_system()

        elif args.command == "stats":
            await manager.get_stats()

        elif args.command == "extract-messages":
            await manager.extract_messages(
                dry_run=args.dry_run,
                filter_by_date=not args.no_date_filter,
                chat_filter=args.chat,
                input_dir=args.input_dir,
                chats_dir=args.chats_dir,
            )

        elif args.command == "deduplicate":
            await manager.deduplicate_messages(args.chats_dir)

        elif args.command == "stop-indexing":
            await manager.stop_indexing()


if __name__ == "__main__":
    asyncio.run(main())
