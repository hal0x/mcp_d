#!/usr/bin/env python3
"""
Скрипт для анализа дат сообщений в чатах.
Подсчитывает количество сообщений по временным периодам.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageDateAnalyzer:
    """Класс для анализа дат сообщений."""

    def __init__(self, chats_dir: str = "chats"):
        self.chats_dir = Path(chats_dir)

        # Определяем временные границы
        now = datetime.now(timezone.utc)
        self.week_ago = now - timedelta(days=7)
        self.month_ago = now - timedelta(days=30)
        self.year_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)

        self.stats = {
            "total_messages": 0,
            "messages_last_week": 0,
            "messages_last_month": 0,
            "messages_this_year": 0,
            "messages_without_date": 0,
            "chats_processed": 0,
            "errors": 0,
        }

    def parse_date(self, date_str: str) -> datetime:
        """Парсит дату из строки в формате ISO 8601."""
        try:
            # Заменяем Z на +00:00 для UTC
            if date_str.endswith("Z"):
                date_str = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None

    def analyze_chat(self, chat_dir: Path) -> dict:
        """Анализирует даты сообщений в одном чате."""
        chat_stats = {
            "total_messages": 0,
            "messages_last_week": 0,
            "messages_last_month": 0,
            "messages_this_year": 0,
            "messages_without_date": 0,
            "errors": 0,
        }

        # Ищем JSON файлы в чате
        json_files = []
        for pattern in ["unknown.json", "result.json"]:
            json_file = chat_dir / pattern
            if json_file.exists():
                json_files.append(json_file)

        if not json_files:
            logger.warning(f"Нет JSON файлов в чате: {chat_dir}")
            return chat_stats

        # Обрабатываем каждый JSON файл
        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            message = json.loads(line)
                            chat_stats["total_messages"] += 1

                            if "date_utc" in message:
                                msg_date = self.parse_date(message["date_utc"])
                                if msg_date:
                                    if msg_date >= self.week_ago:
                                        chat_stats["messages_last_week"] += 1
                                    if msg_date >= self.month_ago:
                                        chat_stats["messages_last_month"] += 1
                                    if msg_date >= self.year_start:
                                        chat_stats["messages_this_year"] += 1
                                else:
                                    chat_stats["messages_without_date"] += 1
                            else:
                                chat_stats["messages_without_date"] += 1

                        except json.JSONDecodeError as e:
                            logger.warning(f"Ошибка JSON в {json_file}: {e}")
                            chat_stats["errors"] += 1

            except Exception as e:
                logger.error(f"Ошибка чтения файла {json_file}: {e}")
                chat_stats["errors"] += 1

        return chat_stats

    def analyze_all_chats(self) -> dict:
        """Анализирует даты сообщений во всех чатах."""
        if not self.chats_dir.exists():
            logger.error(f"Директория {self.chats_dir} не существует")
            return self.stats

        chat_dirs = [d for d in self.chats_dir.iterdir() if d.is_dir()]

        logger.info(f"Анализируем {len(chat_dirs)} чатов")
        logger.info("Периоды анализа:")
        logger.info(
            f"  - Последняя неделя: с {self.week_ago.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        logger.info(
            f"  - Последний месяц: с {self.month_ago.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        logger.info(
            f"  - Текущий год: с {self.year_start.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        for chat_dir in chat_dirs:
            try:
                chat_stats = self.analyze_chat(chat_dir)

                # Обновляем общую статистику
                self.stats["chats_processed"] += 1
                self.stats["total_messages"] += chat_stats["total_messages"]
                self.stats["messages_last_week"] += chat_stats["messages_last_week"]
                self.stats["messages_last_month"] += chat_stats["messages_last_month"]
                self.stats["messages_this_year"] += chat_stats["messages_this_year"]
                self.stats["messages_without_date"] += chat_stats[
                    "messages_without_date"
                ]
                self.stats["errors"] += chat_stats["errors"]

                # Выводим статистику для чатов с активностью за неделю
                if chat_stats["messages_last_week"] > 0:
                    logger.info(
                        f"{chat_dir.name}: {chat_stats['messages_last_week']} сообщений за неделю"
                    )

            except Exception as e:
                logger.error(f"Ошибка обработки чата {chat_dir}: {e}")
                self.stats["errors"] += 1

        return self.stats

    def print_stats(self):
        """Выводит статистику анализа."""
        print("\n" + "=" * 70)
        print("АНАЛИЗ ДАТ СООБЩЕНИЙ")
        print("=" * 70)
        print(f"Обработано чатов: {self.stats['chats_processed']}")
        print(f"Всего сообщений: {self.stats['total_messages']}")
        print(f"Сообщений за последнюю неделю: {self.stats['messages_last_week']}")
        print(f"Сообщений за последний месяц: {self.stats['messages_last_month']}")
        print(f"Сообщений за текущий год: {self.stats['messages_this_year']}")
        print(f"Сообщений без даты: {self.stats['messages_without_date']}")
        print(f"Ошибок: {self.stats['errors']}")

        if self.stats["total_messages"] > 0:
            week_percent = (
                self.stats["messages_last_week"] / self.stats["total_messages"]
            ) * 100
            month_percent = (
                self.stats["messages_last_month"] / self.stats["total_messages"]
            ) * 100
            year_percent = (
                self.stats["messages_this_year"] / self.stats["total_messages"]
            ) * 100

            print("\nПроценты:")
            print(f"  - За неделю: {week_percent:.2f}%")
            print(f"  - За месяц: {month_percent:.2f}%")
            print(f"  - За год: {year_percent:.2f}%")

        print("=" * 70)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Анализ дат сообщений в чатах")
    parser.add_argument(
        "--chats-dir", default="chats", help="Путь к директории с чатами"
    )
    parser.add_argument("--chat", help="Анализировать только указанный чат")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = MessageDateAnalyzer(args.chats_dir)

    if args.chat:
        # Анализ одного чата
        chat_dir = Path(args.chats_dir) / args.chat
        if not chat_dir.exists():
            logger.error(f"Чат {args.chat} не найден в {args.chats_dir}")
            return 1

        logger.info(f"Анализируем чат: {args.chat}")
        chat_stats = analyzer.analyze_chat(chat_dir)

        print(f"\nРезультаты для чата '{args.chat}':")
        print(f"Всего сообщений: {chat_stats['total_messages']}")
        print(f"За последнюю неделю: {chat_stats['messages_last_week']}")
        print(f"За последний месяц: {chat_stats['messages_last_month']}")
        print(f"За текущий год: {chat_stats['messages_this_year']}")
        print(f"Без даты: {chat_stats['messages_without_date']}")

    else:
        # Анализ всех чатов
        analyzer.analyze_all_chats()
        analyzer.print_stats()

    return 0


if __name__ == "__main__":
    exit(main())
