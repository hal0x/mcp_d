#!/usr/bin/env python3
"""
Система континуальной агрегации с скользящим окном и инкрементальной саммаризацией

Стратегия:
1. Свежие сообщения (0-30 дней) - детальное хранение
2. Средние (30-90 дней) - группировка по дням с саммаризацией
3. Старые (90-180 дней) - группировка по неделям с саммаризацией
4. Архивные (>180 дней) - группировка по месяцам с саммаризацией

Обратная совместимость:
- Использует существующий indexing_progress для отслеживания
- Создает отдельную коллекцию для агрегированных саммари
- Не удаляет оригинальные данные, только дополняет
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.lmstudio_client import LMStudioEmbeddingClient

logger = logging.getLogger(__name__)


class TimeWindow:
    """Определение временного окна для агрегации"""

    def __init__(
        self,
        name: str,
        age_days_min: int,
        age_days_max: int,
        group_by: str,  # 'hour', 'day', 'week', 'month'
        keep_original: bool = True,
        summarize: bool = False,
    ):
        self.name = name
        self.age_days_min = age_days_min
        self.age_days_max = age_days_max
        self.group_by = group_by
        self.keep_original = keep_original
        self.summarize = summarize

    def matches(self, age_days: int) -> bool:
        """Проверяет, попадает ли возраст в это окно"""
        return self.age_days_min <= age_days < self.age_days_max

    def __repr__(self):
        return f"<TimeWindow {self.name}: {self.age_days_min}-{self.age_days_max} days, group_by={self.group_by}>"


# Предустановленные стратегии
CONSERVATIVE_STRATEGY = [
    TimeWindow("fresh", 0, 30, "hour", keep_original=True, summarize=False),
    TimeWindow("recent", 30, 90, "day", keep_original=False, summarize=True),
    TimeWindow("medium", 90, 180, "week", keep_original=False, summarize=True),
    TimeWindow("old", 180, 365, "month", keep_original=False, summarize=True),
    TimeWindow("archive", 365, 10000, "month", keep_original=False, summarize=True),
]

AGGRESSIVE_STRATEGY = [
    TimeWindow("fresh", 0, 7, "hour", keep_original=True, summarize=False),
    TimeWindow("recent", 7, 30, "day", keep_original=False, summarize=True),
    TimeWindow("medium", 30, 90, "week", keep_original=False, summarize=True),
    TimeWindow("old", 90, 10000, "month", keep_original=False, summarize=True),
]

MINIMAL_STRATEGY = [
    TimeWindow("fresh", 0, 14, "hour", keep_original=True, summarize=False),
    TimeWindow("all_old", 14, 10000, "week", keep_original=False, summarize=True),
]


class AggregationState:
    """Состояние агрегации для чата"""

    def __init__(self, chat_name: str):
        self.chat_name = chat_name
        self.last_aggregation_time: Optional[datetime] = None
        self.window_boundaries: Dict[str, datetime] = {}  # window_name -> last_date
        self.aggregated_blocks: List[
            Dict[str, Any]
        ] = []  # Список агрегированных блоков
        self.summary_cache: Dict[str, str] = {}  # block_id -> summary

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация состояния"""
        return {
            "chat_name": self.chat_name,
            "last_aggregation_time": self.last_aggregation_time.isoformat()
            if self.last_aggregation_time
            else None,
            "window_boundaries": {
                k: v.isoformat() for k, v in self.window_boundaries.items()
            },
            "aggregated_blocks": self.aggregated_blocks,
            "summary_cache": self.summary_cache,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregationState":
        """Десериализация состояния"""
        from ..utils.datetime_utils import parse_datetime_utc

        state = cls(data["chat_name"])
        if data.get("last_aggregation_time"):
            state.last_aggregation_time = parse_datetime_utc(
                data["last_aggregation_time"], use_zoneinfo=True
            ) or datetime.now(ZoneInfo("UTC"))
        state.window_boundaries = {
            k: parse_datetime_utc(v, use_zoneinfo=True) or datetime.now(ZoneInfo("UTC"))
            for k, v in data.get("window_boundaries", {}).items()
        }
        state.aggregated_blocks = data.get("aggregated_blocks", [])
        state.summary_cache = data.get("summary_cache", {})
        return state


class RollingWindowAggregator:
    """
    Система континуальной агрегации с скользящим окном

    Основные возможности:
    1. Инкрементальная обработка - только новые сообщения
    2. Автоматическая миграция сообщений между окнами
    3. Групповая саммаризация для минимизации запросов к Ollama
    4. Сохранение состояния для обратной совместимости
    """

    def __init__(
        self,
        chats_dir: Path = Path("chats"),
        state_dir: Path = Path("aggregation_state"),
        strategy: List[TimeWindow] = None,
        embedding_client: Optional[LMStudioEmbeddingClient] = None,
        batch_size: int = 50,  # Сколько сообщений группировать для одной саммаризации
    ):
        self.chats_dir = chats_dir
        self.state_dir = state_dir
        self.state_dir.mkdir(exist_ok=True)

        self.strategy = strategy or CONSERVATIVE_STRATEGY
        self.embedding_client = embedding_client or LMStudioEmbeddingClient()
        self.batch_size = batch_size

        logger.info(
            f"Инициализирован RollingWindowAggregator со стратегией: {len(self.strategy)} окон"
        )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсит дату из строки (использует общую утилиту)."""
        from ..utils.datetime_utils import parse_datetime_utc

        return parse_datetime_utc(date_str, return_none_on_error=True, use_zoneinfo=True)

    def _load_state(self, chat_name: str) -> AggregationState:
        """Загружает состояние агрегации для чата"""
        from ..utils.state_manager import StateManager

        manager = StateManager(self.state_dir)
        return manager.load_state(
            chat_name,
            AggregationState,
            default_factory=lambda: AggregationState(chat_name),
        )

    def _save_state(self, state: AggregationState):
        """Сохраняет состояние агрегации"""
        from ..utils.state_manager import StateManager

        manager = StateManager(self.state_dir)
        manager.save_state(state)

    def _load_messages(self, chat_file: Path) -> List[Dict[str, Any]]:
        """Загружает сообщения из файла чата"""
        messages = []

        try:
            with open(chat_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            message = json.loads(line)
                            messages.append(message)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Ошибка загрузки сообщений из {chat_file}: {e}")

        return messages

    def _group_messages_by_window(
        self, messages: List[Dict[str, Any]], current_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Группирует сообщения по временным окнам"""
        grouped = defaultdict(list)

        for msg in messages:
            if "date_utc" not in msg:
                continue

            msg_date = self._parse_date(msg["date_utc"])
            if not msg_date:
                continue

            age_days = (current_date - msg_date).days

            # Находим подходящее окно
            for window in self.strategy:
                if window.matches(age_days):
                    grouped[window.name].append(msg)
                    break

        return grouped

    def _create_aggregation_key(self, msg_date: datetime, group_by: str) -> str:
        """Создает ключ агрегации на основе даты и группировки"""
        if group_by == "hour":
            return msg_date.strftime("%Y-%m-%d-%H")
        elif group_by == "day":
            return msg_date.strftime("%Y-%m-%d")
        elif group_by == "week":
            # ISO неделя
            return f"{msg_date.year}-W{msg_date.isocalendar()[1]:02d}"
        elif group_by == "month":
            return msg_date.strftime("%Y-%m")
        else:
            return msg_date.strftime("%Y-%m-%d")

    def _group_messages_for_aggregation(
        self, messages: List[Dict[str, Any]], window: TimeWindow
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Группирует сообщения для агрегации по временным блокам"""
        blocks = defaultdict(list)

        for msg in messages:
            if "date_utc" not in msg:
                continue

            msg_date = self._parse_date(msg["date_utc"])
            if not msg_date:
                continue

            block_key = self._create_aggregation_key(msg_date, window.group_by)
            blocks[block_key].append(msg)

        return blocks

    async def _summarize_message_block(
        self, messages: List[Dict[str, Any]], block_id: str, window: TimeWindow
    ) -> str:
        """
        Создает саммаризацию блока сообщений

        Группирует сообщения в батчи для эффективной саммаризации
        """
        if not messages:
            return ""

        # Сортируем по времени
        sorted_messages = sorted(
            messages,
            key=lambda m: self._parse_date(m.get("date_utc", "")) or datetime.min,
        )

        # Формируем текст для саммаризации
        text_parts = []
        text_parts.append(f"=== Блок: {block_id} ({len(messages)} сообщений) ===\n")

        for msg in sorted_messages[: self.batch_size]:  # Ограничиваем размер батча
            date = msg.get("date_utc", "Unknown")
            sender = msg.get("from", {})
            sender_name = (
                sender.get("display", "Unknown")
                if isinstance(sender, dict)
                else "Unknown"
            )
            text = msg.get("text", "")

            if text:
                text_parts.append(f"[{date}] {sender_name}: {text[:200]}")

        if len(messages) > self.batch_size:
            text_parts.append(
                f"\n... и еще {len(messages) - self.batch_size} сообщений"
            )

        conversation_text = "\n".join(text_parts)

        # Создаем промпт для саммаризации
        prompt = f"""Создай краткую саммаризацию следующего блока сообщений.
Укажи:
1. Основные темы обсуждения
2. Ключевые решения или выводы
3. Важные упоминания (люди, проекты, события)

Сообщения:
{conversation_text}

Саммаризация (2-3 предложения):"""

        try:
            async with self.embedding_client:
                summary = await self.embedding_client.generate_summary(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=300,
                )
            return summary.strip()
        except Exception as e:
            logger.error(f"Ошибка саммаризации блока {block_id}: {e}")
            # Fallback: простое описание
            return f"Блок из {len(messages)} сообщений за период {block_id}"

    async def aggregate_chat(
        self, chat_name: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Выполняет агрегацию для одного чата

        Returns:
            Статистика агрегации
        """
        logger.info(f"Начало агрегации чата: {chat_name}")

        # Загружаем состояние
        state = self._load_state(chat_name)

        # Находим файл чата
        chat_file = self.chats_dir / chat_name / "unknown.json"
        if not chat_file.exists():
            logger.warning(f"Файл чата не найден: {chat_file}")
            return {"error": "chat_file_not_found"}

        # Загружаем сообщения
        messages = self._load_messages(chat_file)
        if not messages:
            logger.warning(f"Нет сообщений в чате {chat_name}")
            return {"messages_count": 0}

        current_date = datetime.now(datetime.now().astimezone().tzinfo)

        # Группируем по окнам
        windowed_messages = self._group_messages_by_window(messages, current_date)

        stats = {
            "chat_name": chat_name,
            "total_messages": len(messages),
            "windows": {},
            "summaries_created": 0,
            "blocks_aggregated": 0,
        }

        # Обрабатываем каждое окно
        for window in self.strategy:
            window_messages = windowed_messages.get(window.name, [])

            if not window_messages:
                continue

            logger.info(f"Окно '{window.name}': {len(window_messages)} сообщений")

            stats["windows"][window.name] = {
                "messages_count": len(window_messages),
                "keep_original": window.keep_original,
                "summarize": window.summarize,
            }

            if window.summarize:
                # Группируем для агрегации
                blocks = self._group_messages_for_aggregation(window_messages, window)

                logger.info(f"Создано {len(blocks)} блоков для окна '{window.name}'")
                stats["windows"][window.name]["blocks_count"] = len(blocks)

                # Саммаризируем каждый блок
                for block_id, block_messages in blocks.items():
                    # Проверяем, есть ли уже саммаризация
                    cache_key = f"{window.name}_{block_id}"

                    if cache_key in state.summary_cache:
                        logger.debug(
                            f"Используется кэшированная саммаризация для {cache_key}"
                        )
                        continue

                    if not dry_run:
                        summary = await self._summarize_message_block(
                            block_messages, block_id, window
                        )

                        # Сохраняем в кэш
                        state.summary_cache[cache_key] = summary
                        stats["summaries_created"] += 1

                        # Добавляем в список блоков
                        state.aggregated_blocks.append(
                            {
                                "block_id": cache_key,
                                "window": window.name,
                                "time_key": block_id,
                                "message_count": len(block_messages),
                                "summary": summary,
                                "created_at": current_date.isoformat(),
                            }
                        )

                        stats["blocks_aggregated"] += 1

                        # Небольшая задержка между запросами
                        await asyncio.sleep(0.5)

        # Обновляем состояние
        state.last_aggregation_time = current_date

        if not dry_run:
            self._save_state(state)

        logger.info(
            f"Агрегация завершена: {stats['summaries_created']} саммаризаций, "
            f"{stats['blocks_aggregated']} блоков"
        )

        return stats

    async def aggregate_all_chats(
        self, dry_run: bool = False, max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Агрегирует все чаты

        Args:
            dry_run: Тестовый запуск без сохранения
            max_concurrent: Максимум параллельных агрегаций

        Returns:
            Общая статистика
        """
        # Находим все чаты
        chat_dirs = [d for d in self.chats_dir.iterdir() if d.is_dir()]

        logger.info(f"Найдено {len(chat_dirs)} чатов для агрегации")

        total_stats = {
            "total_chats": len(chat_dirs),
            "processed_chats": 0,
            "total_summaries": 0,
            "total_blocks": 0,
            "chats": {},
        }

        # Обрабатываем чаты с ограничением параллелизма
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_chat(chat_dir):
            async with semaphore:
                chat_name = chat_dir.name
                try:
                    stats = await self.aggregate_chat(chat_name, dry_run)
                    return chat_name, stats
                except Exception as e:
                    logger.error(f"Ошибка агрегации чата {chat_name}: {e}")
                    return chat_name, {"error": str(e)}

        tasks = [process_chat(chat_dir) for chat_dir in chat_dirs]
        results = await asyncio.gather(*tasks)

        # Собираем статистику
        for chat_name, stats in results:
            if "error" not in stats:
                total_stats["processed_chats"] += 1
                total_stats["total_summaries"] += stats.get("summaries_created", 0)
                total_stats["total_blocks"] += stats.get("blocks_aggregated", 0)

            total_stats["chats"][chat_name] = stats

        return total_stats

    def get_aggregation_report(self, chat_name: str) -> Dict[str, Any]:
        """Получает отчет об агрегации для чата"""
        state = self._load_state(chat_name)

        return {
            "chat_name": chat_name,
            "last_aggregation": state.last_aggregation_time.isoformat()
            if state.last_aggregation_time
            else None,
            "total_blocks": len(state.aggregated_blocks),
            "total_summaries": len(state.summary_cache),
            "window_boundaries": {
                k: v.isoformat() for k, v in state.window_boundaries.items()
            },
            "recent_blocks": state.aggregated_blocks[-10:]
            if state.aggregated_blocks
            else [],
        }


async def main():
    """Пример использования"""
    logging.basicConfig(level=logging.INFO)

    aggregator = RollingWindowAggregator(
        chats_dir=Path("chats"),
        strategy=CONSERVATIVE_STRATEGY,
        batch_size=30,
    )

    # Тестовый запуск
    stats = await aggregator.aggregate_all_chats(dry_run=True, max_concurrent=2)

    print("\nСтатистика агрегации:")
    print(f"Обработано чатов: {stats['processed_chats']}/{stats['total_chats']}")
    print(f"Создано саммаризаций: {stats['total_summaries']}")
    print(f"Агрегировано блоков: {stats['total_blocks']}")


if __name__ == "__main__":
    asyncio.run(main())
