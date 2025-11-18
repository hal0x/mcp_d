#!/usr/bin/env python3
"""
Умная система континуальной агрегации с учетом связанности сообщений

Ключевые отличия от простой rolling_window:
1. Группировка по связанности (сессии) для чатов, по дням для каналов
2. Специальное окно NOW для самых актуальных сообщений
3. Контекстная саммаризация с учетом истории
4. Интеграция с SessionSegmenter и ContextManager
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.datetime_utils import format_datetime_display

from ..core.lmstudio_client import LMStudioEmbeddingClient
from .context_manager import ContextManager
from .session_segmentation import SessionSegmenter
from .session_summarizer import SessionSummarizer

logger = logging.getLogger(__name__)


class SmartTimeWindow:
    """Умное временное окно с учетом типа группировки"""

    def __init__(
        self,
        name: str,
        age_days_min: int,
        age_days_max: int,
        group_strategy: str,  # 'session', 'day', 'week', 'month'
        keep_original: bool = True,
        summarize: bool = False,
        use_context: bool = False,  # Использовать контекст из предыдущих окон
    ):
        self.name = name
        self.age_days_min = age_days_min
        self.age_days_max = age_days_max
        self.group_strategy = group_strategy
        self.keep_original = keep_original
        self.summarize = summarize
        self.use_context = use_context

    def matches(self, age_days: int) -> bool:
        """Проверяет, попадает ли возраст в это окно"""
        return self.age_days_min <= age_days < self.age_days_max

    def __repr__(self):
        return (
            f"<SmartTimeWindow {self.name}: {self.age_days_min}-{self.age_days_max} days, "
            f"strategy={self.group_strategy}, context={self.use_context}>"
        )


# Умная стратегия с учетом связанности
SMART_STRATEGY = [
    SmartTimeWindow(
        "now", 0, 1, "session", keep_original=True, summarize=True, use_context=True
    ),
    SmartTimeWindow(
        "fresh",
        1,
        14,
        "session",
        keep_original=True,
        summarize=False,
        use_context=False,
    ),
    SmartTimeWindow(
        "recent", 14, 30, "week", keep_original=False, summarize=True, use_context=False
    ),
    SmartTimeWindow(
        "old",
        30,
        10000,
        "month",
        keep_original=False,
        summarize=True,
        use_context=False,
    ),
]

# Стратегия для каналов (без сессий, только временная группировка)
CHANNEL_STRATEGY = [
    SmartTimeWindow(
        "now", 0, 1, "day", keep_original=True, summarize=True, use_context=True
    ),
    SmartTimeWindow(
        "fresh", 1, 14, "day", keep_original=True, summarize=False, use_context=False
    ),
    SmartTimeWindow(
        "recent", 14, 30, "week", keep_original=False, summarize=True, use_context=False
    ),
    SmartTimeWindow(
        "old",
        30,
        10000,
        "month",
        keep_original=False,
        summarize=True,
        use_context=False,
    ),
]


class SmartAggregationState:
    """Состояние умной агрегации"""

    def __init__(self, chat_name: str):
        self.chat_name = chat_name
        self.chat_type: Optional[str] = None  # 'channel', 'group', 'chat'
        self.last_aggregation_time: Optional[datetime] = None
        self.window_summaries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.now_summary_file: Optional[str] = None  # Путь к файлу NOW саммаризации
        self.context_cache: Dict[str, Any] = {}  # Кэш контекста для NOW окна

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "last_aggregation_time": self.last_aggregation_time.isoformat()
            if self.last_aggregation_time
            else None,
            "window_summaries": dict(self.window_summaries.items()),
            "now_summary_file": self.now_summary_file,
            "context_cache": self.context_cache,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartAggregationState":
        """Десериализация"""
        from ..utils.datetime_utils import parse_datetime_utc

        state = cls(data["chat_name"])
        state.chat_type = data.get("chat_type")
        if data.get("last_aggregation_time"):
            state.last_aggregation_time = parse_datetime_utc(
                data["last_aggregation_time"], use_zoneinfo=True
            ) or datetime.now(ZoneInfo("UTC"))
        state.window_summaries = defaultdict(list, data.get("window_summaries", {}))
        state.now_summary_file = data.get("now_summary_file")
        state.context_cache = data.get("context_cache", {})
        return state


class SmartRollingAggregator:
    """
    Умная система континуальной агрегации

    Особенности:
    1. Определяет тип чата (канал/группа/чат)
    2. Использует SessionSegmenter для умной группировки
    3. Специальное окно NOW с контекстом
    4. Сохраняет NOW саммаризацию в отдельный файл
    """

    def __init__(
        self,
        chats_dir: Path = Path("chats"),
        state_dir: Path = Path("artifacts/smart_aggregation_state"),
        summaries_dir: Path = Path("artifacts/reports"),
        now_summaries_dir: Path = Path("artifacts/now_summaries"),
        embedding_client: Optional[LMStudioEmbeddingClient] = None,
        use_smart_strategy: bool = True,
    ):
        self.chats_dir = chats_dir
        self.state_dir = state_dir
        self.summaries_dir = summaries_dir
        self.now_summaries_dir = now_summaries_dir

        # Создаем директории
        self.state_dir.mkdir(exist_ok=True)
        self.now_summaries_dir.mkdir(exist_ok=True)

        self.embedding_client = embedding_client or LMStudioEmbeddingClient()
        self.use_smart_strategy = use_smart_strategy

        # Компоненты для умной обработки
        self.session_segmenter = SessionSegmenter(
            gap_minutes=240,  # Увеличиваем разрыв до 4 часов для максимальной связанности
            max_session_hours=24,  # Увеличиваем максимальную длительность до 24 часов
        )
        self.context_manager = ContextManager(summaries_dir)
        self.session_summarizer = SessionSummarizer(
            embedding_client=self.embedding_client,
            summaries_dir=summaries_dir,
        )

        logger.info("Инициализирован SmartRollingAggregator")

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсит дату (использует общую утилиту)."""
        from ..utils.datetime_utils import parse_datetime_utc

        return parse_datetime_utc(date_str, return_none_on_error=True, use_zoneinfo=True)

    def _detect_chat_type(self, messages: List[Dict[str, Any]]) -> str:
        """
        Определяет тип чата на основе сообщений

        Returns:
            'channel', 'group', или 'chat'
        """
        if not messages:
            return "chat"

        # Проверяем наличие авторов
        messages_with_author = sum(1 for m in messages if m.get("from"))

        # Если большинство сообщений без автора - это канал
        if messages_with_author < len(messages) * 0.3:
            return "channel"

        # Проверяем количество уникальных авторов
        unique_authors = set()
        for msg in messages:
            if msg.get("from"):
                author_id = (
                    msg["from"].get("id") if isinstance(msg["from"], dict) else None
                )
                if author_id:
                    unique_authors.add(author_id)

        # Если много авторов - группа, иначе - чат
        return "group" if len(unique_authors) > 5 else "chat"

    def _load_state(self, chat_name: str) -> SmartAggregationState:
        """Загружает состояние"""
        from ..utils.state_manager import StateManager

        manager = StateManager(self.state_dir)
        return manager.load_state(
            chat_name,
            SmartAggregationState,
            default_factory=lambda: SmartAggregationState(chat_name),
        )

    def _save_state(self, state: SmartAggregationState):
        """Сохраняет состояние"""
        from ..utils.state_manager import StateManager

        manager = StateManager(self.state_dir)
        manager.save_state(state)

    def _load_messages(self, chat_file: Path) -> List[Dict[str, Any]]:
        """Загружает сообщения из файла"""
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
        self,
        messages: List[Dict[str, Any]],
        current_date: datetime,
        strategy: List[SmartTimeWindow],
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
            for window in strategy:
                if window.matches(age_days):
                    grouped[window.name].append(msg)
                    break

        return grouped

    def _group_by_sessions(
        self, messages: List[Dict[str, Any]], chat_name: str
    ) -> List[Dict[str, Any]]:
        """Группирует сообщения по сессиям используя SessionSegmenter"""
        return self.session_segmenter.segment_messages(messages, chat_name)

    def _group_by_smart_strategy(
        self, windowed_messages: Dict[str, List[Dict[str, Any]]], chat_name: str
    ) -> List[Dict[str, Any]]:
        """
        Умная стратегия группировки с учетом окон

        NOW/FRESH: группировка по дням (минимум 10 сообщений)
        RECENT: группировка по неделям (минимум 20 сообщений)
        OLD: группировка по месяцам (минимум 50 сообщений)
        """
        sessions = []
        session_counter = 1

        # Обрабатываем каждое окно отдельно в хронологическом порядке (от старых к новым)
        for window_name in ["old", "recent", "fresh", "now"]:
            window_messages = windowed_messages.get(window_name, [])
            if not window_messages:
                continue

            logger.info(
                f"Умная группировка окна '{window_name}': {len(window_messages)} сообщений"
            )

            if window_name in ["now", "fresh"]:
                # Для актуальных окон группируем по дням
                window_sessions = self._group_by_days_with_minimum(
                    window_messages, chat_name, min_messages_per_day=10
                )
            elif window_name == "recent":
                # Для недавних сообщений группируем по неделям
                window_sessions = self._group_by_weeks_with_minimum(
                    window_messages, chat_name, min_messages_per_week=20
                )
            else:  # old
                # Для старых сообщений группируем по месяцам (каждый месяц = отдельная сессия)
                window_sessions = self._group_by_months_with_minimum(
                    window_messages,
                    chat_name,
                    min_messages_per_month=50,
                    force_split_by_month=True,
                )

            # Добавляем информацию о окне к каждой сессии
            for session in window_sessions:
                session["window"] = window_name
                session["chat"] = chat_name
                session["session_id"] = f"{chat_name}-S{session_counter:04d}"
                sessions.append(session)
                session_counter += 1

        logger.info(f"Умная группировка создала {len(sessions)} сессий")
        return sessions

    def _group_by_days_with_minimum(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        min_messages_per_day: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Группирует сообщения по дням с минимальным количеством сообщений

        Объединяет маленькие дни в более крупные сессии для эффективности
        """
        if not messages:
            return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(
            messages, key=lambda x: self._parse_date(x.get("date_utc", ""))
        )

        # Группируем по дням
        day_groups = defaultdict(list)
        for msg in sorted_messages:
            msg_date = self._parse_date(msg.get("date_utc", ""))
            if msg_date:
                day_key = msg_date.strftime("%Y-%m-%d")
                day_groups[day_key].append(msg)

        # Объединяем маленькие дни в более крупные сессии
        sessions = []
        current_session_messages = []
        current_session_days = []

        for day_key in sorted(day_groups.keys()):
            day_messages = day_groups[day_key]

            # Если текущая сессия + этот день >= min_messages_per_day, создаем сессию
            if (
                len(current_session_messages) + len(day_messages)
                >= min_messages_per_day
            ):
                if current_session_messages:
                    # Создаем сессию из накопленных сообщений
                    session = self._create_session_from_messages(
                        current_session_messages,
                        chat_name,
                        f"{current_session_days[0]} to {current_session_days[-1]}",
                    )
                    sessions.append(session)

                # Начинаем новую сессию с текущего дня
                current_session_messages = day_messages.copy()
                current_session_days = [day_key]
            else:
                # Добавляем к текущей сессии
                current_session_messages.extend(day_messages)
                current_session_days.append(day_key)

        # Добавляем последнюю сессию
        if current_session_messages:
            session = self._create_session_from_messages(
                current_session_messages,
                chat_name,
                f"{current_session_days[0]} to {current_session_days[-1]}",
            )
            sessions.append(session)

        logger.info(
            f"Создано {len(sessions)} сессий по дням (минимум {min_messages_per_day} сообщений)"
        )
        return sessions

    def _group_by_weeks_with_minimum(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        min_messages_per_week: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Группирует сообщения по неделям с минимальным количеством сообщений
        """
        if not messages:
            return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(
            messages, key=lambda x: self._parse_date(x.get("date_utc", ""))
        )

        # Группируем по неделям (ISO недели)
        week_groups = defaultdict(list)
        for msg in sorted_messages:
            msg_date = self._parse_date(msg.get("date_utc", ""))
            if msg_date:
                # Получаем год и номер недели
                year, week, _ = msg_date.isocalendar()
                week_key = f"{year}-W{week:02d}"
                week_groups[week_key].append(msg)

        # Объединяем маленькие недели в более крупные сессии
        sessions = []
        current_session_messages = []
        current_session_weeks = []

        for week_key in sorted(week_groups.keys()):
            week_messages = week_groups[week_key]

            # Если текущая сессия + эта неделя >= min_messages_per_week, создаем сессию
            if (
                len(current_session_messages) + len(week_messages)
                >= min_messages_per_week
            ):
                if current_session_messages:
                    # Создаем сессию из накопленных сообщений
                    session = self._create_session_from_messages(
                        current_session_messages,
                        chat_name,
                        f"{current_session_weeks[0]} to {current_session_weeks[-1]}",
                    )
                    sessions.append(session)

                # Начинаем новую сессию с текущей недели
                current_session_messages = week_messages.copy()
                current_session_weeks = [week_key]
            else:
                # Добавляем к текущей сессии
                current_session_messages.extend(week_messages)
                current_session_weeks.append(week_key)

        # Добавляем последнюю сессию
        if current_session_messages:
            session = self._create_session_from_messages(
                current_session_messages,
                chat_name,
                f"{current_session_weeks[0]} to {current_session_weeks[-1]}",
            )
            sessions.append(session)

        logger.info(
            f"Создано {len(sessions)} сессий по неделям (минимум {min_messages_per_week} сообщений)"
        )
        return sessions

    def _group_by_months_with_minimum(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        min_messages_per_month: int = 50,
        force_split_by_month: bool = True,  # Новый параметр для принудительного разбиения
    ) -> List[Dict[str, Any]]:
        """
        Группирует сообщения по месяцам с минимальным количеством сообщений

        Args:
            messages: Список сообщений
            chat_name: Название чата
            min_messages_per_month: Минимальное количество сообщений для создания сессии
            force_split_by_month: Если True, каждый месяц создает отдельную сессию
        """
        if not messages:
            return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(
            messages, key=lambda x: self._parse_date(x.get("date_utc", ""))
        )

        # Группируем по месяцам
        month_groups = defaultdict(list)
        for msg in sorted_messages:
            msg_date = self._parse_date(msg.get("date_utc", ""))
            if msg_date:
                month_key = msg_date.strftime("%Y-%m")
                month_groups[month_key].append(msg)

        sessions = []

        if force_split_by_month:
            # Создаем отдельную сессию для каждого месяца
            for month_key in sorted(month_groups.keys()):
                month_messages = month_groups[month_key]

                # Создаем сессию для каждого месяца, даже если сообщений мало
                session = self._create_session_from_messages(
                    month_messages,
                    chat_name,
                    month_key,
                )
                sessions.append(session)

            logger.info(
                f"Создано {len(sessions)} сессий по месяцам (принудительное разбиение по месяцам)"
            )
        else:
            # Старая логика с объединением маленьких месяцев
            current_session_messages = []
            current_session_months = []

            for month_key in sorted(month_groups.keys()):
                month_messages = month_groups[month_key]

                # Если текущая сессия + этот месяц >= min_messages_per_month, создаем сессию
                if (
                    len(current_session_messages) + len(month_messages)
                    >= min_messages_per_month
                ):
                    if current_session_messages:
                        # Создаем сессию из накопленных сообщений
                        session = self._create_session_from_messages(
                            current_session_messages,
                            chat_name,
                            f"{current_session_months[0]} to {current_session_months[-1]}",
                        )
                        sessions.append(session)

                    # Начинаем новую сессию с текущего месяца
                    current_session_messages = month_messages.copy()
                    current_session_months = [month_key]
                else:
                    # Добавляем к текущей сессии
                    current_session_messages.extend(month_messages)
                    current_session_months.append(month_key)

            # Добавляем последнюю сессию
            if current_session_messages:
                session = self._create_session_from_messages(
                    current_session_messages,
                    chat_name,
                    f"{current_session_months[0]} to {current_session_months[-1]}",
                )
                sessions.append(session)

            logger.info(
                f"Создано {len(sessions)} сессий по месяцам (минимум {min_messages_per_month} сообщений)"
            )

        return sessions

    def _create_session_from_messages(
        self, messages: List[Dict[str, Any]], chat_name: str, time_range: str
    ) -> Dict[str, Any]:
        """Создает сессию из списка сообщений"""
        if not messages:
            return {}

        sorted_messages = sorted(
            messages, key=lambda x: self._parse_date(x.get("date_utc", ""))
        )

        return {
            "session_id": f"{chat_name}-DAY-{time_range}",
            "chat": chat_name,
            "messages": sorted_messages,
            "start_time": sorted_messages[0].get("date_utc"),
            "end_time": sorted_messages[-1].get("date_utc"),
            "message_count": len(sorted_messages),
            "time_range": time_range,
        }

    def _group_by_time_period(
        self, messages: List[Dict[str, Any]], period: str  # 'day', 'week', 'month'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Группирует сообщения по временным периодам"""
        blocks = defaultdict(list)

        for msg in messages:
            if "date_utc" not in msg:
                continue

            msg_date = self._parse_date(msg["date_utc"])
            if not msg_date:
                continue

            if period == "day":
                key = msg_date.strftime("%Y-%m-%d")
            elif period == "week":
                key = f"{msg_date.year}-W{msg_date.isocalendar()[1]:02d}"
            elif period == "month":
                key = msg_date.strftime("%Y-%m")
            else:
                key = msg_date.strftime("%Y-%m-%d")

            blocks[key].append(msg)

        return blocks

    async def _summarize_with_context(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        window_name: str,
        previous_context: Optional[str] = None,
    ) -> str:
        """
        Саммаризация с учетом контекста

        Для NOW окна использует контекст из предыдущих сообщений
        """
        if not messages:
            return ""

        # Формируем текст сообщений
        text_parts = []
        for msg in messages[:50]:  # Ограничиваем количество
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

        conversation_text = "\n".join(text_parts)

        # Создаем промпт с контекстом
        context_part = ""
        if previous_context:
            context_part = (
                f"\n\nКонтекст из предыдущих сообщений:\n{previous_context}\n"
            )

        prompt = f"""Создай краткую саммаризацию следующих сообщений из чата "{chat_name}".
{context_part}
Укажи:
1. Основные темы обсуждения
2. Ключевые решения или события
3. Важные упоминания (люди, проекты)

Сообщения:
{conversation_text}

Саммаризация (3-4 предложения):"""

        try:
            async with self.embedding_client:
                summary = await self.embedding_client.generate_summary(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=131072,  # Для gpt-oss-20b (максимальный лимит)
                )
            return summary.strip()
        except Exception as e:
            logger.error(f"Ошибка саммаризации: {e}")
            return f"Блок из {len(messages)} сообщений в окне {window_name}"

    async def _process_now_window(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        state: SmartAggregationState,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Обрабатывает окно NOW - самые актуальные сообщения

        Создает отдельный файл с саммаризацией и контекстом
        """
        if not messages:
            return {"messages_count": 0}

        logger.info(f"Обработка NOW окна: {len(messages)} сообщений")

        # Получаем контекст из предыдущих окон
        previous_context = self.context_manager.get_previous_context(chat_name, "NOW")

        # Группируем по сессиям если это чат/группа
        if state.chat_type in ["chat", "group"]:
            sessions = self._group_by_sessions(messages, chat_name)

            # Саммаризируем каждую сессию
            summaries = []
            for session in sessions:
                summary = await self._summarize_with_context(
                    session["messages"], chat_name, "now", previous_context
                )
                summaries.append(
                    {
                        "session_id": session["session_id"],
                        "message_count": len(session["messages"]),
                        "summary": summary,
                        "start_time": session.get("start_time"),
                        "end_time": session.get("end_time"),
                    }
                )

                await asyncio.sleep(0.3)  # Задержка между запросами
        else:
            # Для канала - одна саммаризация
            summary = await self._summarize_with_context(
                messages, chat_name, "now", previous_context
            )
            summaries = [
                {
                    "message_count": len(messages),
                    "summary": summary,
                }
            ]

        # Сохраняем в отдельный файл
        if not dry_run:
            now_file = self.now_summaries_dir / f"{chat_name}_NOW.md"

            with open(now_file, "w", encoding="utf-8") as f:
                f.write(f"# Актуальные сообщения: {chat_name}\n\n")
                f.write(f"**Дата:** {format_datetime_display(datetime.now(timezone.utc), format_type='datetime')}\n")
                f.write(f"**Тип чата:** {state.chat_type}\n")
                f.write(f"**Сообщений:** {len(messages)}\n\n")

                if previous_context:
                    f.write(
                        f"## Контекст из предыдущих сообщений\n\n{previous_context}\n\n"
                    )

                f.write("## Саммаризации\n\n")
                for i, summ in enumerate(summaries, 1):
                    f.write(f"### Блок {i}\n\n")
                    if "session_id" in summ:
                        f.write(f"**Сессия:** {summ['session_id']}\n")
                        f.write(
                            f"**Время:** {summ.get('start_time')} - {summ.get('end_time')}\n"
                        )
                    f.write(f"**Сообщений:** {summ['message_count']}\n\n")
                    f.write(f"{summ['summary']}\n\n")

            state.now_summary_file = str(now_file)
            logger.info(f"NOW саммаризация сохранена: {now_file}")

        return {
            "messages_count": len(messages),
            "summaries_count": len(summaries),
            "now_file": state.now_summary_file,
        }

    async def aggregate_chat(
        self, chat_name: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Выполняет умную агрегацию для чата
        """
        logger.info(f"Начало умной агрегации чата: {chat_name}")

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

        # Определяем тип чата
        if not state.chat_type:
            state.chat_type = self._detect_chat_type(messages)
            logger.info(f"Тип чата определен: {state.chat_type}")

        # Выбираем стратегию
        strategy = CHANNEL_STRATEGY if state.chat_type == "channel" else SMART_STRATEGY

        current_date = datetime.now(datetime.now().astimezone().tzinfo)

        # Сначала сортируем все сообщения по времени для правильной хронологии
        sorted_messages = sorted(
            messages,
            key=lambda x: self._parse_date(x.get("date_utc", "")) or datetime.min,
        )

        # Группируем по окнам
        windowed_messages = self._group_messages_by_window(
            sorted_messages, current_date, strategy
        )

        stats = {
            "chat_name": chat_name,
            "chat_type": state.chat_type,
            "total_messages": len(messages),
            "windows": {},
            "now_window": None,
            "sessions": [],  # Добавляем пустой список сессий для совместимости
        }

        # Создаем сессии для индексации в правильном порядке
        sessions = self._create_sessions_for_indexing(
            windowed_messages, chat_name, state.chat_type
        )
        stats["sessions"] = sessions

        # Обрабатываем NOW окно отдельно
        if "now" in windowed_messages:
            now_stats = await self._process_now_window(
                windowed_messages["now"], chat_name, state, dry_run
            )
            stats["now_window"] = now_stats

        # Обрабатываем остальные окна
        for window in strategy:
            if window.name == "now":
                continue  # Уже обработали

            window_messages = windowed_messages.get(window.name, [])
            if not window_messages:
                continue

            logger.info(
                f"Окно '{window.name}': {len(window_messages)} сообщений, стратегия: {window.group_strategy}"
            )

            stats["windows"][window.name] = {
                "messages_count": len(window_messages),
                "group_strategy": window.group_strategy,
                "summarize": window.summarize,
            }

        # Создаем сессии для основного пайплайна индексации
        sessions = self._create_sessions_for_indexing(
            windowed_messages, chat_name, state.chat_type
        )
        stats["sessions"] = sessions

        # Создаем/обновляем накопительный контекст чата
        if not dry_run:
            await self._update_chat_context(chat_name, sessions, state.chat_type)

        # Обновляем состояние
        state.last_aggregation_time = current_date

        if not dry_run:
            self._save_state(state)

        logger.info(f"Умная агрегация завершена для {chat_name}")

        return stats

    def _create_sessions_for_indexing(
        self,
        windowed_messages: Dict[str, List[Dict[str, Any]]],
        chat_name: str,
        chat_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Создает сессии для основного пайплайна индексации

        Для чатов/групп: сначала объединяет все сообщения, потом группирует по связанности
        Для каналов: создает сессии по окнам
        """
        sessions = []
        session_counter = 1

        if chat_type in ["chat", "group"]:
            # Для чатов/групп объединяем все сообщения и группируем по связанности
            # Порядок: от старых к новым для правильной хронологии
            all_messages = []
            for window_name in ["old", "recent", "fresh", "now"]:
                window_messages = windowed_messages.get(window_name, [])
                all_messages.extend(window_messages)

            if all_messages:
                logger.info(
                    f"Создание сессий по связанности для чата '{chat_name}': {len(all_messages)} сообщений"
                )

                # Группируем все сообщения по связанности (не по окнам!)
                all_sessions = self._group_by_sessions(all_messages, chat_name)

                # Если сессий слишком много, используем умную стратегию группировки
                if (
                    len(all_sessions) > len(all_messages) // 10
                ):  # Если больше 10% от сообщений
                    logger.info(
                        f"Слишком много сессий ({len(all_sessions)}), переключаемся на умную группировку"
                    )
                    all_sessions = self._group_by_smart_strategy(
                        windowed_messages, chat_name
                    )

                for session in all_sessions:
                    # Определяем окно для сессии на основе времени
                    session_start = self._parse_date(session.get("start_time", ""))
                    if session_start:
                        current_date = datetime.now(datetime.now().astimezone().tzinfo)
                        age_days = (current_date - session_start).days

                        # Определяем окно для сессии
                        if age_days <= 1:
                            window_name = "now"
                        elif age_days <= 14:
                            window_name = "fresh"
                        elif age_days <= 30:
                            window_name = "recent"
                        else:
                            window_name = "old"
                    else:
                        window_name = "unknown"

                    # Добавляем информацию о окне
                    session["window"] = window_name
                    session["chat"] = chat_name
                    session["session_id"] = f"{chat_name}-S{session_counter:04d}"
                    sessions.append(session)
                    session_counter += 1

        else:
            # Для каналов создаем сессии по окнам с правильным разбиением
            # Порядок: от старых к новым для правильной хронологии
            window_order = ["old", "recent", "fresh", "now"]

            for window_name in window_order:
                window_messages = windowed_messages.get(window_name, [])
                if not window_messages:
                    continue

                logger.info(
                    f"Создание сессий из окна '{window_name}': {len(window_messages)} сообщений"
                )

                # Определяем стратегию группировки для окна
                if window_name == "old":
                    group_strategy = "month"
                elif window_name == "recent":
                    group_strategy = "week"
                elif window_name == "fresh":
                    group_strategy = "day"
                else:  # now
                    group_strategy = "session"

                # Используем DayGroupingSegmenter для разбиения сообщений
                from ..analysis.day_grouping import DayGroupingSegmenter

                segmenter = DayGroupingSegmenter()

                window_sessions = segmenter.group_messages_by_window_strategy(
                    window_messages, chat_name, group_strategy
                )

                # Добавляем информацию об окне к каждой сессии
                for session in window_sessions:
                    session["window"] = window_name
                    session["chat"] = chat_name
                    session[
                        "session_id"
                    ] = f"{chat_name}-{window_name}-S{session_counter:04d}"
                    sessions.append(session)
                    session_counter += 1

                logger.info(
                    f"Создано {len(window_sessions)} сессий для окна '{window_name}'"
                )

        logger.info(f"Создано {len(sessions)} сессий для индексации")
        return sessions

    async def _update_chat_context(
        self, chat_name: str, sessions: List[Dict[str, Any]], chat_type: str
    ) -> None:
        """
        Создает/обновляет накопительный контекст чата

        Создает файл {chat_name}_context.md с единым образом чата
        """
        if not sessions:
            return

        # Создаем директорию для контекстных файлов
        context_dir = Path("artifacts/chat_contexts")
        context_dir.mkdir(exist_ok=True)

        # Загружаем существующий контекст
        context_file = context_dir / f"{chat_name}_context.md"
        existing_context = ""
        if context_file.exists():
            try:
                with open(context_file, encoding="utf-8") as f:
                    existing_context = f.read()
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующий контекст: {e}")

        # Получаем последние сессии для обновления контекста
        recent_sessions = sessions[-10:]  # Последние 10 сессий

        # Создаем промпт для обновления контекста
        sessions_text = []
        for session in recent_sessions:
            session_id = session.get("session_id", "unknown")
            window = session.get("window", "unknown")
            messages = session.get("messages", [])

            session_summary = f"Сессия {session_id} (окно: {window}):\n"
            for msg in messages[:5]:  # Первые 5 сообщений
                sender = msg.get("from", {})
                sender_name = (
                    sender.get("display", "Unknown")
                    if isinstance(sender, dict)
                    else "Unknown"
                )
                text = msg.get("text", "")[:100]  # Первые 100 символов
                if text:
                    session_summary += f"- {sender_name}: {text}\n"
            sessions_text.append(session_summary)

        sessions_content = "\n".join(sessions_text)

        prompt = f"""Обнови образ чата "{chat_name}" на основе новых сессий.

Существующий контекст чата:
{existing_context}

Новые сессии:
{sessions_content}

Создай обновленный образ чата, который включает:
1. Основные темы и интересы участников
2. Актуальные проекты и планы
3. Важные решения и договоренности
4. Особенности общения и стиль чата
5. Ключевые участники и их роли

Образ чата должен быть кратким (3-4 абзаца) и отражать текущее состояние чата с затуханием старой информации."""

        try:
            async with self.ollama_client:
                updated_context = await self.ollama_client.generate_summary(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=131072,  # Для gpt-oss-20b (максимальный лимит)
                )

            # Сохраняем обновленный контекст
            with open(context_file, "w", encoding="utf-8") as f:
                f.write(f"# Контекст чата: {chat_name}\n\n")
                f.write(f"**Обновлено:** {format_datetime_display(datetime.now(timezone.utc), format_type='datetime')}\n")
                f.write(f"**Тип чата:** {chat_type}\n")
                f.write(f"**Всего сессий:** {len(sessions)}\n\n")
                f.write("## Образ чата\n\n")
                f.write(updated_context.strip())
                f.write("\n\n")
                f.write("## Последние сессии\n\n")
                f.write(sessions_content)

            logger.info(f"Обновлен контекст чата: {context_file}")

        except Exception as e:
            logger.error(f"Ошибка обновления контекста чата: {e}")


async def main():
    """Пример использования"""
    logging.basicConfig(level=logging.INFO)

    aggregator = SmartRollingAggregator(
        chats_dir=Path("chats"),
        use_smart_strategy=True,
    )

    # Тестовый запуск на одном чате
    stats = await aggregator.aggregate_chat("Семья", dry_run=True)

    print("\nСтатистика:")
    print(f"Тип чата: {stats.get('chat_type')}")
    print(f"Всего сообщений: {stats.get('total_messages')}")

    if stats.get("now_window"):
        print("\nNOW окно:")
        print(f"  Сообщений: {stats['now_window']['messages_count']}")
        print(f"  Саммаризаций: {stats['now_window'].get('summaries_count', 0)}")

    print("\nДругие окна:")
    for window_name, window_stats in stats.get("windows", {}).items():
        print(
            f"  {window_name}: {window_stats['messages_count']} сообщений, "
            f"стратегия: {window_stats['group_strategy']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
