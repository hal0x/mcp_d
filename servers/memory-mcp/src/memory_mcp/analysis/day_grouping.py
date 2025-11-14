#!/usr/bin/env python3
"""
Модуль для группировки сообщений по дням с учётом естественных разрывов
Оптимизация для ускорения саммаризации
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from .message_filter import MessageFilter

logger = logging.getLogger(__name__)


class DayGroupingSegmenter:
    """Класс для группировки сообщений по дням с естественными разрывами"""

    def __init__(
        self,
        min_messages_per_group: int = 10,
        max_messages_per_group: int = 100,
        night_gap_hours: int = 4,
        timezone: str = "Asia/Bangkok",
        enable_filtering: bool = True,
    ):
        """
        Инициализация сегментатора

        Args:
            min_messages_per_group: Минимум сообщений в группе
            max_messages_per_group: Максимум сообщений в группе
            night_gap_hours: Разрыв в часах для определения конца обсуждения ночью
            timezone: Часовой пояс для анализа времени
            enable_filtering: Включить фильтрацию и дедупликацию сообщений
        """
        self.min_messages_per_group = min_messages_per_group
        self.max_messages_per_group = max_messages_per_group
        self.night_gap_hours = night_gap_hours
        self.tz = ZoneInfo(timezone)
        self.enable_filtering = enable_filtering
        self.message_filter = MessageFilter() if enable_filtering else None

    def group_messages_by_days(
        self, messages: List[Dict[str, Any]], chat_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Группировка сообщений по дням с учётом естественных разрывов

        Args:
            messages: Список сообщений
            chat_name: Название чата

        Returns:
            Список групп сообщений
        """
        if not messages:
            return []

        # Фильтруем пустые и дублирующиеся сообщения
        if self.enable_filtering and self.message_filter:
            messages = self.message_filter.filter_messages(messages)
            if not messages:
                logger.warning(
                    f"После фильтрации не осталось сообщений для чата {chat_name}"
                )
                return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(messages, key=lambda x: self._parse_message_time(x))

        # Определяем естественные разрывы в обсуждениях
        natural_breaks = self._find_natural_breaks(sorted_messages)

        # Группируем сообщения по дням с учётом разрывов
        day_groups = self._group_by_days_with_breaks(sorted_messages, natural_breaks)

        # Объединяем маленькие группы
        optimized_groups = self._merge_small_groups(day_groups)

        # Создаём финальные сессии
        sessions = []
        for i, group in enumerate(optimized_groups):
            session = self._create_session_from_group(group, i, chat_name)
            sessions.append(session)

        logger.info(
            f"Создано {len(sessions)} групп из {len(sorted_messages)} сообщений"
        )
        return sessions

    def group_messages_by_window_strategy(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str = None,
        window_strategy: str = "day",
    ) -> List[Dict[str, Any]]:
        """
        Группировка сообщений по различным стратегиям в зависимости от окна

        Args:
            messages: Список сообщений
            chat_name: Название чата
            window_strategy: Стратегия группировки ('day', 'week', 'month', 'session')

        Returns:
            Список групп сообщений
        """
        if not messages:
            return []

        # Фильтруем пустые и дублирующиеся сообщения
        if self.enable_filtering and self.message_filter:
            messages = self.message_filter.filter_messages(messages)
            if not messages:
                logger.warning(
                    f"После фильтрации не осталось сообщений для чата {chat_name}"
                )
                return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(messages, key=lambda x: self._parse_message_time(x))

        if window_strategy == "day":
            # Группировка по дням (как раньше)
            natural_breaks = self._find_natural_breaks(sorted_messages)
            day_groups = self._group_by_days_with_breaks(
                sorted_messages, natural_breaks
            )
            optimized_groups = self._merge_small_groups(day_groups)
        elif window_strategy == "week":
            # Группировка по неделям
            optimized_groups = self._group_by_weeks(sorted_messages)
        elif window_strategy == "month":
            # Группировка по месяцам
            optimized_groups = self._group_by_months(sorted_messages)
        elif window_strategy == "session":
            # Группировка по сессиям (используем SessionSegmenter)
            from .session_segmentation import SessionSegmenter

            session_segmenter = SessionSegmenter()
            sessions = session_segmenter.segment_messages(sorted_messages, chat_name)
            return sessions
        else:
            # По умолчанию группировка по дням
            natural_breaks = self._find_natural_breaks(sorted_messages)
            day_groups = self._group_by_days_with_breaks(
                sorted_messages, natural_breaks
            )
            optimized_groups = self._merge_small_groups(day_groups)

        # Создаём финальные сессии
        sessions = []
        for i, group in enumerate(optimized_groups):
            session = self._create_session_from_group(group, i, chat_name)
            session["group_type"] = f"{window_strategy}_grouped"
            sessions.append(session)

        logger.info(
            f"Создано {len(sessions)} групп по стратегии '{window_strategy}' из {len(sorted_messages)} сообщений"
        )
        return sessions

    def _find_natural_breaks(self, messages: List[Dict[str, Any]]) -> List[int]:
        """
        Поиск естественных разрывов в обсуждениях

        Args:
            messages: Отсортированные сообщения

        Returns:
            Список индексов разрывов
        """
        breaks = []

        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            prev_time = self._parse_message_time(prev_msg)
            curr_time = self._parse_message_time(curr_msg)

            # Разрыв больше night_gap_hours часов
            time_diff = curr_time - prev_time
            if time_diff.total_seconds() / 3600 > self.night_gap_hours:
                breaks.append(i)

        return breaks

    def _group_by_days_with_breaks(
        self, messages: List[Dict[str, Any]], natural_breaks: List[int]
    ) -> List[List[Dict[str, Any]]]:
        """
        Группировка сообщений по дням с учётом естественных разрывов

        Args:
            messages: Отсортированные сообщения
            natural_breaks: Индексы разрывов

        Returns:
            Список групп сообщений
        """
        groups = []
        current_group = []

        for i, msg in enumerate(messages):
            current_group.append(msg)

            # Проверяем, нужно ли завершить группу
            should_break = False

            # 1. Естественный разрыв
            if i in natural_breaks:
                should_break = True

            # 2. Смена дня (но не по полуночи, а по активности)
            elif len(current_group) > 1:
                first_msg_time = self._parse_message_time(current_group[0])
                last_msg_time = self._parse_message_time(current_group[-1])

                # Если группа охватывает более 24 часов активности
                if self._is_new_day_activity(first_msg_time, last_msg_time):
                    should_break = True

            # 3. Превышение максимального размера группы
            elif len(current_group) >= self.max_messages_per_group:
                should_break = True

            if should_break:
                groups.append(current_group)
                current_group = []

        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)

        return groups

    def _is_new_day_activity(self, start_time: datetime, end_time: datetime) -> bool:
        """
        Определение смены дня по активности (не по полуночи)

        Args:
            start_time: Время начала группы
            end_time: Время конца группы

        Returns:
            True если это новый день активности
        """
        # Конвертируем в локальный часовой пояс
        start_local = start_time.astimezone(self.tz)
        end_local = end_time.astimezone(self.tz)

        # Если группа охватывает более 24 часов
        if (end_local - start_local).total_seconds() > 24 * 3600:
            return True

        # Если группа начинается в один день, а заканчивается в другой
        # И между ними есть значительный разрыв активности
        if start_local.date() != end_local.date():
            # Проверяем, есть ли разрыв в активности ночью
            night_start = start_local.replace(
                hour=23, minute=0, second=0, microsecond=0
            )
            night_end = end_local.replace(hour=6, minute=0, second=0, microsecond=0)

            # Если между 23:00 и 6:00 есть разрыв больше 2 часов
            if (night_end - night_start).total_seconds() > 2 * 3600:
                return True

        return False

    def _merge_small_groups(
        self, groups: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Объединение маленьких групп для оптимизации

        Args:
            groups: Список групп сообщений

        Returns:
            Оптимизированный список групп
        """
        if not groups:
            return []

        optimized = []
        current_group = []

        for group in groups:
            # Если текущая группа пустая, начинаем новую
            if not current_group:
                current_group = group.copy()
                continue

            # Проверяем, можно ли объединить с предыдущей группой
            if len(current_group) < self.min_messages_per_group:
                # Объединяем группы
                current_group.extend(group)

                # Проверяем, не превысили ли максимум
                if len(current_group) > self.max_messages_per_group:
                    # Разбиваем на две группы
                    split_point = self.max_messages_per_group
                    optimized.append(current_group[:split_point])
                    current_group = current_group[split_point:]
            else:
                # Текущая группа достаточно большая, сохраняем её
                optimized.append(current_group)
                current_group = group.copy()

        # Добавляем последнюю группу
        if current_group:
            optimized.append(current_group)

        return optimized

    def _create_session_from_group(
        self, group: List[Dict[str, Any]], group_index: int, chat_name: str = None
    ) -> Dict[str, Any]:
        """
        Создание сессии из группы сообщений

        Args:
            group: Группа сообщений
            group_index: Индекс группы

        Returns:
            Словарь с метаданными сессии
        """
        if not group:
            return {}

        start_time = self._parse_message_time(group[0])
        end_time = self._parse_message_time(group[-1])

        # Генерируем session_id
        session_id = (
            f"{chat_name}-D{group_index + 1:04d}"
            if chat_name
            else f"D{group_index + 1:04d}"
        )

        # Подсчитываем участников
        participants = set()
        for msg in group:
            from_user = msg.get("from", {})
            if isinstance(from_user, dict):
                username = from_user.get("username") or from_user.get("display")
                if username:
                    participants.add(username)
            elif isinstance(from_user, str):
                participants.add(from_user)

        # Определяем доминирующий язык
        language_counts = {}
        for msg in group:
            lang = msg.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

        dominant_language = (
            max(language_counts.items(), key=lambda x: x[1])[0]
            if language_counts
            else "unknown"
        )

        # Конвертируем времена в Bangkok timezone
        start_time_bkk = start_time.astimezone(self.tz)
        end_time_bkk = end_time.astimezone(self.tz)

        # Определяем период активности
        if start_time_bkk.date() == end_time_bkk.date():
            # Один день
            time_range = f"{start_time_bkk.strftime('%Y-%m-%d %H:%M')}–{end_time_bkk.strftime('%H:%M')} BKK"
        else:
            # Несколько дней
            time_range = f"{start_time_bkk.strftime('%Y-%m-%d %H:%M')}–{end_time_bkk.strftime('%Y-%m-%d %H:%M')} BKK"

        session = {
            "session_id": session_id,
            "chat": chat_name,
            "messages": group,
            "message_count": len(group),
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "start_time_bkk": start_time_bkk.strftime("%Y-%m-%d %H:%M BKK"),
            "end_time_bkk": end_time_bkk.strftime("%Y-%m-%d %H:%M BKK"),
            "time_range_bkk": time_range,
            "duration_minutes": round((end_time - start_time).total_seconds() / 60, 1),
            "participants": sorted(participants),
            "participant_count": len(participants),
            "dominant_language": dominant_language,
            "group_type": "day_grouped",
        }

        return session

    def _parse_message_time(self, msg: Dict[str, Any]) -> datetime:
        """
        Парсинг времени сообщения (использует общую утилиту).

        Args:
            msg: Сообщение

        Returns:
            datetime объект в UTC
        """
        from ..utils.datetime_utils import parse_message_time

        return parse_message_time(msg, use_zoneinfo=True)

    def _group_by_weeks(
        self, messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Группировка сообщений по неделям

        Args:
            messages: Отсортированные сообщения

        Returns:
            Список групп сообщений по неделям
        """
        groups = []
        current_group = []
        current_week = None

        for msg in messages:
            msg_time = self._parse_message_time(msg)
            msg_week = msg_time.isocalendar()[:2]  # (year, week)

            if current_week is None or current_week != msg_week:
                # Начинаем новую неделю
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
                current_week = msg_week
            else:
                # Добавляем к текущей неделе
                current_group.append(msg)

        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)

        return groups

    def _group_by_months(
        self, messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Группировка сообщений по месяцам

        Args:
            messages: Отсортированные сообщения

        Returns:
            Список групп сообщений по месяцам
        """
        groups = []
        current_group = []
        current_month = None

        for msg in messages:
            msg_time = self._parse_message_time(msg)
            msg_month = (msg_time.year, msg_time.month)

            if current_month is None or current_month != msg_month:
                # Начинаем новый месяц
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
                current_month = msg_month
            else:
                # Добавляем к текущему месяцу
                current_group.append(msg)

        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)

        return groups


def group_chat_messages_by_days(
    messages: List[Dict[str, Any]],
    chat_name: str = None,
    min_messages: int = 10,
    max_messages: int = 100,
    enable_filtering: bool = True,
) -> List[Dict[str, Any]]:
    """
    Удобная функция для группировки сообщений по дням

    Args:
        messages: Список сообщений
        chat_name: Название чата
        min_messages: Минимум сообщений в группе
        max_messages: Максимум сообщений в группе
        enable_filtering: Включить фильтрацию и дедупликацию

    Returns:
        Список групп-сессий
    """
    segmenter = DayGroupingSegmenter(
        min_messages_per_group=min_messages,
        max_messages_per_group=max_messages,
        enable_filtering=enable_filtering,
    )
    return segmenter.group_messages_by_days(messages, chat_name)


if __name__ == "__main__":
    # Тест модуля
    from datetime import datetime, timedelta

    # Создаём тестовые сообщения
    base_time = datetime.now(ZoneInfo("UTC"))
    test_messages = []

    # День 1: 25 сообщений
    for i in range(25):
        test_messages.append(
            {
                "id": str(i + 1),
                "date_utc": (base_time + timedelta(hours=i)).isoformat(),
                "text": f"Message {i + 1}",
                "from": {"username": f"user{(i % 3) + 1}"},
                "language": "ru",
            }
        )

    # Разрыв 6 часов
    # День 2: 15 сообщений
    for i in range(15):
        test_messages.append(
            {
                "id": str(i + 26),
                "date_utc": (base_time + timedelta(hours=25 + i)).isoformat(),
                "text": f"Message {i + 26}",
                "from": {"username": f"user{(i % 2) + 1}"},
                "language": "ru",
            }
        )

    segmenter = DayGroupingSegmenter()
    sessions = segmenter.group_messages_by_days(test_messages, chat_name="TestChat")

    print(f"Создано групп: {len(sessions)}")
    for session in sessions:
        print(
            f"  {session['session_id']}: {session['message_count']} сообщений, {session['time_range_bkk']}"
        )
