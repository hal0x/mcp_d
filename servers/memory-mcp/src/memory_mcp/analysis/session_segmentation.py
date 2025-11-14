#!/usr/bin/env python3
"""
Модуль для сегментации сообщений на сессии
Правила:
- Gap = 120 минут (UTC) - увеличен для уменьшения количества маленьких сессий
- Ночная склейка MSK 00:00-09:00 (эквивалент Asia/Bangkok 04:00-13:00)
- Ограничение длины сессии ≤ 6 часов
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from .time_processor import TimeProcessor

logger = logging.getLogger(__name__)


class SessionSegmenter:
    """Класс для сегментации сообщений на сессии"""

    def __init__(
        self,
        gap_minutes: int = 120,  # Увеличено с 60 до 120 минут
        max_session_hours: int = 6,
        night_merge_tz: str = "Europe/Moscow",
        night_merge_start: str = "00:00",
        night_merge_end: str = "09:00",
        enable_time_analysis: bool = True,
    ):
        """
        Инициализация сегментатора

        Args:
            gap_minutes: Максимальный разрыв между сообщениями в одной сессии (минуты)
            max_session_hours: Максимальная длина сессии (часы)
            night_merge_tz: Часовой пояс для ночной склейки
            night_merge_start: Начало ночного окна (HH:MM)
            night_merge_end: Конец ночного окна (HH:MM)
            enable_time_analysis: Включить анализ временных паттернов
        """
        self.gap_minutes = gap_minutes
        self.max_session_hours = max_session_hours
        self.night_merge_tz = ZoneInfo(night_merge_tz)
        self.enable_time_analysis = enable_time_analysis

        # Парсим время начала и конца ночного окна
        start_parts = night_merge_start.split(":")
        self.night_start_hour = int(start_parts[0])
        self.night_start_minute = int(start_parts[1]) if len(start_parts) > 1 else 0

        end_parts = night_merge_end.split(":")
        self.night_end_hour = int(end_parts[0])
        self.night_end_minute = int(end_parts[1]) if len(end_parts) > 1 else 0

        # Инициализируем TimeProcessor для анализа временных паттернов
        self.time_processor = TimeProcessor() if enable_time_analysis else None

    def segment_messages(
        self, messages: List[Dict[str, Any]], chat_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Сегментация сообщений на сессии с анализом временных паттернов

        Args:
            messages: Список сообщений
            chat_name: Название чата (опционально)

        Returns:
            Список сессий, каждая сессия - словарь с метаданными и списком сообщений
        """
        if not messages:
            return []

        # Анализируем временные паттерны если включено
        activity_patterns = {}
        if self.time_processor and self.enable_time_analysis:
            try:
                activity_patterns = self.time_processor.analyze_activity_patterns(messages)
                logger.info(f"Анализ временных паттернов завершен для чата {chat_name}")
            except Exception as e:
                logger.warning(f"Ошибка анализа временных паттернов: {e}")

        # Сортируем сообщения по времени
        sorted_messages = sorted(messages, key=lambda x: self._parse_message_time(x))

        sessions = []
        current_session = {
            "messages": [],
            "start_time": None,
            "end_time": None,
            "chat": chat_name,
        }

        for _i, msg in enumerate(sorted_messages):
            msg_time = self._parse_message_time(msg)

            if not current_session["messages"]:
                # Начинаем новую сессию
                current_session["messages"].append(msg)
                current_session["start_time"] = msg_time
                current_session["end_time"] = msg_time
            else:
                # Проверяем, нужно ли начать новую сессию
                last_msg_time = self._parse_message_time(
                    current_session["messages"][-1]
                )
                time_diff = (msg_time - last_msg_time).total_seconds() / 60  # в минутах
                session_duration = (
                    msg_time - current_session["start_time"]
                ).total_seconds() / 3600  # в часах

                # Проверяем условия разрыва сессии
                should_break = False

                # 1. Проверка gap (но с учётом ночного окна)
                if time_diff > self.gap_minutes:
                    # Проверяем, не попадает ли разрыв в ночное окно
                    if not self._is_night_merge_applicable(last_msg_time, msg_time):
                        should_break = True

                # 2. Проверка максимальной длины сессии
                if session_duration >= self.max_session_hours:
                    should_break = True

                if should_break:
                    # Завершаем текущую сессию
                    session = self._finalize_session(current_session, len(sessions))
                    # Добавляем информацию о временных паттернах
                    if activity_patterns:
                        session["activity_patterns"] = activity_patterns
                    sessions.append(session)

                    # Начинаем новую сессию
                    current_session = {
                        "messages": [msg],
                        "start_time": msg_time,
                        "end_time": msg_time,
                        "chat": chat_name,
                    }
                else:
                    # Добавляем сообщение к текущей сессии
                    current_session["messages"].append(msg)
                    current_session["end_time"] = msg_time

        # Добавляем последнюю сессию
        if current_session["messages"]:
            session = self._finalize_session(current_session, len(sessions))
            # Добавляем информацию о временных паттернах
            if activity_patterns:
                session["activity_patterns"] = activity_patterns
            sessions.append(session)

        logger.info(
            f"Сегментировано {len(sorted_messages)} сообщений в {len(sessions)} сессий"
        )
        return sessions

    def _is_night_merge_applicable(self, time1: datetime, time2: datetime) -> bool:
        """
        Проверка, попадает ли разрыв между двумя сообщениями в ночное окно
        Если оба сообщения или разрыв между ними попадает в ночное окно MSK,
        то склеиваем сессию

        Args:
            time1: Время первого сообщения (UTC)
            time2: Время второго сообщения (UTC)

        Returns:
            True если разрыв в ночном окне
        """
        # Конвертируем в timezone для ночного окна (MSK)
        time1_local = time1.astimezone(self.night_merge_tz)
        time2_local = time2.astimezone(self.night_merge_tz)

        # Проверяем, попадает ли время в ночное окно
        def is_in_night_window(dt: datetime) -> bool:
            hour = dt.hour
            minute = dt.minute

            # Преобразуем время в минуты от начала суток
            current_minutes = hour * 60 + minute
            start_minutes = self.night_start_hour * 60 + self.night_start_minute
            end_minutes = self.night_end_hour * 60 + self.night_end_minute

            return start_minutes <= current_minutes < end_minutes

        # Если хотя бы одно из времён попадает в ночное окно, склеиваем
        return is_in_night_window(time1_local) or is_in_night_window(time2_local)

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

    def _finalize_session(
        self, session: Dict[str, Any], session_index: int
    ) -> Dict[str, Any]:
        """
        Финализация сессии с добавлением метаданных

        Args:
            session: Сессия для финализации
            session_index: Индекс сессии

        Returns:
            Финализированная сессия с полными метаданными
        """
        messages = session["messages"]
        start_time = session["start_time"]
        end_time = session["end_time"]
        chat = session["chat"]

        # Генерируем session_id
        session_id = (
            f"{chat}-S{session_index + 1:04d}" if chat else f"S{session_index + 1:04d}"
        )

        # Подсчитываем статистику
        total_messages = len(messages)
        duration_minutes = (end_time - start_time).total_seconds() / 60

        # Извлекаем участников
        participants = set()
        for msg in messages:
            from_user = msg.get("from", {})
            if isinstance(from_user, dict):
                username = from_user.get("username") or from_user.get("display")
                if username:
                    participants.add(username)
            elif isinstance(from_user, str):
                participants.add(from_user)

        # Определяем доминирующий язык
        language_counts = {}
        for msg in messages:
            lang = msg.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

        dominant_language = (
            max(language_counts.items(), key=lambda x: x[1])[0]
            if language_counts
            else "unknown"
        )

        # Конвертируем времена в Bangkok timezone для отображения
        bkk_tz = ZoneInfo("Asia/Bangkok")
        start_time_bkk = start_time.astimezone(bkk_tz)
        end_time_bkk = end_time.astimezone(bkk_tz)

        finalized = {
            "session_id": session_id,
            "chat": chat,
            "messages": messages,
            "message_count": total_messages,
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "start_time_bkk": start_time_bkk.strftime("%Y-%m-%d %H:%M BKK"),
            "end_time_bkk": end_time_bkk.strftime("%Y-%m-%d %H:%M BKK"),
            "time_range_bkk": f"{start_time_bkk.strftime('%Y-%m-%d %H:%M')}–{end_time_bkk.strftime('%H:%M')} BKK",
            "duration_minutes": round(duration_minutes, 1),
            "participants": sorted(participants),
            "participant_count": len(participants),
            "dominant_language": dominant_language,
        }

        return finalized

    def split_long_session(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбиение длинной сессии на части для иерархической саммаризации

        Args:
            session: Сессия для разбиения

        Returns:
            Список подсессий
        """
        messages = session["messages"]
        duration_hours = session["duration_minutes"] / 60

        if duration_hours <= self.max_session_hours:
            return [session]

        # Вычисляем количество частей
        num_parts = int(duration_hours / self.max_session_hours) + 1
        messages_per_part = len(messages) // num_parts

        subsessions = []
        for i in range(num_parts):
            start_idx = i * messages_per_part
            end_idx = (
                start_idx + messages_per_part if i < num_parts - 1 else len(messages)
            )

            if start_idx >= len(messages):
                break

            subsession_messages = messages[start_idx:end_idx]
            if not subsession_messages:
                continue

            subsession = {
                "messages": subsession_messages,
                "start_time": self._parse_message_time(subsession_messages[0]),
                "end_time": self._parse_message_time(subsession_messages[-1]),
                "chat": session["chat"],
            }

            subsessions.append(self._finalize_session(subsession, i))

        logger.info(f"Длинная сессия разбита на {len(subsessions)} подсессий")
        return subsessions

    def _should_split_based_on_patterns(
        self, prev_time: datetime, curr_time: datetime, activity_patterns: Dict[str, Any]
    ) -> bool:
        """
        Дополнительная проверка разделения сессии на основе временных паттернов

        Args:
            prev_time: Время предыдущего сообщения
            curr_time: Время текущего сообщения
            activity_patterns: Анализ паттернов активности

        Returns:
            True если нужно разделить сессию
        """
        if not self.time_processor or not activity_patterns:
            return False

        try:
            # Определяем категории времени для обоих сообщений
            prev_category = self.time_processor.get_time_of_day_category(prev_time)
            curr_category = self.time_processor.get_time_of_day_category(curr_time)

            # Если сообщения в разных категориях времени суток
            if prev_category != curr_category:
                # Проверяем, является ли это переходом в неактивное время
                inactive_categories = ["night"]
                if prev_category not in inactive_categories and curr_category in inactive_categories:
                    return True

            # Анализируем пиковые часы
            peak_hours = activity_patterns.get("peak_hours", [])
            if peak_hours:
                prev_hour = prev_time.hour
                curr_hour = curr_time.hour
                
                # Если переходим от пикового часа к непиковому
                peak_hour_list = [ph["hour"] for ph in peak_hours[:3]]  # Топ-3 пиковых часа
                if prev_hour in peak_hour_list and curr_hour not in peak_hour_list:
                    return True

            return False

        except Exception as e:
            logger.debug(f"Ошибка анализа паттернов для разделения сессии: {e}")
            return False

    def get_time_analysis(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Получение анализа временных паттернов для сообщений

        Args:
            messages: Список сообщений

        Returns:
            Словарь с анализом временных паттернов
        """
        if not self.time_processor or not messages:
            return {}

        try:
            return self.time_processor.analyze_activity_patterns(messages)
        except Exception as e:
            logger.warning(f"Ошибка анализа временных паттернов: {e}")
            return {}


def segment_chat_messages(
    messages: List[Dict[str, Any]],
    chat_name: str = None,
    gap_minutes: int = 60,
    max_session_hours: int = 6,
) -> List[Dict[str, Any]]:
    """
    Удобная функция для быстрой сегментации сообщений

    Args:
        messages: Список сообщений
        chat_name: Название чата
        gap_minutes: Разрыв между сессиями в минутах
        max_session_hours: Максимальная длина сессии в часах

    Returns:
        Список сессий
    """
    segmenter = SessionSegmenter(
        gap_minutes=gap_minutes, max_session_hours=max_session_hours
    )
    return segmenter.segment_messages(messages, chat_name)


if __name__ == "__main__":
    # Тест модуля
    from datetime import datetime, timedelta

    # Создаём тестовые сообщения
    base_time = datetime.now(ZoneInfo("UTC"))
    test_messages = [
        {
            "id": "1",
            "date_utc": base_time.isoformat(),
            "text": "Message 1",
            "from": {"username": "alice"},
        },
        {
            "id": "2",
            "date_utc": (base_time + timedelta(minutes=10)).isoformat(),
            "text": "Message 2",
            "from": {"username": "bob"},
        },
        {
            "id": "3",
            "date_utc": (base_time + timedelta(minutes=90)).isoformat(),
            "text": "Message 3",
            "from": {"username": "alice"},
        },
    ]

    segmenter = SessionSegmenter()
    sessions = segmenter.segment_messages(test_messages, chat_name="TestChat")

    print(f"Сегментировано сессий: {len(sessions)}")
    for session in sessions:
        print(
            f"  {session['session_id']}: {session['message_count']} сообщений, {session['duration_minutes']} минут"
        )
