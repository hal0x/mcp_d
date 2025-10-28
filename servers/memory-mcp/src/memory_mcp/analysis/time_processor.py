#!/usr/bin/env python3
"""
Модуль для комплексной обработки временных меток и анализа паттернов активности
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Временные категории
TIME_OF_DAY_CATEGORIES = {
    "night": (0, 6),      # Ночь: 00:00-06:00
    "morning": (6, 12),   # Утро: 06:00-12:00
    "afternoon": (12, 18), # День: 12:00-18:00
    "evening": (18, 24),  # Вечер: 18:00-24:00
}

# Периоды активности
ACTIVITY_PERIODS = {
    "workday": [0, 1, 2, 3, 4],  # Понедельник-пятница
    "weekend": [5, 6],           # Суббота-воскресенье
}

# Часовые пояса для определения
COMMON_TIMEZONES = [
    "Europe/Moscow",
    "Europe/London", 
    "America/New_York",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "UTC",
]


class TimeProcessor:
    """Класс для комплексной обработки временных меток и анализа паттернов"""

    def __init__(self):
        """Инициализация процессора времени"""
        self.detected_timezone: Optional[str] = None
        self.activity_patterns: Dict[str, Any] = {}

    def normalize_timestamp(self, date_str: str) -> Optional[datetime]:
        """
        Нормализация временной метки в различных форматах

        Args:
            date_str: Строка с датой/временем

        Returns:
            Нормализованный datetime объект в UTC или None
        """
        if not date_str:
            return None

        try:
            # Убираем 'Z' и добавляем '+00:00' если нужно
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"

            # Парсим ISO формат
            dt = datetime.fromisoformat(date_str)

            # Убеждаемся, что время в UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            else:
                dt = dt.astimezone(ZoneInfo("UTC"))

            return dt

        except Exception as e:
            logger.debug(f"Ошибка парсинга времени '{date_str}': {e}")
            return None

    def detect_timezone(self, messages: List[Dict]) -> str:
        """
        Определение временной зоны на основе паттернов активности

        Args:
            messages: Список сообщений

        Returns:
            Определенная временная зона
        """
        if not messages:
            return "UTC"

        # Анализируем часы активности
        hour_counts = Counter()
        valid_messages = 0

        for msg in messages:
            timestamp = self.normalize_timestamp(msg.get("date_utc") or msg.get("date", ""))
            if timestamp:
                hour_counts[timestamp.hour] += 1
                valid_messages += 1

        if valid_messages < 10:
            return "UTC"

        # Определяем наиболее вероятную временную зону
        best_timezone = "UTC"
        best_score = 0

        for tz_name in COMMON_TIMEZONES:
            score = self._calculate_timezone_score(hour_counts, tz_name)
            if score > best_score:
                best_score = score
                best_timezone = tz_name

        self.detected_timezone = best_timezone
        logger.info(f"Определена временная зона: {best_timezone} (score: {best_score:.2f})")
        return best_timezone

    def _calculate_timezone_score(self, hour_counts: Counter, timezone: str) -> float:
        """
        Расчет скора для временной зоны на основе паттернов активности

        Args:
            hour_counts: Счетчики часов активности
            timezone: Название временной зоны

        Returns:
            Скор временной зоны
        """
        try:
            tz = ZoneInfo(timezone)
            score = 0.0
            total_messages = sum(hour_counts.values())

            # Анализируем активность в локальном времени
            local_hour_counts = Counter()
            for utc_hour, count in hour_counts.items():
                # Конвертируем UTC час в локальное время
                utc_dt = datetime(2024, 1, 1, utc_hour, tzinfo=ZoneInfo("UTC"))
                local_dt = utc_dt.astimezone(tz)
                local_hour_counts[local_dt.hour] += count

            # Оцениваем паттерны активности
            # Высокая активность в рабочие часы (9-18) дает больше очков
            work_hours = list(range(9, 18))
            work_activity = sum(local_hour_counts[h] for h in work_hours)
            work_score = work_activity / total_messages if total_messages > 0 else 0

            # Низкая активность в ночные часы (0-6) дает больше очков
            night_hours = list(range(0, 6))
            night_activity = sum(local_hour_counts[h] for h in night_hours)
            night_score = 1.0 - (night_activity / total_messages) if total_messages > 0 else 0

            score = work_score * 0.7 + night_score * 0.3
            return score

        except Exception as e:
            logger.debug(f"Ошибка расчета скора для {timezone}: {e}")
            return 0.0

    def analyze_activity_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Анализ паттернов активности

        Args:
            messages: Список сообщений

        Returns:
            Словарь с паттернами активности
        """
        if not messages:
            return {}

        # Определяем временную зону если еще не определена
        if not self.detected_timezone:
            self.detect_timezone(messages)

        # Нормализуем временные метки
        timestamps = []
        for msg in messages:
            timestamp = self.normalize_timestamp(msg.get("date_utc") or msg.get("date", ""))
            if timestamp:
                timestamps.append(timestamp)

        if not timestamps:
            return {}

        timestamps.sort()

        # Анализируем различные паттерны
        patterns = {
            "timezone": self.detected_timezone,
            "time_of_day_distribution": self._analyze_time_of_day(timestamps),
            "day_of_week_distribution": self._analyze_day_of_week(timestamps),
            "activity_periods": self._analyze_activity_periods(timestamps),
            "session_patterns": self._analyze_session_patterns(timestamps),
            "peak_hours": self._find_peak_hours(timestamps),
            "silence_periods": self._analyze_silence_periods(timestamps),
            "message_frequency": self._calculate_message_frequency(timestamps),
        }

        self.activity_patterns = patterns
        return patterns

    def _analyze_time_of_day(self, timestamps: List[datetime]) -> Dict[str, int]:
        """Анализ распределения по времени суток"""
        distribution = defaultdict(int)

        for timestamp in timestamps:
            # Конвертируем в локальное время если определена зона
            if self.detected_timezone and self.detected_timezone != "UTC":
                local_timestamp = timestamp.astimezone(ZoneInfo(self.detected_timezone))
            else:
                local_timestamp = timestamp

            hour = local_timestamp.hour

            # Определяем категорию времени суток
            for category, (start_hour, end_hour) in TIME_OF_DAY_CATEGORIES.items():
                if start_hour <= hour < end_hour:
                    distribution[category] += 1
                    break

        return dict(distribution)

    def _analyze_day_of_week(self, timestamps: List[datetime]) -> Dict[str, int]:
        """Анализ распределения по дням недели"""
        distribution = defaultdict(int)
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        for timestamp in timestamps:
            # Конвертируем в локальное время если определена зона
            if self.detected_timezone and self.detected_timezone != "UTC":
                local_timestamp = timestamp.astimezone(ZoneInfo(self.detected_timezone))
            else:
                local_timestamp = timestamp

            weekday = local_timestamp.weekday()
            distribution[day_names[weekday]] += 1

        return dict(distribution)

    def _analyze_activity_periods(self, timestamps: List[datetime]) -> Dict[str, int]:
        """Анализ периодов активности (рабочие дни vs выходные)"""
        distribution = defaultdict(int)

        for timestamp in timestamps:
            # Конвертируем в локальное время если определена зона
            if self.detected_timezone and self.detected_timezone != "UTC":
                local_timestamp = timestamp.astimezone(ZoneInfo(self.detected_timezone))
            else:
                local_timestamp = timestamp

            weekday = local_timestamp.weekday()

            if weekday in ACTIVITY_PERIODS["workday"]:
                distribution["workday"] += 1
            elif weekday in ACTIVITY_PERIODS["weekend"]:
                distribution["weekend"] += 1

        return dict(distribution)

    def _analyze_session_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Анализ паттернов сессий"""
        if len(timestamps) < 2:
            return {}

        # Группируем сообщения в сессии (разрыв > 2 часов)
        sessions = []
        current_session = [timestamps[0]]

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > timedelta(hours=2):
                # Началась новая сессия
                sessions.append(current_session)
                current_session = [timestamps[i]]
            else:
                current_session.append(timestamps[i])

        sessions.append(current_session)

        # Анализируем сессии
        session_durations = []
        session_sizes = []

        for session in sessions:
            if len(session) > 1:
                duration = session[-1] - session[0]
                session_durations.append(duration.total_seconds() / 3600)  # в часах
                session_sizes.append(len(session))

        if not session_durations:
            return {}

        return {
            "total_sessions": len(sessions),
            "avg_session_duration_hours": sum(session_durations) / len(session_durations),
            "avg_session_size": sum(session_sizes) / len(session_sizes),
            "max_session_duration_hours": max(session_durations),
            "min_session_duration_hours": min(session_durations),
        }

    def _find_peak_hours(self, timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Поиск пиковых часов активности"""
        hour_counts = Counter()

        for timestamp in timestamps:
            # Конвертируем в локальное время если определена зона
            if self.detected_timezone and self.detected_timezone != "UTC":
                local_timestamp = timestamp.astimezone(ZoneInfo(self.detected_timezone))
            else:
                local_timestamp = timestamp

            hour_counts[local_timestamp.hour] += 1

        # Возвращаем топ-5 часов
        top_hours = hour_counts.most_common(5)
        return [{"hour": hour, "count": count} for hour, count in top_hours]

    def _analyze_silence_periods(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Анализ периодов молчания"""
        if len(timestamps) < 2:
            return {}

        gaps = []
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            gaps.append(gap.total_seconds() / 3600)  # в часах

        if not gaps:
            return {}

        return {
            "avg_gap_hours": sum(gaps) / len(gaps),
            "max_gap_hours": max(gaps),
            "min_gap_hours": min(gaps),
            "gaps_over_24h": len([g for g in gaps if g > 24]),
            "gaps_over_week": len([g for g in gaps if g > 168]),  # 7 дней
        }

    def _calculate_message_frequency(self, timestamps: List[datetime]) -> Dict[str, float]:
        """Расчет частоты сообщений"""
        if len(timestamps) < 2:
            return {}

        total_duration = timestamps[-1] - timestamps[0]
        total_hours = total_duration.total_seconds() / 3600

        if total_hours == 0:
            return {}

        return {
            "messages_per_hour": len(timestamps) / total_hours,
            "messages_per_day": len(timestamps) / (total_hours / 24),
            "total_duration_hours": total_hours,
            "total_messages": len(timestamps),
        }

    def get_time_of_day_category(self, dt: datetime) -> str:
        """
        Определение категории времени суток

        Args:
            dt: Временная метка

        Returns:
            Категория времени суток
        """
        # Конвертируем в локальное время если определена зона
        if self.detected_timezone and self.detected_timezone != "UTC":
            local_dt = dt.astimezone(ZoneInfo(self.detected_timezone))
        else:
            local_dt = dt

        hour = local_dt.hour

        for category, (start_hour, end_hour) in TIME_OF_DAY_CATEGORIES.items():
            if start_hour <= hour < end_hour:
                return category

        return "unknown"

    def get_activity_period(self, dt: datetime) -> str:
        """
        Определение периода активности

        Args:
            dt: Временная метка

        Returns:
            Период активности
        """
        # Конвертируем в локальное время если определена зона
        if self.detected_timezone and self.detected_timezone != "UTC":
            local_dt = dt.astimezone(ZoneInfo(self.detected_timezone))
        else:
            local_dt = dt

        weekday = local_dt.weekday()

        if weekday in ACTIVITY_PERIODS["workday"]:
            return "workday"
        elif weekday in ACTIVITY_PERIODS["weekend"]:
            return "weekend"
        else:
            return "unknown"

    def predict_next_activity(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Предсказание следующей активности

        Args:
            messages: Список сообщений

        Returns:
            Предсказания активности
        """
        if not messages:
            return {}

        # Анализируем паттерны если еще не сделано
        if not self.activity_patterns:
            self.analyze_activity_patterns(messages)

        patterns = self.activity_patterns
        predictions = {}

        # Предсказание следующего пикового часа
        peak_hours = patterns.get("peak_hours", [])
        if peak_hours:
            # Берем самый активный час
            top_hour = peak_hours[0]["hour"]
            predictions["next_peak_hour"] = top_hour

        # Предсказание следующего дня активности
        day_distribution = patterns.get("day_of_week_distribution", {})
        if day_distribution:
            # Находим самый активный день
            most_active_day = max(day_distribution.items(), key=lambda x: x[1])
            predictions["most_active_day"] = most_active_day[0]

        # Предсказание времени следующей сессии
        session_patterns = patterns.get("session_patterns", {})
        if session_patterns:
            avg_duration = session_patterns.get("avg_session_duration_hours", 0)
            predictions["expected_session_duration_hours"] = avg_duration

        return predictions

    def get_timezone_aware_timestamp(self, dt: datetime) -> datetime:
        """
        Получение временной метки с учетом определенной временной зоны

        Args:
            dt: Временная метка в UTC

        Returns:
            Временная метка в локальной зоне
        """
        if self.detected_timezone and self.detected_timezone != "UTC":
            return dt.astimezone(ZoneInfo(self.detected_timezone))
        return dt

    def format_timestamp(self, dt: datetime, format_type: str = "human") -> str:
        """
        Форматирование временной метки

        Args:
            dt: Временная метка
            format_type: Тип форматирования (human, iso, short)

        Returns:
            Отформатированная строка
        """
        local_dt = self.get_timezone_aware_timestamp(dt)

        if format_type == "human":
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        elif format_type == "iso":
            return local_dt.isoformat()
        elif format_type == "short":
            return local_dt.strftime("%d.%m %H:%M")
        else:
            return str(local_dt)


if __name__ == "__main__":
    # Тест модуля
    processor = TimeProcessor()
    
    # Тестовые сообщения
    test_messages = [
        {"date_utc": "2024-01-15T09:30:00Z", "text": "Утреннее сообщение"},
        {"date_utc": "2024-01-15T14:15:00Z", "text": "Дневное сообщение"},
        {"date_utc": "2024-01-15T20:45:00Z", "text": "Вечернее сообщение"},
        {"date_utc": "2024-01-16T10:00:00Z", "text": "Следующий день"},
    ]
    
    # Определяем временную зону
    timezone = processor.detect_timezone(test_messages)
    print(f"Определенная временная зона: {timezone}")
    
    # Анализируем паттерны
    patterns = processor.analyze_activity_patterns(test_messages)
    print(f"Паттерны активности: {patterns}")
    
    # Предсказываем активность
    predictions = processor.predict_next_activity(test_messages)
    print(f"Предсказания: {predictions}")
