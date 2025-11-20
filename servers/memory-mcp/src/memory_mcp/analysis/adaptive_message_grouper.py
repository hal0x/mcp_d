"""Адаптивная группировка сообщений для эффективного использования большого контекста LLM."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class AdaptiveMessageGrouper:
    """
    Адаптивная группировка сообщений для эффективного использования большого контекста LLM.
    
    Параметры модели:
    - Максимальный контекст: 131072 токенов (gpt-oss-20b)
    - Резерв для промпта: ~5000 токенов
    - Доступно для сообщений: ~126000 токенов
    - Средний размер сообщения: ~50-200 токенов
    - Оптимальный размер группы: 500-2000 сообщений (в зависимости от длины)
    """

    def __init__(
        self,
        max_tokens: int = 25000,  # Доступно для сообщений (30000 - 5000 резерв)
        prompt_reserve_tokens: int = 5000,
        avg_tokens_per_char: float = 0.25,  # ~4 символа на токен
        min_group_size_tokens: int = 50000,  # Минимум для эффективного использования
        strategy: str = "hybrid",  # "hybrid", "temporal", "quantitative", "semantic"
    ):
        """
        Инициализация адаптивного группировщика.

        Args:
            max_tokens: Максимальное количество токенов для одной группы
            prompt_reserve_tokens: Резерв токенов для промпта
            avg_tokens_per_char: Среднее количество токенов на символ
            min_group_size_tokens: Минимальный размер группы для эффективного использования
            strategy: Стратегия группировки ("hybrid", "temporal", "quantitative")
        """
        self.max_tokens = max_tokens
        self.prompt_reserve_tokens = prompt_reserve_tokens
        self.avg_tokens_per_char = avg_tokens_per_char
        self.min_group_size_tokens = min_group_size_tokens
        self.strategy = strategy

    def estimate_tokens(self, text: str) -> int:
        """
        Оценка количества токенов в тексте.

        Args:
            text: Текст для оценки

        Returns:
            Примерное количество токенов
        """
        if not text:
            return 0
        return int(len(text) * self.avg_tokens_per_char)

    def estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Оценка количества токенов в сообщении.

        Args:
            message: Сообщение для оценки

        Returns:
            Примерное количество токенов
        """
        tokens = 0

        # Основной текст сообщения
        content = message.get("text") or message.get("content") or ""
        tokens += self.estimate_tokens(content)

        # Автор (если есть)
        author = message.get("from") or message.get("author")
        if author:
            tokens += self.estimate_tokens(str(author)) + 5  # +5 для метаданных

        # Вложения (если есть)
        if "media" in message or "attachments" in message:
            tokens += 20  # Примерная оценка для вложений

        return tokens

    def estimate_group_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Оценка общего количества токенов в группе сообщений.

        Args:
            messages: Список сообщений

        Returns:
            Общее количество токенов
        """
        total = 0
        for msg in messages:
            total += self.estimate_message_tokens(msg)
        # Добавляем накладные расходы на форматирование
        total += len(messages) * 5  # ~5 токенов на сообщение для форматирования
        return total

    def _parse_message_time(self, message: Dict[str, Any]) -> datetime:
        """Парсинг времени сообщения."""
        time_str = message.get("date") or message.get("timestamp")
        if isinstance(time_str, datetime):
            return time_str
        if isinstance(time_str, str):
            try:
                return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except Exception:
                pass
        return datetime.now()

    def group_messages_adaptively(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Адаптивная группировка сообщений для эффективного использования большого контекста.

        Алгоритм:
        1. Оценить размер всех сообщений в токенах
        2. Если total < max_tokens → одна группа
        3. Если total > max_tokens:
           a) Вычислить средний размер сообщения
           b) Определить оптимальное количество сообщений на группу
           c) Вычислить временные окна (недели/месяцы)
           d) Создать группы:
              - Приоритет временным окнам (сохраняет контекст)
              - Ограничение по количеству (не превышает max_tokens)
              - Объединение маленьких соседних групп

        Args:
            messages: Список сообщений для группировки
            chat_name: Название чата (опционально)

        Returns:
            Список групп сообщений
        """
        if not messages:
            return []

        # Сортируем сообщения по времени
        sorted_messages = sorted(messages, key=lambda x: self._parse_message_time(x))

        # Оцениваем общий размер
        total_tokens = self.estimate_group_tokens(sorted_messages)

        logger.info(
            f"Группировка {len(sorted_messages)} сообщений "
            f"(~{total_tokens} токенов) для чата {chat_name}"
        )

        # Если все помещается в один запрос
        if total_tokens <= self.max_tokens:
            logger.info(f"Все сообщения помещаются в одну группу ({total_tokens} токенов)")
            return [sorted_messages]

        # Выбираем стратегию группировки
        if self.strategy == "hybrid":
            return self._group_hybrid(sorted_messages, chat_name)
        elif self.strategy == "temporal":
            return self._group_temporal(sorted_messages, chat_name)
        elif self.strategy == "quantitative":
            return self._group_quantitative(sorted_messages, chat_name)
        else:
            logger.warning(f"Неизвестная стратегия {self.strategy}, используем hybrid")
            return self._group_hybrid(sorted_messages, chat_name)

    def _group_hybrid(
        self, messages: List[Dict[str, Any]], chat_name: Optional[str]
    ) -> List[List[Dict[str, Any]]]:
        """
        Гибридная стратегия: комбинация временной и количественной группировки.

        Алгоритм:
        1. Вычисляем средний размер сообщения
        2. Определяем оптимальное количество сообщений на группу
        3. Группируем по временным окнам (недели/месяцы)
        4. Ограничиваем размер групп по количеству токенов
        5. Объединяем маленькие соседние группы
        """
        # Вычисляем средний размер сообщения
        total_tokens = self.estimate_group_tokens(messages)
        avg_tokens_per_message = total_tokens / len(messages) if messages else 0

        # Определяем оптимальное количество сообщений на группу
        optimal_messages_per_group = int(self.max_tokens / avg_tokens_per_message) if avg_tokens_per_message > 0 else 1000
        optimal_messages_per_group = max(500, min(optimal_messages_per_group, 2000))

        logger.info(
            f"Средний размер сообщения: {avg_tokens_per_message:.1f} токенов, "
            f"оптимально {optimal_messages_per_group} сообщений на группу"
        )

        # Определяем временное окно на основе активности
        activity = self._estimate_activity(messages)
        if activity == "high":
            time_window = "week"  # Для активных чатов - недели
        elif activity == "medium":
            time_window = "month"  # Для средних - месяцы
        else:
            time_window = "month"  # Для неактивных - месяцы

        logger.info(f"Выбрано временное окно: {time_window} (активность: {activity})")

        # Группируем по временным окнам
        temporal_groups = self._group_by_time_window(messages, time_window)

        # Ограничиваем размер групп по токенам
        final_groups = []
        for group in temporal_groups:
            group_tokens = self.estimate_group_tokens(group)

            if group_tokens <= self.max_tokens:
                # Группа помещается целиком
                final_groups.append(group)
            else:
                # Разбиваем группу по количеству сообщений
                logger.info(
                    f"Группа слишком большая ({group_tokens} токенов), "
                    f"разбиваем на части по {optimal_messages_per_group} сообщений"
                )
                split_groups = self._split_by_count(group, optimal_messages_per_group)
                final_groups.extend(split_groups)

        # Объединяем маленькие соседние группы
        final_groups = self._merge_small_groups(final_groups)

        logger.info(
            f"Создано {len(final_groups)} групп из {len(messages)} сообщений "
            f"(гибридная стратегия)"
        )

        return final_groups

    def _group_temporal(
        self, messages: List[Dict[str, Any]], chat_name: Optional[str]
    ) -> List[List[Dict[str, Any]]]:
        """Временная группировка по неделям/месяцам."""
        activity = self._estimate_activity(messages)
        if activity == "high":
            time_window = "week"
        else:
            time_window = "month"

        groups = self._group_by_time_window(messages, time_window)

        # Разбиваем слишком большие группы
        final_groups = []
        for group in groups:
            group_tokens = self.estimate_group_tokens(group)
            if group_tokens <= self.max_tokens:
                final_groups.append(group)
            else:
                # Разбиваем на более мелкие временные окна
                if time_window == "month":
                    split_groups = self._group_by_time_window(group, "week")
                else:
                    # Разбиваем по дням
                    split_groups = self._group_by_time_window(group, "day")
                final_groups.extend(split_groups)

        logger.info(
            f"Создано {len(final_groups)} групп из {len(messages)} сообщений "
            f"(временная стратегия: {time_window})"
        )

        return final_groups

    def _group_quantitative(
        self, messages: List[Dict[str, Any]], chat_name: Optional[str]
    ) -> List[List[Dict[str, Any]]]:
        """Количественная группировка по фиксированному количеству сообщений."""
        # Вычисляем оптимальное количество сообщений на группу
        total_tokens = self.estimate_group_tokens(messages)
        avg_tokens_per_message = total_tokens / len(messages) if messages else 0
        messages_per_group = int(self.max_tokens / avg_tokens_per_message) if avg_tokens_per_message > 0 else 1000
        messages_per_group = max(500, min(messages_per_group, 2000))

        groups = self._split_by_count(messages, messages_per_group)

        logger.info(
            f"Создано {len(groups)} групп из {len(messages)} сообщений "
            f"(количественная стратегия: {messages_per_group} сообщений на группу)"
        )

        return groups

    def _estimate_activity(self, messages: List[Dict[str, Any]]) -> str:
        """
        Оценка активности чата на основе сообщений.

        Returns:
            "high", "medium", "low"
        """
        if not messages:
            return "low"

        # Вычисляем среднее количество сообщений в день
        if len(messages) < 2:
            return "low"

        first_time = self._parse_message_time(messages[0])
        last_time = self._parse_message_time(messages[-1])
        days = (last_time - first_time).days + 1

        if days == 0:
            return "high"

        messages_per_day = len(messages) / days

        if messages_per_day > 50:
            return "high"
        elif messages_per_day > 10:
            return "medium"
        else:
            return "low"

    def _group_by_time_window(
        self, messages: List[Dict[str, Any]], window: str
    ) -> List[List[Dict[str, Any]]]:
        """
        Группировка сообщений по временным окнам.

        Args:
            messages: Список сообщений
            window: "day", "week", "month"

        Returns:
            Список групп сообщений
        """
        if not messages:
            return []

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_window_start: Optional[datetime] = None

        for msg in messages:
            msg_time = self._parse_message_time(msg)

            # Определяем начало текущего окна
            if window == "day":
                window_start = msg_time.replace(hour=0, minute=0, second=0, microsecond=0)
            elif window == "week":
                # Начало недели (понедельник)
                days_since_monday = msg_time.weekday()
                window_start = (msg_time - timedelta(days=days_since_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            elif window == "month":
                window_start = msg_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                window_start = msg_time

            # Если это новое окно, начинаем новую группу
            if current_window_start is None or window_start != current_window_start:
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
                current_window_start = window_start
            else:
                current_group.append(msg)

        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)

        return groups

    def _split_by_count(
        self, messages: List[Dict[str, Any]], count: int
    ) -> List[List[Dict[str, Any]]]:
        """Разбиение группы на части по количеству сообщений."""
        groups = []
        for i in range(0, len(messages), count):
            groups.append(messages[i : i + count])
        return groups

    def _merge_small_groups(
        self, groups: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Объединение маленьких соседних групп для эффективного использования контекста.

        Объединяем группы, если:
        - Размер группы < min_group_size_tokens
        - Соседняя группа тоже маленькая
        - Объединенная группа <= max_tokens
        """
        if not groups:
            return []

        merged_groups: List[List[Dict[str, Any]]] = []
        i = 0

        while i < len(groups):
            current_group = groups[i]
            current_tokens = self.estimate_group_tokens(current_group)

            # Если группа достаточно большая, оставляем как есть
            if current_tokens >= self.min_group_size_tokens:
                merged_groups.append(current_group)
                i += 1
                continue

            # Пытаемся объединить с соседними группами
            merged = current_group.copy()
            merged_tokens = current_tokens
            j = i + 1

            while j < len(groups):
                next_group = groups[j]
                next_tokens = self.estimate_group_tokens(next_group)
                combined_tokens = merged_tokens + next_tokens

                # Если объединение не превышает лимит, объединяем
                if combined_tokens <= self.max_tokens:
                    merged.extend(next_group)
                    merged_tokens = combined_tokens
                    j += 1
                else:
                    break

            merged_groups.append(merged)
            i = j

        logger.info(
            f"Объединено {len(groups)} групп в {len(merged_groups)} "
            f"(минимум {self.min_group_size_tokens} токенов на группу)"
        )

        return merged_groups

