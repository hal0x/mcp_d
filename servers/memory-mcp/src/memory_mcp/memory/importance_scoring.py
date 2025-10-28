#!/usr/bin/env python3
"""
Система оценки важности сообщений и eviction scoring

Вдохновлено архитектурой памяти HALv1:
- Importance Score - насколько сообщение важно для сохранения
- Eviction Score - приоритет удаления (выше = быстрее удалить)
- Time Decay - учёт возраста данных
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Вычисление importance score для сообщений"""

    def __init__(
        self,
        entity_weight: float = 0.1,
        task_weight: float = 0.3,
        length_weight: float = 0.2,
        search_hits_weight: float = 0.4,
    ):
        """
        Инициализация scorer

        Args:
            entity_weight: Вес за каждую сущность
            task_weight: Вес за наличие задачи
            length_weight: Вес за длину сообщения
            search_hits_weight: Вес за частоту поиска
        """
        self.entity_weight = entity_weight
        self.task_weight = task_weight
        self.length_weight = length_weight
        self.search_hits_weight = search_hits_weight

        logger.info(
            f"Инициализирован ImportanceScorer (entity={entity_weight}, "
            f"task={task_weight}, length={length_weight}, "
            f"search_hits={search_hits_weight})"
        )

    def compute_importance(
        self, message: Dict[str, Any], metadata: Optional[Dict] = None
    ) -> float:
        """
        Вычисление importance score для сообщения

        Score от 0.0 (не важно) до 1.0 (очень важно)

        Факторы важности:
        - Наличие сущностей (люди, проекты, термины)
        - Наличие action items / задач
        - Длина сообщения (длинные = более содержательные)
        - Частота поиска (часто находимые = важные)
        - Наличие ссылок, упоминаний, хештегов

        Args:
            message: Сообщение (dict с полями text, entities и т.д.)
            metadata: Дополнительные метаданные (search_hits и т.д.)

        Returns:
            Importance score (0.0 - 1.0)
        """
        score = 0.0

        # 1. Сущности (entities)
        entities = message.get("entities", [])
        if entities:
            # Каждая сущность добавляет вес
            entity_score = min(len(entities) * self.entity_weight, 0.5)
            score += entity_score

        # 2. Задачи (action items)
        has_task = message.get("has_task", False) or message.get(
            "is_action_item", False
        )
        if has_task:
            score += self.task_weight

        # 3. Длина сообщения
        text = message.get("text", "")
        text_length = len(text)
        if text_length > 500:
            score += self.length_weight
        elif text_length > 200:
            score += self.length_weight * 0.5

        # 4. Частота поиска (из метаданных)
        if metadata:
            search_hits = metadata.get("_search_hits", 0)
            if search_hits > 0:
                # Нормализуем: 10+ поисков = максимальный вес
                search_score = min(search_hits / 10.0, 1.0) * self.search_hits_weight
                score += search_score

        # 5. Специальные паттерны
        special_score = self._compute_special_patterns(text)
        score += special_score

        # Нормализуем к диапазону [0, 1]
        score = min(score, 1.0)

        return score

    def _compute_special_patterns(self, text: str) -> float:
        """
        Дополнительные баллы за специальные паттерны в тексте

        Args:
            text: Текст сообщения

        Returns:
            Дополнительный score
        """
        bonus = 0.0

        # Ссылки (http, https)
        if re.search(r"https?://", text):
            bonus += 0.05

        # Упоминания (@username)
        mentions = re.findall(r"@\w+", text)
        if mentions:
            bonus += min(len(mentions) * 0.02, 0.1)

        # Хештеги (#hashtag)
        hashtags = re.findall(r"#\w+", text)
        if hashtags:
            bonus += min(len(hashtags) * 0.02, 0.1)

        # Вопросы (?)
        if "?" in text:
            bonus += 0.03

        # Код (тройные обратные кавычки или множество специальных символов)
        if "```" in text or text.count("{") > 3 or text.count("(") > 5:
            bonus += 0.05

        # Числа (потенциально важные данные)
        numbers = re.findall(r"\d+", text)
        if len(numbers) > 3:
            bonus += 0.03

        return min(bonus, 0.3)  # Максимум 0.3 за паттерны


class EvictionScorer:
    """Вычисление eviction score для управления памятью"""

    def __init__(self, importance_scorer: Optional[ImportanceScorer] = None):
        """
        Инициализация eviction scorer

        Args:
            importance_scorer: ImportanceScorer для вычисления важности
        """
        self.importance_scorer = importance_scorer or ImportanceScorer()

        logger.info("Инициализирован EvictionScorer")

    def compute_eviction_score(
        self,
        message: Dict[str, Any],
        metadata: Optional[Dict] = None,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Вычисление eviction score

        Формула (из HALv1):
        evict_score = (1 - importance) * time_decay * (1 - usage_freq)

        Где:
        - importance: важность сообщения (0-1)
        - time_decay: 1 / (1 + age_days)
        - usage_freq: normalized search_hits

        Высокий eviction_score = кандидат на удаление

        Args:
            message: Сообщение
            metadata: Метаданные (search_hits, etc.)
            current_time: Текущее время (для вычисления age)

        Returns:
            Eviction score (0.0 - 1.0, выше = приоритет удаления)
        """
        # 1. Importance (чем меньше, тем выше eviction)
        importance = self.importance_scorer.compute_importance(message, metadata)

        # 2. Time decay (старые сообщения = выше eviction)
        time_decay = self._compute_time_decay(message, current_time)

        # 3. Usage frequency (редко используемые = выше eviction)
        usage_freq = self._compute_usage_frequency(metadata)

        # Итоговая формула
        eviction_score = (1.0 - importance) * time_decay * (1.0 - usage_freq)

        return eviction_score

    def _compute_time_decay(
        self, message: Dict[str, Any], current_time: Optional[datetime] = None
    ) -> float:
        """
        Вычисление time decay

        time_decay = 1 / (1 + age_days)

        Args:
            message: Сообщение
            current_time: Текущее время

        Returns:
            Time decay (0-1)
        """
        if current_time is None:
            current_time = datetime.now()

        # Получаем дату сообщения
        date_str = message.get("date_utc") or message.get("date", "")
        if not date_str:
            return 1.0  # Если нет даты, считаем старым

        try:
            # Парсим дату
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"

            msg_time = datetime.fromisoformat(date_str)

            # Убираем timezone для вычислений
            if msg_time.tzinfo is not None:
                msg_time = msg_time.replace(tzinfo=None)
            if current_time.tzinfo is not None:
                current_time = current_time.replace(tzinfo=None)

            # Возраст в днях
            age = current_time - msg_time
            age_days = max(age.days, 0)

            # Time decay: 1 / (1 + age_days)
            # Свежие сообщения (0 дней) = 1.0
            # Месячные (30 дней) = 0.032
            # Годовалые (365 дней) = 0.0027
            time_decay = 1.0 / (1.0 + age_days)

            return time_decay

        except Exception as e:
            logger.warning(f"Ошибка парсинга даты '{date_str}': {e}")
            return 1.0

    def _compute_usage_frequency(self, metadata: Optional[Dict] = None) -> float:
        """
        Вычисление usage frequency (normalized)

        Args:
            metadata: Метаданные с search_hits

        Returns:
            Usage frequency (0-1)
        """
        if not metadata:
            return 0.0

        search_hits = metadata.get("_search_hits", 0)

        # Нормализуем: 100+ поисков = 1.0
        usage_freq = min(search_hits / 100.0, 1.0)

        return usage_freq

    def get_eviction_candidates(
        self,
        messages: List[Dict[str, Any]],
        metadata_map: Optional[Dict[str, Dict]] = None,
        threshold: float = 0.7,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получить кандидатов на удаление

        Args:
            messages: Список сообщений
            metadata_map: Словарь {msg_id: metadata}
            threshold: Порог eviction score (выше = удалить)
            limit: Максимум кандидатов

        Returns:
            Список кандидатов с eviction_score
        """
        candidates = []
        current_time = datetime.now()

        for msg in messages:
            msg_id = msg.get("id") or msg.get("msg_id")
            metadata = metadata_map.get(msg_id) if metadata_map else None

            eviction_score = self.compute_eviction_score(msg, metadata, current_time)

            if eviction_score >= threshold:
                candidates.append(
                    {
                        "message": msg,
                        "eviction_score": eviction_score,
                        "msg_id": msg_id,
                    }
                )

        # Сортируем по убыванию eviction_score
        candidates.sort(key=lambda x: x["eviction_score"], reverse=True)

        if limit:
            candidates = candidates[:limit]

        return candidates


class MemoryPruner:
    """Автоматическая очистка памяти на основе eviction scores"""

    def __init__(
        self,
        eviction_scorer: Optional[EvictionScorer] = None,
        max_messages: int = 100000,
        eviction_threshold: float = 0.7,
    ):
        """
        Инициализация pruner

        Args:
            eviction_scorer: EvictionScorer
            max_messages: Максимум сообщений в памяти
            eviction_threshold: Порог для удаления
        """
        self.eviction_scorer = eviction_scorer or EvictionScorer()
        self.max_messages = max_messages
        self.eviction_threshold = eviction_threshold

        logger.info(
            f"Инициализирован MemoryPruner (max={max_messages}, "
            f"threshold={eviction_threshold})"
        )

    def should_prune(self, current_count: int) -> bool:
        """
        Проверка, нужна ли очистка

        Args:
            current_count: Текущее количество сообщений

        Returns:
            True если нужна очистка
        """
        return current_count > self.max_messages

    def prune_messages(
        self,
        messages: List[Dict[str, Any]],
        metadata_map: Optional[Dict[str, Dict]] = None,
        target_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Очистка сообщений

        Args:
            messages: Список сообщений
            metadata_map: Метаданные сообщений
            target_count: Целевое количество (None = max_messages)

        Returns:
            Результаты очистки
        """
        if target_count is None:
            target_count = self.max_messages

        current_count = len(messages)

        if current_count <= target_count:
            logger.info(f"Очистка не требуется: {current_count} <= {target_count}")
            return {
                "pruned": False,
                "removed_count": 0,
                "remaining_count": current_count,
            }

        # Сколько нужно удалить
        to_remove = current_count - target_count

        logger.info(
            f"Начало очистки: {current_count} → {target_count} "
            f"(удалить {to_remove})"
        )

        # Получаем кандидатов на удаление
        candidates = self.eviction_scorer.get_eviction_candidates(
            messages, metadata_map, threshold=self.eviction_threshold, limit=to_remove
        )

        if len(candidates) < to_remove:
            # Недостаточно кандидатов по порогу, снижаем порог
            logger.warning(
                f"Недостаточно кандидатов ({len(candidates)} < {to_remove}), "
                f"снижаем порог"
            )
            candidates = self.eviction_scorer.get_eviction_candidates(
                messages, metadata_map, threshold=0.0, limit=to_remove
            )

        # Формируем результат
        removed_ids = [c["msg_id"] for c in candidates]

        logger.info(f"Очистка завершена: удалено {len(removed_ids)} сообщений")

        return {
            "pruned": True,
            "removed_count": len(removed_ids),
            "removed_ids": removed_ids,
            "remaining_count": current_count - len(removed_ids),
            "candidates": candidates,  # Для архивации
        }
