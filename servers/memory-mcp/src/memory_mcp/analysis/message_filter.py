#!/usr/bin/env python3
"""
Модуль для фильтрации и дедупликации сообщений
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..utils.deduplication import deduplicate_consecutive

logger = logging.getLogger(__name__)


class MessageFilter:
    """Класс для фильтрации и дедупликации сообщений"""

    def __init__(
        self,
        min_text_length: int = 3,
        similarity_threshold: float = 0.85,
        max_consecutive_duplicates: int = 1,
    ):
        """
        Инициализация фильтра

        Args:
            min_text_length: Минимальная длина текста сообщения
            similarity_threshold: Порог схожести для дедупликации (0.0-1.0)
            max_consecutive_duplicates: Максимум подряд идущих похожих сообщений
        """
        self.min_text_length = min_text_length
        self.similarity_threshold = similarity_threshold
        self.max_consecutive_duplicates = max_consecutive_duplicates

    def filter_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Фильтрация сообщений с удалением пустых и дедупликацией

        Args:
            messages: Список сообщений

        Returns:
            Отфильтрованный список сообщений
        """
        if not messages:
            return []

        filtered = []
        stats = {
            "total": len(messages),
            "empty": 0,
            "too_short": 0,
            "duplicates": 0,
            "bot_spam": 0,
            "service": 0,
        }

        for msg in messages:
            # Пропускаем пустые сообщения
            if self._is_empty_message(msg):
                stats["empty"] += 1
                continue

            # Пропускаем сервисные сообщения
            if self._is_service_message(msg):
                stats["service"] += 1
                continue

            # Пропускаем слишком короткие сообщения
            text = self._get_message_text(msg)
            if text and len(text.strip()) < self.min_text_length:
                stats["too_short"] += 1
                continue

            # Пропускаем спам от ботов
            if self._is_bot_spam(msg):
                stats["bot_spam"] += 1
                continue

            filtered.append(msg)

        # Дедупликация последовательных похожих сообщений
        deduplicated = deduplicate_consecutive(
            filtered,
            threshold=self.similarity_threshold,
            max_consecutive=self.max_consecutive_duplicates,
            get_text_func=self._get_message_text,
        )
        stats["duplicates"] = len(filtered) - len(deduplicated)

        logger.info(
            f"Фильтрация: {stats['total']} → {len(deduplicated)} сообщений "
            f"(пустых: {stats['empty']}, коротких: {stats['too_short']}, "
            f"дублей: {stats['duplicates']}, спама: {stats['bot_spam']}, "
            f"сервисных: {stats['service']})"
        )

        return deduplicated

    def _is_empty_message(self, msg: Dict[str, Any]) -> bool:
        """
        Проверка, является ли сообщение пустым

        Args:
            msg: Сообщение

        Returns:
            True если сообщение пустое
        """
        text = self._get_message_text(msg)

        # Нет текста
        if not text or not text.strip():
            # Проверяем, есть ли медиа, файлы или другой контент
            if not msg.get("file") and not msg.get("media_type"):
                return True

        return False

    def _is_service_message(self, msg: Dict[str, Any]) -> bool:
        """
        Проверка, является ли сообщение сервисным

        Args:
            msg: Сообщение

        Returns:
            True если это сервисное сообщение
        """
        # Проверяем тип действия
        action = msg.get("action")
        if action:
            # Системные действия (вход/выход из чата, смена названия и т.д.)
            service_actions = [
                "invite_members",
                "remove_members",
                "pin_message",
                "create_group",
                "migrate_to_supergroup",
                "phone_call",
            ]
            if action in service_actions:
                return True

        # Проверяем текст на служебные паттерны
        text = self._get_message_text(msg)
        if text:
            service_patterns = [
                r"^joined the (group|channel)$",
                r"^left the (group|channel)$",
                r"^pinned a message$",
                r"changed the (group|channel) (photo|title|description)",
            ]
            for pattern in service_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True

        return False

    def _is_bot_spam(self, msg: Dict[str, Any]) -> bool:
        """
        Проверка на спам от ботов

        Args:
            msg: Сообщение

        Returns:
            True если это спам от бота
        """
        text = self._get_message_text(msg)
        if not text:
            return False

        # Паттерны спама
        spam_patterns = [
            r"Вы получили \d+ звёзд",  # Уведомления о звёздах
            r"You received \d+ stars",
            r"^\/start$",  # Команды запуска бота
            r"^\/help$",
            r"🎁 Новый подарок получен",  # Уведомления о подарках
            r"🎁 New gift received",
            r"^Подарок отправлен$",
            r"^Gift sent$",
        ]

        for pattern in spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Проверяем, от бота ли сообщение
        from_user = msg.get("from", {})
        if isinstance(from_user, dict):
            # Если username содержит 'bot' или 'Bot'
            username = from_user.get("username") or ""
            if (
                username
                and "bot" in username.lower()
                and username.lower().endswith("bot")
            ):
                # Короткие однотипные сообщения от ботов
                if len(text.strip()) < 50 and (
                    text.startswith("✅")
                    or text.startswith("❌")
                    or text.startswith("⚠️")
                    or re.match(
                        r"^[\d\s\.\,\+\-\*\/\=\(\)]+$", text
                    )  # Только цифры и символы
                ):
                    return True

        return False

    # Методы _deduplicate_consecutive, _is_similar и _normalize_text удалены,
    # так как теперь используются функции из utils.deduplication

    def _get_message_text(self, msg: Dict[str, Any]) -> str:
        """
        Получение текста сообщения

        Args:
            msg: Сообщение

        Returns:
            Текст сообщения
        """
        text = msg.get("text", "")

        # Если текст - список (форматированный текст)
        if isinstance(text, list):
            text_parts = []
            for item in text:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
            text = "".join(text_parts)

        return text if isinstance(text, str) else str(text)


def filter_and_deduplicate(
    messages: List[Dict[str, Any]], min_length: int = 3, similarity: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Удобная функция для фильтрации и дедупликации сообщений

    Args:
        messages: Список сообщений
        min_length: Минимальная длина текста
        similarity: Порог схожести для дедупликации

    Returns:
        Отфильтрованный список
    """
    filter_obj = MessageFilter(
        min_text_length=min_length, similarity_threshold=similarity
    )
    return filter_obj.filter_messages(messages)


if __name__ == "__main__":
    # Тест модуля
    test_messages = [
        {"id": "1", "text": "Привет!", "from": {"username": "user1"}},
        {"id": "2", "text": "   ", "from": {"username": "user2"}},  # Пустое
        {"id": "3", "text": "Привет!", "from": {"username": "user1"}},  # Дубль
        {"id": "4", "text": "Привет!!!", "from": {"username": "user1"}},  # Похожее
        {"id": "5", "text": "Как дела?", "from": {"username": "user2"}},
        {"id": "6", "text": "a", "from": {"username": "user3"}},  # Слишком короткое
        {"id": "7", "text": "Всё отлично!", "from": {"username": "user1"}},
        {"id": "8", "text": "Вы получили 5 звёзд", "from": {"username": "bot"}},  # Спам
    ]

    filter_obj = MessageFilter()
    filtered = filter_obj.filter_messages(test_messages)

    print(f"Исходных сообщений: {len(test_messages)}")
    print(f"После фильтрации: {len(filtered)}")
    print("\nОставлено сообщений:")
    for msg in filtered:
        dup_count = msg.get("_duplicate_count", 0)
        dup_marker = f" [+{dup_count} похожих]" if dup_count > 0 else ""
        print(f"  {msg['id']}: {msg['text']}{dup_marker}")
