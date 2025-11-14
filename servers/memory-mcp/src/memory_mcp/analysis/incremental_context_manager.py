#!/usr/bin/env python3
"""
Модуль для расширенного контекста при доиндексации
Обеспечивает загрузку предыдущих сообщений и глобального контекста чата
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import chromadb

logger = logging.getLogger(__name__)


class IncrementalContextManager:
    """
    Менеджер контекста для инкрементальной индексации

    Особенности:
    1. Загружает предыдущие сообщения для контекста
    2. Интегрирует глобальный контекст чата
    3. Предоставляет расширенный контекст для малых сессий
    """

    def __init__(self, chroma_path: str = "./chroma_db"):
        """
        Инициализация менеджера контекста

        Args:
            chroma_path: Путь к ChromaDB
        """
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Инициализируем коллекции только если они существуют
        try:
            self.messages_collection = self.chroma_client.get_collection(
                "chat_messages"
            )
        except Exception:
            self.messages_collection = None

        try:
            self.sessions_collection = self.chroma_client.get_collection(
                "chat_sessions"
            )
        except Exception:
            self.sessions_collection = None

    def get_extended_context_for_session(
        self,
        chat_name: str,
        session_messages: List[Dict[str, Any]],
        context_window_hours: int = 24,
        max_previous_messages: int = 50,
    ) -> Dict[str, Any]:
        """
        Получает расширенный контекст для сессии при доиндексации

        Args:
            chat_name: Название чата
            session_messages: Сообщения текущей сессии
            context_window_hours: Окно времени для поиска предыдущих сообщений (часы)
            max_previous_messages: Максимальное количество предыдущих сообщений

        Returns:
            Словарь с расширенным контекстом
        """
        if not session_messages:
            return self._empty_context()

        # Определяем временные границы для поиска предыдущих сообщений
        session_start_time = self._get_session_start_time(session_messages)
        context_start_time = session_start_time - timedelta(hours=context_window_hours)

        logger.info(
            f"Поиск контекста для сессии {chat_name}: "
            f"с {context_start_time} до {session_start_time}"
        )

        # Загружаем предыдущие сообщения из ChromaDB
        previous_messages = self._load_previous_messages(
            chat_name, context_start_time, session_start_time, max_previous_messages
        )

        # Загружаем глобальный контекст чата
        chat_context = self._load_chat_context(chat_name)

        # Загружаем предыдущие сессии
        previous_sessions = self._load_previous_sessions(chat_name, session_start_time)

        # Формируем расширенный контекст
        extended_context = {
            "previous_messages": previous_messages,
            "previous_messages_count": len(previous_messages),
            "chat_context": chat_context,
            "previous_sessions": previous_sessions,
            "previous_sessions_count": len(previous_sessions),
            "context_window_hours": context_window_hours,
            "session_start_time": session_start_time.isoformat(),
            "context_start_time": context_start_time.isoformat(),
        }

        logger.info(
            f"Загружен контекст: {len(previous_messages)} предыдущих сообщений, "
            f"{len(previous_sessions)} предыдущих сессий"
        )

        return extended_context

    def _get_session_start_time(
        self, session_messages: List[Dict[str, Any]]
    ) -> datetime:
        """Получает время начала сессии"""
        if not session_messages:
            return datetime.now(ZoneInfo("UTC"))

        # Находим самое раннее сообщение в сессии
        earliest_time = min(self._parse_message_time(msg) for msg in session_messages)
        return earliest_time

    def _parse_message_time(self, message: Dict[str, Any]) -> datetime:
        """Парсит время сообщения (использует общую утилиту)."""
        from ..utils.datetime_utils import parse_message_time

        return parse_message_time(message, use_zoneinfo=True)

    def _load_previous_messages(
        self,
        chat_name: str,
        start_time: datetime,
        end_time: datetime,
        max_messages: int,
    ) -> List[Dict[str, Any]]:
        """
        Загружает предыдущие сообщения из ChromaDB

        Args:
            chat_name: Название чата
            start_time: Начало временного окна
            end_time: Конец временного окна
            max_messages: Максимальное количество сообщений

        Returns:
            Список предыдущих сообщений
        """
        if not self.messages_collection:
            logger.debug("Коллекция сообщений не доступна")
            return []

        try:
            # Запрашиваем сообщения из ChromaDB
            # ChromaDB не поддерживает сложные where условия, поэтому используем простой фильтр
            results = self.messages_collection.get(
                where={"chat": chat_name},
                limit=max_messages * 2,  # Берем больше, чтобы отфильтровать по времени
                include=["metadatas", "documents"],
            )

            # Преобразуем результаты в нужный формат и фильтруем по времени
            previous_messages = []
            for i, metadata in enumerate(results["metadatas"]):
                message = {
                    "text": results["documents"][i],
                    "date_utc": metadata.get("date_utc", ""),
                    "msg_id": metadata.get("msg_id", ""),
                    "session_id": metadata.get("session_id", ""),
                    "has_context": metadata.get("has_context", False),
                }

                # Фильтруем по времени
                message_time = self._parse_message_time(message)
                if start_time <= message_time < end_time:
                    previous_messages.append(message)

            # Сортируем по времени (от старых к новым)
            previous_messages.sort(key=lambda x: self._parse_message_time(x))

            # Ограничиваем количество сообщений
            if len(previous_messages) > max_messages:
                previous_messages = previous_messages[:max_messages]

            logger.debug(f"Загружено {len(previous_messages)} предыдущих сообщений")
            return previous_messages

        except Exception as e:
            logger.warning(f"Ошибка при загрузке предыдущих сообщений: {e}")
            return []

    def _load_chat_context_from_path(
        self, chat_name: str, context_dir: Path
    ) -> Optional[str]:
        """
        Загружает глобальный контекст чата из указанного пути

        Args:
            chat_name: Название чата
            context_dir: Путь к директории с контекстами

        Returns:
            Контекст чата или None
        """
        try:
            context_file = context_dir / f"{chat_name}_context.md"
            if context_file.exists():
                with open(context_file, encoding="utf-8") as f:
                    content = f.read()

                    # Извлекаем образ чата (между ## Образ чата и ## Последние сессии)
                    start_marker = "## Образ чата"
                    end_marker = "## Последние сессии"

                    start_idx = content.find(start_marker)
                    end_idx = content.find(end_marker)

                    if start_idx != -1 and end_idx != -1:
                        context_section = content[
                            start_idx + len(start_marker) : end_idx
                        ].strip()
                        return context_section
                    else:
                        # Если маркеры не найдены, возвращаем весь контент
                        return content.strip()

            logger.debug(f"Контекст чата {chat_name} не найден в {context_dir}")
            return None

        except Exception as e:
            logger.warning(f"Ошибка при загрузке контекста чата {chat_name}: {e}")
            return None

    def _load_chat_context(self, chat_name: str) -> Optional[str]:
        """
        Загружает глобальный контекст чата из файла

        Args:
            chat_name: Название чата

        Returns:
            Контекст чата или None
        """
        try:
            context_file = Path("artifacts/chat_contexts") / f"{chat_name}_context.md"
            if context_file.exists():
                with open(context_file, encoding="utf-8") as f:
                    content = f.read()

                    # Извлекаем образ чата (между ## Образ чата и ## Последние сессии)
                    start_marker = "## Образ чата"
                    end_marker = "## Последние сессии"

                    start_idx = content.find(start_marker)
                    end_idx = content.find(end_marker)

                    if start_idx != -1 and end_idx != -1:
                        context_section = content[
                            start_idx + len(start_marker) : end_idx
                        ].strip()
                        return context_section
                    else:
                        # Если маркеры не найдены, возвращаем весь контент
                        return content.strip()

            logger.debug(f"Контекст чата {chat_name} не найден")
            return None

        except Exception as e:
            logger.warning(f"Ошибка при загрузке контекста чата {chat_name}: {e}")
            return None

    def _load_previous_sessions(
        self, chat_name: str, session_start_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Загружает предыдущие сессии из ChromaDB

        Args:
            chat_name: Название чата
            session_start_time: Время начала текущей сессии

        Returns:
            Список предыдущих сессий
        """
        if not self.sessions_collection:
            logger.debug("Коллекция сессий не доступна")
            return []

        try:
            # Запрашиваем сессии из ChromaDB
            # ChromaDB не поддерживает сложные where условия, поэтому используем простой фильтр
            results = self.sessions_collection.get(
                where={"chat": chat_name},
                limit=20,  # Берем больше, чтобы отфильтровать по времени
                include=["metadatas", "documents"],
            )

            # Преобразуем результаты в нужный формат и фильтруем по времени
            previous_sessions = []
            for i, metadata in enumerate(results["metadatas"]):
                session = {
                    "session_id": metadata.get("session_id", ""),
                    "summary": results["documents"][i],
                    "message_count": metadata.get("message_count", 0),
                    "quality_score": metadata.get("quality_score", 0),
                    "end_time_utc": metadata.get("end_time_utc", ""),
                    "topics_count": metadata.get("topics_count", 0),
                    "claims_count": metadata.get("claims_count", 0),
                }

                # Фильтруем по времени (только сессии, которые закончились до текущей)
                session_end_time = None

                # Пробуем получить время окончания из разных полей
                if session["end_time_utc"]:
                    from ..utils.datetime_utils import parse_datetime_utc

                    session_end_time = parse_datetime_utc(
                        session["end_time_utc"], return_none_on_error=True, use_zoneinfo=True
                    )

                # Если end_time_utc пустое, пробуем парсить из time_span
                if not session_end_time and metadata.get("time_span"):
                    try:
                        time_span = metadata["time_span"]
                        # Парсим формат "2025-05-11 16:13 – 00:58 BKK"
                        if " – " in time_span:
                            end_part = time_span.split(" – ")[1].split(" BKK")[0]
                            # Предполагаем, что это время в том же дне
                            date_part = time_span.split(" – ")[0].split(" ")[0]
                            from ..utils.datetime_utils import parse_datetime_utc

                            session_end_time = parse_datetime_utc(
                                f"{date_part}T{end_part}:00+07:00", return_none_on_error=True, use_zoneinfo=True
                            )
                    except (ValueError, TypeError, IndexError):
                        pass

                # Если время найдено, проверяем, что сессия закончилась до текущей
                if session_end_time and session_end_time < session_start_time:
                    previous_sessions.append(session)

            # Сортируем по времени (от старых к новым)
            def get_sort_key(session):
                # Пробуем получить время для сортировки
                if session.get("end_time_utc"):
                    from ..utils.datetime_utils import parse_datetime_utc

                    result = parse_datetime_utc(session["end_time_utc"], return_none_on_error=True, use_zoneinfo=True)
                    if result:
                        return result
                # Если нет end_time_utc, используем session_id как fallback
                return session.get("session_id", "")

            previous_sessions.sort(key=get_sort_key)

            # Ограничиваем количество сессий
            if len(previous_sessions) > 10:
                previous_sessions = previous_sessions[:10]

            logger.debug(f"Загружено {len(previous_sessions)} предыдущих сессий")
            return previous_sessions

        except Exception as e:
            logger.warning(f"Ошибка при загрузке предыдущих сессий: {e}")
            return []

    def _empty_context(self) -> Dict[str, Any]:
        """Возвращает пустой контекст"""
        return {
            "previous_messages": [],
            "previous_messages_count": 0,
            "chat_context": None,
            "previous_sessions": [],
            "previous_sessions_count": 0,
            "context_window_hours": 0,
            "session_start_time": None,
            "context_start_time": None,
        }

    def format_context_for_prompt(
        self, extended_context: Dict[str, Any], max_context_length: int = 8192
    ) -> str:
        """
        Форматирует расширенный контекст для включения в промпт

        Args:
            extended_context: Расширенный контекст
            max_context_length: Максимальная длина контекста в символах

        Returns:
            Отформатированный контекст для промпта
        """
        context_parts = []

        # Добавляем глобальный контекст чата
        if extended_context.get("chat_context"):
            chat_context = extended_context["chat_context"]
            if len(chat_context) > 3000:  # Увеличиваем до 3000 символов
                chat_context = chat_context[:3000] + "..."
            context_parts.append(f"## Образ чата\n{chat_context}")

        # Добавляем предыдущие сессии
        if extended_context.get("previous_sessions"):
            sessions_text = "## Предыдущие сессии\n"
            for session in extended_context["previous_sessions"][
                -5:
            ]:  # Увеличиваем до последних 5 сессий
                sessions_text += f"- {session['session_id']}: {session['summary'][:300]}...\n"  # Увеличиваем до 300 символов
            context_parts.append(sessions_text)

        # Добавляем предыдущие сообщения
        if extended_context.get("previous_messages"):
            messages_text = "## Предыдущие сообщения\n"
            for msg in extended_context["previous_messages"][
                -15:
            ]:  # Увеличиваем до последних 15 сообщений
                messages_text += (
                    f"- {msg['text'][:150]}...\n"  # Увеличиваем до 150 символов
                )
            context_parts.append(messages_text)

        # Объединяем все части
        full_context = "\n\n".join(context_parts)

        # Обрезаем если слишком длинный
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."

        return full_context

    def should_use_extended_context(
        self, session_messages: List[Dict[str, Any]], min_messages_threshold: int = 10
    ) -> bool:
        """
        Определяет, нужно ли использовать расширенный контекст

        Args:
            session_messages: Сообщения текущей сессии
            min_messages_threshold: Минимальный порог сообщений для использования расширенного контекста

        Returns:
            True если нужно использовать расширенный контекст
        """
        return len(session_messages) < min_messages_threshold


def get_extended_context_for_session(
    chat_name: str,
    session_messages: List[Dict[str, Any]],
    context_window_hours: int = 24,
    max_previous_messages: int = 50,
) -> Dict[str, Any]:
    """
    Удобная функция для получения расширенного контекста

    Args:
        chat_name: Название чата
        session_messages: Сообщения текущей сессии
        context_window_hours: Окно времени для поиска предыдущих сообщений
        max_previous_messages: Максимальное количество предыдущих сообщений

    Returns:
        Расширенный контекст
    """
    context_manager = IncrementalContextManager()
    return context_manager.get_extended_context_for_session(
        chat_name, session_messages, context_window_hours, max_previous_messages
    )


if __name__ == "__main__":
    # Тест модуля
    context_manager = IncrementalContextManager()

    # Тестовые сообщения
    test_messages = [
        {
            "text": "Тестовое сообщение",
            "date_utc": "2025-10-06T18:00:00+00:00",
        }
    ]

    context = context_manager.get_extended_context_for_session(
        "Test Chat", test_messages
    )

    print("Расширенный контекст:")
    print(f"- Предыдущих сообщений: {context['previous_messages_count']}")
    print(f"- Предыдущих сессий: {context['previous_sessions_count']}")
    print(f"- Контекст чата: {'Да' if context['chat_context'] else 'Нет'}")
