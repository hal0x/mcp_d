#!/usr/bin/env python3
"""
Модуль для управления контекстом между сессиями
Обеспечивает передачу контекста от предыдущих сессий к новым
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.naming import slugify

logger = logging.getLogger(__name__)


class ContextManager:
    """Класс для управления контекстом между сессиями"""

    def __init__(
        self,
        summaries_dir: Path = Path("artifacts/reports"),
        max_context_size: int = 100000,  # Максимальный размер контекста в символах
        enable_cache: bool = True,
        cache_size: int = 100,  # Размер кэша для больших контекстов
    ):
        """
        Инициализация менеджера контекста

        Args:
            summaries_dir: Директория с саммаризациями
            max_context_size: Максимальный размер контекста в символах (для больших контекстов)
            enable_cache: Включить кэширование больших контекстов
            cache_size: Размер кэша для больших контекстов
        """
        self.artifacts_dir = summaries_dir
        self.artifacts_dir.mkdir(exist_ok=True)
        self.max_context_size = max_context_size
        self.enable_cache = enable_cache
        self._context_cache: Dict[str, Any] = {}  # Кэш для больших контекстов
        self._cache_size = cache_size
        
        # Промежуточный контекст в памяти во время индексации
        self._accumulative_contexts: Dict[str, str] = {}  # chat_name -> накопительный контекст
        self._session_summaries: Dict[str, List[Dict[str, Any]]] = {}  # chat_name -> список саммаризаций сессий

    def get_previous_context(
        self, chat_name: str, current_session_id: str, max_sessions: int = 10
    ) -> Dict[str, Any]:
        """
        Получение контекста из предыдущих сессий

        Args:
            chat_name: Название чата
            current_session_id: ID текущей сессии
            max_sessions: Максимальное количество предыдущих сессий для включения

        Returns:
            Словарь с контекстом предыдущих сессий
        """
        try:
            # Сначала пытаемся получить накопительный контекст чата
            chat_context = self._get_chat_context(chat_name)

            # Получаем список всех сессий для чата
            chat_sessions = self._get_chat_sessions(chat_name)

            if not chat_sessions:
                # Если нет сессий, но есть контекст чата, используем его
                if chat_context:
                    return {
                        "previous_sessions_count": 0,
                        "recent_context": chat_context,
                        "ongoing_decisions": [],
                        "open_risks": [],
                        "key_links": [],
                        "session_timeline": [],
                        "chat_context": chat_context,
                    }
                return self._empty_context()

            # Находим текущую сессию и предыдущие
            current_index = self._find_session_index(chat_sessions, current_session_id)

            if current_index is None:
                # При первой индексации сессия может еще не существовать - это нормально
                logger.debug(
                    f"Сессия {current_session_id} не найдена в чате {chat_name} (возможно, первая индексация)"
                )
                if chat_context:
                    return {
                        "previous_sessions_count": 0,
                        "recent_context": chat_context,
                        "ongoing_decisions": [],
                        "open_risks": [],
                        "key_links": [],
                        "session_timeline": [],
                        "chat_context": chat_context,
                    }
                return self._empty_context()

            # Берем предыдущие сессии (не включая текущую)
            previous_sessions = chat_sessions[:current_index]

            # Ограничиваем количество предыдущих сессий
            if len(previous_sessions) > max_sessions:
                previous_sessions = previous_sessions[-max_sessions:]

            # Проверяем кэш для больших контекстов
            cache_key = self._get_cache_key(chat_name, current_session_id, previous_sessions)
            if self.enable_cache and cache_key in self._context_cache:
                logger.debug(f"Использован кэш для контекста {cache_key}")
                return self._context_cache[cache_key]

            # Загружаем контекст из предыдущих сессий
            context = self._build_context_from_sessions(previous_sessions)
            
            # Оптимизируем размер контекста для больших контекстов
            context = self._optimize_context_size(context)
            
            # Сохраняем в кэш
            if self.enable_cache:
                self._add_to_cache(cache_key, context)

            # Добавляем накопительный контекст чата
            if chat_context:
                context["chat_context"] = chat_context
                # Объединяем с существующим контекстом
                if context["recent_context"]:
                    context[
                        "recent_context"
                    ] = f"{chat_context}\n\n{context['recent_context']}"
                else:
                    context["recent_context"] = chat_context

            logger.info(
                f"Загружен контекст из {len(previous_sessions)} предыдущих сессий для {current_session_id}"
            )
            return context

        except Exception as e:
            logger.error(
                f"Ошибка при получении контекста для {current_session_id}: {e}"
            )
            return self._empty_context()

    def _get_chat_sessions(self, chat_name: str) -> List[Dict[str, Any]]:
        """
        Получение списка всех сессий для чата

        Args:
            chat_name: Название чата

        Returns:
            Список сессий с метаданными
        """
        chat_dir = self.artifacts_dir / slugify(chat_name) / "sessions"

        if not chat_dir.exists():
            return []

        sessions = []

        # Сканируем файлы сессий
        for session_file in chat_dir.glob("*.json"):
            try:
                session_info = self._parse_session_file(session_file)
                if session_info:
                    sessions.append(session_info)
            except Exception as e:
                logger.warning(f"Ошибка при чтении файла сессии {session_file}: {e}")
                continue

        # Сортируем по session_id для правильного порядка сессий
        sessions.sort(key=lambda x: x.get("session_id", ""))

        return sessions

    def _parse_session_file(self, session_file: Path) -> Optional[Dict[str, Any]]:
        """
        Парсинг файла сессии для извлечения метаданных

        Args:
            session_file: Путь к файлу сессии

        Returns:
            Словарь с метаданными сессии или None
        """
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))

            session_id = data.get("session_id") or session_file.stem
            meta = data.get("meta", {})
            time_range = meta.get("time_span", "")
            topics = data.get("topics", [])
            context_summary = " ".join(
                topic.get("summary", "") for topic in topics[:2]
            ).strip()

            decisions = [
                f"- {action.get('text', '')}" for action in data.get("actions", [])
            ]
            risks = [f"- {risk.get('text', '')}" for risk in data.get("risks", [])]
            links = [
                attachment.split(":", 1)[1]
                for attachment in data.get("attachments", [])
                if attachment.startswith("link:")
            ]

            return {
                "session_id": session_id,
                "file_path": session_file,
                "file_time": session_file.stat().st_mtime,
                "time_range": time_range,
                "context": context_summary,
                "decisions": decisions,
                "risks": risks,
                "links": links,
            }

        except Exception as e:
            logger.error(f"Ошибка при парсинге файла сессии {session_file}: {e}")
            return None

    def _find_session_index(
        self, sessions: List[Dict[str, Any]], session_id: str
    ) -> Optional[int]:
        """Поиск индекса сессии в списке"""
        for i, session in enumerate(sessions):
            if session["session_id"] == session_id:
                return i
        return None

    def _build_context_from_sessions(
        self, sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Построение контекста из предыдущих сессий

        Args:
            sessions: Список предыдущих сессий

        Returns:
            Словарь с агрегированным контекстом
        """
        context = {
            "previous_sessions_count": len(sessions),
            "recent_context": "",
            "ongoing_decisions": [],
            "open_risks": [],
            "key_links": [],
            "session_timeline": [],
        }

        if not sessions:
            return context

        # Собираем информацию из последних сессий
        recent_contexts = []
        all_decisions = []
        all_risks = []
        all_links = []

        for session in sessions:
            # Добавляем в timeline
            context["session_timeline"].append(
                {
                    "session_id": session["session_id"],
                    "time_range": session["time_range"],
                    "context_summary": session["context"][:200] + "..."
                    if len(session["context"]) > 200
                    else session["context"],
                }
            )

            # Собираем контекст
            if session["context"]:
                recent_contexts.append(session["context"])

            # Собираем решения
            all_decisions.extend(session["decisions"])

            # Собираем риски
            all_risks.extend(session["risks"])

            # Собираем ссылки
            all_links.extend(session["links"])

        # Формируем агрегированный контекст
        if recent_contexts:
            context["recent_context"] = " ".join(
                recent_contexts[-5:]
            )  # Последние 5 контекстов

        # Берем последние решения и риски
        context["ongoing_decisions"] = all_decisions[-10:] if all_decisions else []
        context["open_risks"] = all_risks[-5:] if all_risks else []

        # Уникальные ссылки
        context["key_links"] = list(set(all_links))[-20:] if all_links else []

        return context

    def _get_chat_context(self, chat_name: str) -> Optional[str]:
        """
        Получает накопительный контекст чата из файла

        Args:
            chat_name: Название чата

        Returns:
            Контекст чата или None, если файл не найден
        """
        try:
            context_file = Path("artifacts/chat_contexts") / f"{chat_name}_context.md"
            if context_file.exists():
                with open(context_file, encoding="utf-8") as f:
                    content = f.read()
                    # Извлекаем только образ чата (между ## Образ чата и ## Последние сессии)
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
            return None
        except Exception as e:
            logger.warning(f"Не удалось загрузить контекст чата {chat_name}: {e}")
            return None

    def _get_cache_key(
        self, chat_name: str, session_id: str, sessions: List[Dict[str, Any]]
    ) -> str:
        """Генерация ключа кэша для контекста."""
        import hashlib

        session_ids = [s.get("session_id", "") for s in sessions]
        key_data = f"{chat_name}:{session_id}:{','.join(session_ids)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _optimize_context_size(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация размера контекста для больших контекстов.

        Обрезает большие текстовые поля до max_context_size.
        """
        optimized = context.copy()

        # Оптимизируем recent_context
        if "recent_context" in optimized and optimized["recent_context"]:
            if len(optimized["recent_context"]) > self.max_context_size:
                optimized["recent_context"] = (
                    optimized["recent_context"][: self.max_context_size] + "..."
                )

        # Оптимизируем chat_context
        if "chat_context" in optimized and optimized["chat_context"]:
            if len(optimized["chat_context"]) > self.max_context_size:
                optimized["chat_context"] = (
                    optimized["chat_context"][: self.max_context_size] + "..."
                )

        return optimized

    def _add_to_cache(self, key: str, context: Dict[str, Any]) -> None:
        """Добавление контекста в кэш с ограничением размера."""
        # Очищаем кэш, если он превышает размер
        if len(self._context_cache) >= self._cache_size:
            # Удаляем самый старый элемент (FIFO)
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]

        self._context_cache[key] = context

    def clear_cache(self) -> None:
        """Очистка кэша контекста."""
        self._context_cache.clear()

    def get_accumulative_context(self, chat_name: str) -> str:
        """
        Возвращает накопительный контекст для использования в группировке.

        Args:
            chat_name: Название чата

        Returns:
            Накопительный контекст в виде строки
        """
        # Сначала проверяем промежуточный контекст в памяти
        if chat_name in self._accumulative_contexts:
            return self._accumulative_contexts[chat_name]

        # Если нет в памяти, загружаем из файла
        chat_context = self._get_chat_context(chat_name)
        if chat_context:
            # Сохраняем в памяти для быстрого доступа
            self._accumulative_contexts[chat_name] = chat_context
            return chat_context

        return ""

    def update_context_after_session(
        self,
        chat_name: str,
        session: Dict[str, Any],
        session_summary: Optional[str] = None,
    ) -> None:
        """
        Обновляет накопительный контекст после обработки сессии/дня.

        Args:
            chat_name: Название чата
            session: Обработанная сессия
            session_summary: Саммаризация сессии (опционально)
        """
        # Инициализируем структуры для чата, если их еще нет
        if chat_name not in self._accumulative_contexts:
            # Загружаем существующий контекст из файла
            existing_context = self._get_chat_context(chat_name)
            self._accumulative_contexts[chat_name] = existing_context or ""

        if chat_name not in self._session_summaries:
            self._session_summaries[chat_name] = []

        # Добавляем саммаризацию сессии
        session_id = session.get("session_id", "unknown")
        time_range = session.get("time_range", "unknown")
        window = session.get("window", "unknown")
        message_count = session.get("message_count", len(session.get("messages", [])))

        session_info = {
            "session_id": session_id,
            "time_range": time_range,
            "window": window,
            "message_count": message_count,
            "summary": session_summary or session.get("summary", ""),
        }

        self._session_summaries[chat_name].append(session_info)

        # Обновляем накопительный контекст
        # Берем последние N саммаризаций для контекста
        recent_summaries = self._session_summaries[chat_name][-20:]  # Последние 20 сессий

        # Формируем обновленный контекст
        context_parts = []

        # Добавляем существующий контекст, если он есть
        existing_context = self._accumulative_contexts[chat_name]
        if existing_context:
            context_parts.append(f"Предыдущий контекст:\n{existing_context}\n")

        # Добавляем саммаризации последних сессий
        context_parts.append("Последние сессии:\n")
        for sess_info in recent_summaries:
            summary = sess_info.get("summary", "")
            if summary:
                context_parts.append(
                    f"- {sess_info['session_id']} ({sess_info['time_range']}, "
                    f"окно: {sess_info['window']}): {summary[:300]}"
                )

        # Объединяем и ограничиваем размер
        new_context = "\n".join(context_parts)
        if len(new_context) > self.max_context_size:
            # Обрезаем до максимального размера, оставляя последние части
            new_context = new_context[-self.max_context_size:]

        self._accumulative_contexts[chat_name] = new_context

        logger.debug(
            f"Обновлен накопительный контекст для {chat_name}: "
            f"{len(self._session_summaries[chat_name])} сессий, "
            f"{len(new_context)} символов"
        )

    def flush_accumulative_context(self, chat_name: str) -> None:
        """
        Сохраняет накопительный контекст в файл и очищает память.

        Args:
            chat_name: Название чата
        """
        if chat_name not in self._accumulative_contexts:
            return

        context = self._accumulative_contexts[chat_name]
        if not context:
            return

        # Сохраняем в файл через существующий метод
        context_dir = Path("artifacts/chat_contexts")
        context_dir.mkdir(exist_ok=True)
        context_file = context_dir / f"{chat_name}_context.md"

        try:
            with open(context_file, "w", encoding="utf-8") as f:
                f.write(context)
            logger.info(f"Накопительный контекст сохранен в {context_file}")

            # Очищаем память (но оставляем в кэше)
            del self._accumulative_contexts[chat_name]
            if chat_name in self._session_summaries:
                del self._session_summaries[chat_name]

        except Exception as e:
            logger.error(f"Ошибка при сохранении накопительного контекста: {e}")

    def clear_accumulative_context(self, chat_name: str) -> None:
        """
        Очищает накопительный контекст в памяти (не сохраняя в файл).

        Args:
            chat_name: Название чата
        """
        if chat_name in self._accumulative_contexts:
            del self._accumulative_contexts[chat_name]
        if chat_name in self._session_summaries:
            del self._session_summaries[chat_name]

    def _empty_context(self) -> Dict[str, Any]:
        """Возвращает пустой контекст"""
        return {
            "previous_sessions_count": 0,
            "recent_context": "",
            "ongoing_decisions": [],
            "open_risks": [],
            "key_links": [],
            "session_timeline": [],
        }


def get_session_context(
    chat_name: str,
    current_session_id: str,
    summaries_dir: Path = Path("artifacts/reports"),
    max_sessions: int = 3,
) -> Dict[str, Any]:
    """
    Удобная функция для получения контекста сессии

    Args:
        chat_name: Название чата
        current_session_id: ID текущей сессии
        summaries_dir: Директория с саммаризациями
        max_sessions: Максимальное количество предыдущих сессий

    Returns:
        Словарь с контекстом предыдущих сессий
    """
    context_manager = ContextManager(summaries_dir)
    return context_manager.get_previous_context(
        chat_name, current_session_id, max_sessions
    )


if __name__ == "__main__":
    # Тест модуля
    context_manager = ContextManager()

    # Тестируем на чате Newcoin
    context = context_manager.get_previous_context(
        "Newcoin - Возможности крипты", "Newcoin - Возможности крипты-D0002"
    )

    print("Контекст для D0002:")
    print(f"  Предыдущих сессий: {context['previous_sessions_count']}")
    print(f"  Недавний контекст: {context['recent_context'][:200]}...")
    print(f"  Текущие решения: {len(context['ongoing_decisions'])}")
    print(f"  Открытые риски: {len(context['open_risks'])}")
    print(f"  Ключевые ссылки: {len(context['key_links'])}")
