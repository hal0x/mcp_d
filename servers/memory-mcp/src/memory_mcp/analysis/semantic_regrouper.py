"""Семантическая перегруппировка сессий через LLM."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.lmstudio_client import LMStudioEmbeddingClient
else:
    from ..core.lmstudio_client import LMStudioEmbeddingClient

logger = logging.getLogger(__name__)


class SemanticRegrouper:
    """
    Класс для семантической перегруппировки временно сгруппированных сессий.
    
    Использует LLM для понимания контекста и перегруппировки сессий по смыслу,
    темам и связям между сообщениями.
    """

    def __init__(
        self,
        embedding_client: Optional[LMStudioEmbeddingClient] = None,
    ):
        """
        Инициализация семантического перегруппировщика.

        Args:
            embedding_client: Клиент для LLM (опционально)
        """
        self.embedding_client = embedding_client

    async def regroup_sessions(
        self,
        sessions: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Перегруппировывает сессии по смыслу с использованием LLM.

        Args:
            sessions: Список временно сгруппированных сессий
            chat_name: Название чата
            accumulative_context: Накопительный контекст для понимания тем

        Returns:
            Список перегруппированных сессий с обоснованием группировки
        """
        if not sessions:
            return []

        if len(sessions) == 1:
            # Одна сессия - нечего перегруппировывать
            return sessions

        logger.info(
            f"Семантическая перегруппировка {len(sessions)} сессий для чата {chat_name}"
        )

        # Формируем промпт для перегруппировки
        prompt = self._create_regroup_prompt(sessions, chat_name, accumulative_context)

        # Отправляем запрос в LLM
        if not self.embedding_client:
            raise ValueError("LLM клиент не доступен для семантической перегруппировки")

        async with self.embedding_client:
            response = await self.embedding_client.generate_summary(
                prompt=prompt,
                temperature=0.3,
                max_tokens=8192,
            )

        # Парсим ответ от LLM
        regrouped_sessions = self._parse_llm_response(response, sessions)

        logger.info(
            f"Перегруппировано {len(sessions)} сессий в {len(regrouped_sessions)} групп"
        )

        return regrouped_sessions

    def _create_regroup_prompt(
        self,
        sessions: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str],
    ) -> str:
        """Создает промпт для семантической перегруппировки."""
        context_part = ""
        if accumulative_context:
            context_part = f"\n\nНакопительный контекст чата:\n{accumulative_context}\n"

        # Формируем информацию о сессиях
        sessions_info = []
        for i, session in enumerate(sessions, 1):
            session_id = session.get("session_id", f"session_{i}")
            messages = session.get("messages", [])
            time_range = session.get("time_range", "unknown")
            window = session.get("window", "unknown")

            # Извлекаем первые несколько сообщений для понимания темы
            sample_messages = []
            for msg in messages[:3]:
                sender = msg.get("from", {})
                sender_name = (
                    sender.get("display", "Unknown")
                    if isinstance(sender, dict)
                    else "Unknown"
                )
                text = msg.get("text", "")[:200]
                if text:
                    sample_messages.append(f"  - {sender_name}: {text}")

            sessions_info.append(
                f"Сессия {i}:\n"
                f"  ID: {session_id}\n"
                f"  Окно: {window}\n"
                f"  Время: {time_range}\n"
                f"  Сообщений: {len(messages)}\n"
                f"  Примеры сообщений:\n" + "\n".join(sample_messages)
            )

        return f"""Проанализируй следующие сессии из чата "{chat_name}" и перегруппируй их по смыслу и темам.
{context_part}
Сессии для анализа:
{chr(10).join(sessions_info)}

Задачи:
1. Определи основные темы и контекст каждой сессии
2. Объедини связанные по смыслу сессии в новые группы
3. Обоснуй каждую группировку (почему эти сессии связаны)
4. Сохрани хронологический порядок внутри групп
5. Если сессии не связаны по смыслу, оставь их отдельными

Верни результат в формате JSON:
{{
  "groups": [
    {{
      "group_id": "group_1",
      "theme": "Основная тема группы",
      "rationale": "Обоснование группировки",
      "session_ids": ["session_1", "session_2"],
      "combined_summary": "Краткая саммаризация объединенной группы"
    }}
  ]
}}

Важно: каждый session_id должен быть включен ровно в одну группу."""

    def _parse_llm_response(
        self, response: str, original_sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Парсит ответ от LLM и создает перегруппированные сессии.

        Args:
            response: Ответ от LLM
            original_sessions: Оригинальные сессии

        Returns:
            Список перегруппированных сессий
        """
        try:
            # Пытаемся извлечь JSON из ответа
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.warning("Не найден JSON в ответе LLM, используем оригинальные сессии")
                return original_sessions

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Создаем словарь для быстрого доступа к сессиям по ID
            sessions_by_id = {
                session.get("session_id", f"session_{i}"): session
                for i, session in enumerate(original_sessions, 1)
            }

            regrouped_sessions = []
            groups = data.get("groups", [])

            for group in groups:
                session_ids = group.get("session_ids", [])
                theme = group.get("theme", "Unknown theme")
                rationale = group.get("rationale", "")
                combined_summary = group.get("combined_summary", "")

                # Объединяем сообщения из сессий группы
                combined_messages = []
                combined_time_ranges = []
                group_session_ids = []

                for session_id in session_ids:
                    if session_id in sessions_by_id:
                        session = sessions_by_id[session_id]
                        combined_messages.extend(session.get("messages", []))
                        time_range = session.get("time_range", "")
                        if time_range:
                            combined_time_ranges.append(time_range)
                        group_session_ids.append(session_id)

                if not combined_messages:
                    continue

                # Сортируем сообщения по времени
                combined_messages.sort(
                    key=lambda x: x.get("date_utc", "") or x.get("date", "")
                )

                # Создаем новую перегруппированную сессию
                regrouped_session = {
                    "session_id": f"regrouped_{group.get('group_id', 'group_1')}",
                    "original_session_ids": group_session_ids,
                    "chat": original_sessions[0].get("chat", "Unknown"),
                    "messages": combined_messages,
                    "message_count": len(combined_messages),
                    "time_range": " - ".join(combined_time_ranges) if combined_time_ranges else "unknown",
                    "window": original_sessions[0].get("window", "unknown"),
                    "theme": theme,
                    "regroup_rationale": rationale,
                    "combined_summary": combined_summary,
                    "regrouped": True,
                }

                # Добавляем метаданные из первой сессии
                first_session = sessions_by_id.get(session_ids[0] if session_ids else "")
                if first_session:
                    regrouped_session["start_time"] = first_session.get("start_time")
                    regrouped_session["end_time"] = (
                        sessions_by_id.get(session_ids[-1], {}).get("end_time")
                        if session_ids
                        else None
                    )

                regrouped_sessions.append(regrouped_session)

            # Проверяем, что все сессии включены
            included_ids = set()
            for session in regrouped_sessions:
                included_ids.update(session.get("original_session_ids", []))

            all_ids = set(sessions_by_id.keys())
            missing_ids = all_ids - included_ids

            # Добавляем пропущенные сессии как отдельные группы
            for session_id in missing_ids:
                logger.warning(f"Сессия {session_id} не включена в перегруппировку, добавляем отдельно")
                regrouped_sessions.append(sessions_by_id[session_id])

            return regrouped_sessions

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа от LLM: {e}")
            logger.debug(f"Ответ LLM: {response[:500]}")
            raise RuntimeError(
                f"Ошибка семантической перегруппировки: не удалось распарсить JSON ответ от LLM: {e}. "
                f"Ответ LLM: {response[:500] if len(response) > 500 else response}. "
                f"Проверьте конфигурацию LLM клиента."
            ) from e
        except Exception as e:
            logger.error(f"Ошибка при обработке ответа LLM: {e}")
            response_preview = response[:500] if 'response' in locals() and response else 'N/A'
            raise RuntimeError(
                f"Ошибка семантической перегруппировки: {e}. "
                f"Ответ LLM: {response_preview}. "
                f"Проверьте конфигурацию LLM клиента."
            ) from e

