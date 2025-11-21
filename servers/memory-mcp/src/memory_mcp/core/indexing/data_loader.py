"""Загрузка и подготовка данных для индексации."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...analysis.segmentation import DayGroupingSegmenter, SessionSegmenter
from ...analysis.utils import TimeProcessor
from ...utils.system.naming import slugify

logger = logging.getLogger(__name__)

MIN_SESSION_MESSAGES = 15


class DataLoader:
    """Загрузка сообщений из JSON, группировка по дням, объединение маленьких сессий."""

    def __init__(
        self,
        session_segmenter: SessionSegmenter,
        day_grouping_segmenter: DayGroupingSegmenter,
        time_processor: Optional[TimeProcessor] = None,
        enable_time_analysis: bool = False,
    ):
        """Инициализирует загрузчик данных.

        Args:
            session_segmenter: Сегментатор сессий
            day_grouping_segmenter: Сегментатор дневных групп
            time_processor: Процессор временных паттернов (опционально)
            enable_time_analysis: Включить анализ временных паттернов
        """
        self.session_segmenter = session_segmenter
        self.day_grouping_segmenter = day_grouping_segmenter
        self.time_processor = time_processor
        self.enable_time_analysis = enable_time_analysis

    async def load_messages_from_chat(self, chat_dir: Path) -> List[Dict[str, Any]]:
        """
        Загрузка сообщений из JSON файлов чата (использует общую утилиту).

        Args:
            chat_dir: Директория чата

        Returns:
            Список сообщений
        """
        from ...utils.data.json_loader import load_json_or_jsonl

        messages = []
        json_files = list(chat_dir.glob("*.json"))

        for json_file in json_files:
            try:
                file_messages, _ = load_json_or_jsonl(json_file)
                messages.extend(file_messages)
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {json_file}: {e}")
                continue

        # Сортируем по времени
        messages.sort(key=lambda x: x.get("date_utc") or x.get("date", ""))

        return messages

    def expand_day_groups(
        self, day_groups: List[Dict[str, Any]], chat_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Расширяет дневные группы в полноценные сессии с учётом разрывов."""
        if not day_groups:
            return []

        sessions: List[Dict[str, Any]] = []
        chat_slug = slugify(chat_name) if chat_name else ""

        for day_index, day_group in enumerate(day_groups):
            base_id = day_group.get("session_id")
            if not base_id:
                base_id = (
                    f"{chat_slug}-D{day_index + 1:04d}"
                    if chat_slug
                    else f"D{day_index + 1:04d}"
                )

            raw_messages = day_group.get("messages", [])
            splitted = self.session_segmenter.segment_messages(raw_messages, chat_name)
            merged_segments = (
                self.merge_small_sessions(
                    splitted,
                    chat_name=chat_name,
                    min_messages=MIN_SESSION_MESSAGES,
                )
                if splitted
                else []
            )

            segments_to_use = merged_segments or splitted

            if not segments_to_use:
                session = day_group.copy()
                session["session_id"] = base_id
                session["day_group_id"] = base_id
                session["parent_session_id"] = base_id
                session["group_type"] = day_group.get("group_type", "day_grouped")
                session["chat"] = chat_name
                if chat_slug:
                    session["chat_id"] = chat_slug
                session["messages"] = session.get("messages", raw_messages)
                session["message_count"] = len(session.get("messages", []))

                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(
                            session.get("messages", [])
                        )
                        session["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(
                            f"Ошибка анализа временных паттернов для сессии {base_id}: {e}"
                        )

                sessions.append(session)
                continue

            if len(segments_to_use) == 1:
                session_data = segments_to_use[0].copy()
                session_data["session_id"] = base_id
                session_data["day_group_id"] = base_id
                session_data["parent_session_id"] = base_id
                session_data["group_type"] = session_data.get("group_type") or (
                    "session_segmented"
                    if splitted
                    else day_group.get("group_type", "day_grouped")
                )
                session_data["chat"] = chat_name
                if chat_slug:
                    session_data["chat_id"] = chat_slug
                session_data["messages"] = session_data.get("messages", raw_messages)
                session_data["message_count"] = len(session_data["messages"])

                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(
                            session_data.get("messages", [])
                        )
                        session_data["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(
                            f"Ошибка анализа временных паттернов для сессии {base_id}: {e}"
                        )

                sessions.append(session_data)
                continue

            for split_index, split_session in enumerate(segments_to_use):
                session_copy = split_session.copy()
                session_copy["session_id"] = f"{base_id}-S{split_index + 1:02d}"
                session_copy["day_group_id"] = base_id
                session_copy["parent_session_id"] = base_id
                session_copy["group_type"] = session_copy.get(
                    "group_type", "session_segmented"
                )
                session_copy["chat"] = chat_name
                if chat_slug:
                    session_copy["chat_id"] = chat_slug
                session_copy["messages"] = split_session.get("messages", raw_messages)
                session_copy["message_count"] = len(session_copy["messages"])

                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(
                            session_copy.get("messages", [])
                        )
                        session_copy["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(
                            f"Ошибка анализа временных паттернов для сессии {session_copy['session_id']}: {e}"
                        )

                sessions.append(session_copy)

        return sessions

    def merge_small_sessions(
        self,
        segments: List[Dict[str, Any]],
        chat_name: Optional[str],
        min_messages: int,
    ) -> List[Dict[str, Any]]:
        """Объединяет маленькие сессии в более крупные."""
        if not segments:
            return []

        grouped: List[List[Dict[str, Any]]] = []
        buffer: List[Dict[str, Any]] = []

        def segment_len(segment: Dict[str, Any]) -> int:
            return segment.get("message_count") or len(segment.get("messages", []))

        for segment in segments:
            count = segment_len(segment)
            if not buffer:
                buffer.append(segment)
                continue

            buffer_count = sum(segment_len(item) for item in buffer)

            # Более агрессивное объединение маленьких сессий
            # Объединяем если:
            # 1. Текущая сессия меньше min_messages ИЛИ
            # 2. Буфер меньше min_messages ИЛИ
            # 3. Текущая сессия очень маленькая (≤3 сообщения) И буфер тоже маленький (≤10 сообщений)
            should_merge = (
                count < min_messages
                or buffer_count < min_messages
                or (count <= 3 and buffer_count <= 10)
            )

            if should_merge:
                buffer.append(segment)
            else:
                grouped.append(buffer)
                buffer = [segment]

        if buffer:
            grouped.append(buffer)

        normalized: List[Dict[str, Any]] = []
        for group in grouped:
            total_messages = sum(segment_len(item) for item in group)

            if len(group) == 1 and total_messages >= min_messages:
                segment_copy = group[0].copy()
                segment_copy["group_type"] = segment_copy.get(
                    "group_type", "session_segmented"
                )
                normalized.append(segment_copy)
                continue

            if len(group) > 1:
                segment_sizes = [segment_len(item) for item in group]
                logger.info(
                    f"🔗 Объединение {len(group)} маленьких сессий "
                    f"(размеры: {segment_sizes}) в одну сессию с {total_messages} сообщениями"
                )

            combined_messages: List[Dict[str, Any]] = []
            for segment in group:
                combined_messages.extend(segment.get("messages", []))

            combined_messages.sort(
                key=lambda msg: self.session_segmenter._parse_message_time(msg)
            )

            raw_session = {
                "messages": combined_messages,
                "start_time": self.session_segmenter._parse_message_time(
                    combined_messages[0]
                ),
                "end_time": self.session_segmenter._parse_message_time(
                    combined_messages[-1]
                ),
                "chat": chat_name,
            }
            merged_session = self.session_segmenter._finalize_session(
                raw_session,
                len(normalized),
            )
            merged_session["group_type"] = "session_merged"
            normalized.append(merged_session)

        return normalized

    def parse_message_time(self, msg: Dict[str, Any]) -> datetime:
        """Парсинг времени сообщения (использует общую утилиту)."""
        from ...utils.processing.datetime_utils import parse_message_time

        return parse_message_time(msg, use_zoneinfo=True)

    def parse_session_start_time(self, session: Dict[str, Any]) -> datetime:
        """
        Парсит время начала сессии для хронологической сортировки (использует общую утилиту).

        Args:
            session: Словарь с данными сессии

        Returns:
            datetime: Время начала сессии или минимальная дата, если не удалось распарсить
        """
        from ...utils.processing.datetime_utils import parse_datetime_utc

        # Пробуем разные поля для времени начала
        start_time = session.get("start_time")
        if start_time:
            if isinstance(start_time, str):
                result = parse_datetime_utc(start_time, use_zoneinfo=True)
                if result:
                    return result
            elif isinstance(start_time, datetime):
                return start_time

        # Если нет start_time, берем время первого сообщения
        messages = session.get("messages", [])
        if messages:
            first_message = messages[0]
            msg_date = first_message.get("date_utc")
            if msg_date:
                result = parse_datetime_utc(msg_date, use_zoneinfo=True)
                if result:
                    return result

        # Если ничего не найдено, возвращаем минимальную дату
        return datetime.min.replace(tzinfo=None)

