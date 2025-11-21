"""Умная агрегация с временными окнами."""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ...memory.qdrant_collections import QdrantCollectionsManager
from ...utils.naming import slugify

logger = logging.getLogger(__name__)


class SmartAggregationManager:
    """Умная группировка с временными окнами."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        sessions_collection: Optional[str],
        messages_collection: Optional[str],
        graph: Optional[Any],
        strategy_threshold: int = 1000,
        progress_manager: Optional[Any] = None,
    ):
        """Инициализирует менеджер умной агрегации.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            sessions_collection: Имя коллекции сессий
            messages_collection: Имя коллекции сообщений
            graph: Граф памяти
            strategy_threshold: Порог количества сообщений для перехода между стратегиями
            progress_manager: Менеджер прогресса (для подсчета проиндексированных сообщений)
        """
        self.qdrant_manager = qdrant_manager
        self.sessions_collection = sessions_collection
        self.messages_collection = messages_collection
        self.graph = graph
        self.strategy_threshold = strategy_threshold
        self.progress_manager = progress_manager

    def group_messages_by_smart_strategy(
        self, messages: List[Dict[str, Any]], chat_name: str
    ) -> List[Dict[str, Any]]:
        """
        Группировка сообщений с умной стратегией окон

        NOW (0-1 день): группировка по дням или сессиям
        FRESH (1-14 дней): группировка по дням с минимумом сообщений
        RECENT (14-30 дней): группировка по неделям
        OLD (30+ дней): группировка по месяцам

        Переход между стратегиями:
        - Если уже проиндексировано >1000 сообщений в чате, переходим к следующей стратегии
        - fresh -> recent -> old
        """
        from datetime import datetime

        if not messages:
            return []

        current_date = datetime.now(datetime.now().astimezone().tzinfo)

        # Подсчитываем уже проиндексированные сообщения в чате
        indexed_messages_count = 0
        if self.progress_manager:
            indexed_messages_count = self.progress_manager.count_indexed_messages_in_chat(
                chat_name
            )

        # Определяем стратегию на основе количества уже проиндексированных сообщений
        if indexed_messages_count >= self.strategy_threshold:
            logger.info(
                f"🔄 Переход стратегии для чата {chat_name}: "
                f"уже проиндексировано {indexed_messages_count} сообщений "
                f"(порог: {self.strategy_threshold})"
            )

            # Определяем текущую стратегию на основе существующих сессий
            current_strategy = self.determine_current_strategy(chat_name)
            logger.info(f"📊 Текущая стратегия для чата {chat_name}: {current_strategy}")

            # Переходим к следующей стратегии
            next_strategy = self.get_next_strategy(current_strategy)
            logger.info(f"➡️  Переход к стратегии: {next_strategy}")

            # Применяем новую стратегию
            return self.apply_strategy_transition(
                messages, chat_name, next_strategy, current_date
            )

        # Определяем начальный номер сессии на основе существующих сессий
        existing_session_ids = set()
        try:
            existing_sessions = None
            if self.qdrant_manager and self.sessions_collection:
                existing_sessions = self.qdrant_manager.get(
                    collection_name=self.sessions_collection, where={"chat": chat_name}
                )
            if existing_sessions and existing_sessions.get("ids"):
                existing_session_ids = set(existing_sessions["ids"])
                logger.info(
                    f"Найдено {len(existing_session_ids)} существующих сессий для чата {chat_name}"
                )
        except Exception as e:
            logger.warning(f"Ошибка при получении существующих сессий: {e}")

        # Определяем максимальный номер сессии для каждого окна
        window_max_numbers = {}
        for session_id in existing_session_ids:
            if f"{slugify(chat_name)}-" in session_id:
                parts = session_id.split("-")
                if len(parts) >= 3:
                    window_name = parts[1]
                    try:
                        session_num = int(parts[2][1:])  # Убираем 'S' и берем число
                        if window_name not in window_max_numbers:
                            window_max_numbers[window_name] = 0
                        window_max_numbers[window_name] = max(
                            window_max_numbers[window_name], session_num
                        )
                    except (ValueError, IndexError):
                        continue

        logger.info(f"Максимальные номера сессий по окнам: {window_max_numbers}")

        # Применяем стратегию fresh по умолчанию для новых чатов
        return self.apply_strategy_transition(
            messages, chat_name, "fresh", current_date
        )

    def determine_current_strategy(self, chat_name: str) -> str:
        """
        Определяет текущую стратегию на основе существующих сессий

        Returns:
            str: текущая стратегия (fresh, recent, old)
        """
        try:
            existing_sessions = None
            if self.qdrant_manager and self.sessions_collection:
                existing_sessions = self.qdrant_manager.get(
                    collection_name=self.sessions_collection, where={"chat": chat_name}
                )

            if existing_sessions and existing_sessions.get("ids"):
                # Анализируем имена сессий для определения стратегии
                for session_id in existing_sessions["ids"]:
                    if "-fresh-" in session_id:
                        return "fresh"
                    elif "-recent-" in session_id:
                        return "recent"
                    elif "-old-" in session_id:
                        return "old"

            # По умолчанию возвращаем fresh
            return "fresh"
        except Exception as e:
            logger.warning(f"Ошибка при определении текущей стратегии: {e}")
            return "fresh"

    def get_next_strategy(self, current_strategy: str) -> str:
        """
        Получить следующую стратегию после текущей

        Args:
            current_strategy: Текущая стратегия

        Returns:
            str: следующая стратегия
        """
        strategy_sequence = ["fresh", "recent", "old"]
        try:
            current_index = strategy_sequence.index(current_strategy)
            if current_index < len(strategy_sequence) - 1:
                return strategy_sequence[current_index + 1]
            return current_strategy  # Уже на последней стратегии
        except ValueError:
            # Неизвестная стратегия, возвращаем fresh
            return "fresh"

    def apply_strategy_transition(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        strategy: str,
        current_date: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Применяет переход к новой стратегии группировки

        Args:
            messages: Список сообщений
            chat_name: Название чата
            strategy: Стратегия группировки (fresh, recent, old)
            current_date: Текущая дата

        Returns:
            Список сгруппированных сообщений
        """
        # Эта логика должна быть реализована в SmartRollingAggregator
        # Здесь возвращаем пустой список, так как реальная логика находится в другом месте
        logger.warning(
            f"apply_strategy_transition не реализован полностью, "
            f"используется стратегия {strategy} для чата {chat_name}"
        )
        return []

    def link_session_to_previous_sessions(
        self, session_id: str, chat: str, session_timestamp: datetime
    ) -> None:
        """Создает связи между текущей сессией и предыдущими сессиями того же чата."""
        if not self.graph:
            return

        try:
            from ...memory.graph_types import GraphEdge, EdgeType

            cursor = self.graph.conn.cursor()

            # Нормализуем имя чата для поиска (сессии могут иметь формат "semya-old-S0001")
            # Используем функцию slugify для получения нормализованного имени
            chat_slug = slugify(chat) if chat else ""

            # Ищем предыдущие сессии того же чата
            # Сессии могут иметь формат: "semya-old-S0001", "Семья-old-S0001", "semya-S0001" и т.д.
            # Ищем по паттерну, который включает нормализованное имя чата или оригинальное имя
            query = """
                SELECT id, properties FROM nodes
                WHERE type = 'DocChunk' 
                AND id != ?
                AND properties IS NOT NULL
                AND json_extract(properties, '$.session_type') = 'session_summary'
                AND (
                    json_extract(properties, '$.chat') = ?
                    OR json_extract(properties, '$.source') = ?
                )
                AND (
                    id LIKE ? 
                    OR id LIKE ?
                    OR (id LIKE ? AND ? != '')
                    OR id LIKE ?
                    OR id LIKE ?
                )
                ORDER BY json_extract(properties, '$.timestamp') DESC
                LIMIT 5
            """

            # Ищем сессии с различными форматами:
            # 1. С нормализованным именем: "semya-old-S%", "semya-S%"
            # 2. С оригинальным именем: "Семья-old-S%", "Семья-S%"
            # 3. С любым префиксом, если chat_slug пустой
            # 4. Regrouped groups: "regrouped_group_%", "regrouped_%"
            # 5. Day grouping: "%-D%"
            pattern1 = f"{chat_slug}-%-S%" if chat_slug else "%"
            pattern2 = f"{chat}-%-S%" if chat else "%"
            pattern3 = f"{chat_slug}-S%" if chat_slug else "%"
            pattern4 = "regrouped_%"  # Для regrouped groups
            pattern5 = f"{chat_slug}-%-D%" if chat_slug else "%"  # Для day grouping

            cursor.execute(
                query,
                (
                    session_id,
                    chat,
                    chat,
                    pattern1,
                    pattern2,
                    pattern3,
                    chat_slug,
                    pattern4,
                    pattern5,
                ),
            )
            existing_sessions = cursor.fetchall()

            for row in existing_sessions:
                try:
                    props = (
                        json.loads(row["properties"])
                        if isinstance(row["properties"], str)
                        else row["properties"]
                    )
                    if not props:
                        continue

                    # Получаем timestamp предыдущей сессии
                    prev_timestamp_str = props.get("timestamp") or props.get(
                        "start_time_utc"
                    )
                    if not prev_timestamp_str:
                        continue

                    from ...utils.datetime_utils import parse_datetime_utc

                    prev_timestamp = parse_datetime_utc(prev_timestamp_str, default=None)
                    if not prev_timestamp:
                        continue

                    # Создаем связь только если сессии близки по времени (в пределах 7 дней)
                    time_diff = abs(
                        (session_timestamp - prev_timestamp).total_seconds()
                    )
                    if time_diff <= 7 * 24 * 3600:  # 7 дней
                        prev_session_id = row["id"]

                        # Определяем направление связи (от более старой к более новой)
                        if session_timestamp > prev_timestamp:
                            source_id = prev_session_id
                            target_id = session_id
                        else:
                            source_id = session_id
                            target_id = prev_session_id

                        edge = GraphEdge(
                            id=f"{source_id}-next-session-{target_id}",
                            source_id=source_id,
                            target_id=target_id,
                            type=EdgeType.RELATES_TO,
                            weight=0.7,  # Высокий вес для связей между сессиями
                            properties={
                                "time_diff_seconds": time_diff,
                                "relation_type": "session_sequence",
                            },
                        )
                        try:
                            self.graph.add_edge(edge)
                            logger.debug(
                                f"Создана связь между сессиями {source_id} -> {target_id}"
                            )
                        except Exception as e:
                            # Игнорируем ошибки, если связь уже существует
                            logger.debug(
                                f"Не удалось создать связь между сессиями {source_id} и {target_id}: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Ошибка при создании связи с сессией {row['id']}: {e}"
                    )
                    continue
        except Exception as e:
            logger.debug(
                f"Ошибка при связывании сессии {session_id} с предыдущими: {e}"
            )

