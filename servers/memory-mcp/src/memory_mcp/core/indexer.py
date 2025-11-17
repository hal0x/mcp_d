#!/usr/bin/env python3
"""
Модуль для двухуровневой индексации
L1 - sessions (саммари + E1)
L2 - messages (текст + симметричный контекст)
"""

import asyncio
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import chromadb

from ..analysis.cluster_summarizer import ClusterSummarizer
from ..analysis.day_grouping import DayGroupingSegmenter
from ..analysis.entity_extraction import EntityExtractor
from ..analysis.entity_dictionary import get_entity_dictionary
from ..analysis.instruction_manager import InstructionManager
from ..analysis.markdown_renderer import MarkdownRenderer
from ..analysis.session_clustering import SessionClusterer
from ..analysis.session_segmentation import SessionSegmenter
from ..analysis.session_summarizer import SessionSummarizer
from ..analysis.smart_rolling_aggregator import SmartRollingAggregator
from ..analysis.time_processor import TimeProcessor
from ..utils.naming import slugify
from ..utils.url_validator import validate_embedding_text
from .lmstudio_client import LMStudioEmbeddingClient

logger = logging.getLogger(__name__)

MIN_SESSION_MESSAGES = (
    15  # Уменьшено с 30 до 15 для более гибкого объединения маленьких сессий
)


class TwoLevelIndexer:
    """Класс для двухуровневой индексации (L1: sessions, L2: messages)"""

    def __init__(
        self,
        chroma_path: str = "./artifacts/chroma_db",
        artifacts_path: str = "./artifacts",
        embedding_client: Optional[LMStudioEmbeddingClient] = None,
        enable_quality_check: bool = True,
        enable_iterative_refinement: bool = True,
        min_quality_score: float = 80.0,
        enable_clustering: bool = True,
        clustering_threshold: float = 0.8,
        min_cluster_size: int = 2,
        max_messages_per_group: int = 100,
        max_session_hours: int = 6,
        gap_minutes: int = 60,
        enable_smart_aggregation: bool = False,
        aggregation_strategy: str = "smart",
        now_window_hours: int = 24,
        fresh_window_days: int = 14,
        recent_window_days: int = 30,
        strategy_threshold: int = 1000,
        force: bool = False,
        enable_entity_learning: bool = True,
        enable_time_analysis: bool = True,
        graph: Optional[Any] = None,  # TypedGraphMemory
    ):
        """
        Инициализация индексатора

        Args:
            chroma_path: Путь к ChromaDB
            artifacts_path: Каталог с артефактами (reports, контексты, коллекции)
            embedding_client: Клиент для генерации эмбеддингов (LM Studio)
            enable_quality_check: Включить проверку качества саммаризации
            enable_iterative_refinement: Включить автоматическое улучшение
            min_quality_score: Минимальный приемлемый балл качества
            enable_clustering: Включить автоматическую кластеризацию сессий
            clustering_threshold: Порог сходства для кластеризации
            min_cluster_size: Минимальный размер кластера
            max_messages_per_group: Максимальное количество сообщений в группе
            max_session_hours: Максимальная длительность сессии в часах
            gap_minutes: Максимальный разрыв между сообщениями в минутах
            enable_smart_aggregation: Включить умную группировку с скользящими окнами
            aggregation_strategy: Стратегия группировки (smart/channel/legacy)
            now_window_hours: Размер NOW окна в часах
            fresh_window_days: Размер FRESH окна в днях
            recent_window_days: Размер RECENT окна в днях
            strategy_threshold: Порог количества сообщений для перехода между стратегиями
            force: Принудительно пересоздать существующие артефакты
            enable_entity_learning: Включить автоматическое обучение словарей сущностей
            enable_time_analysis: Включить анализ временных паттернов
            graph: Граф памяти для синхронизации записей (опционально)
        """
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.artifacts_path = Path(artifacts_path).expanduser()
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.reports_path = self.artifacts_path / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client or LMStudioEmbeddingClient()

        # Флаги и параметры кластеризации
        self.enable_clustering = enable_clustering
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size

        # Параметры группировки
        self.max_messages_per_group = max_messages_per_group
        self.max_session_hours = max_session_hours
        self.gap_minutes = gap_minutes

        # Параметры умной группировки
        self.enable_smart_aggregation = enable_smart_aggregation
        self.aggregation_strategy = aggregation_strategy
        self.now_window_hours = now_window_hours
        self.fresh_window_days = fresh_window_days
        self.recent_window_days = recent_window_days
        self.strategy_threshold = strategy_threshold
        self.force = force

        # Новые параметры для расширенного анализа
        self.enable_entity_learning = enable_entity_learning
        self.enable_time_analysis = enable_time_analysis

        self.session_segmenter = SessionSegmenter(
            gap_minutes=gap_minutes,
            max_session_hours=max_session_hours,
            enable_time_analysis=enable_time_analysis,
        )
        self.day_grouping_segmenter = DayGroupingSegmenter(
            max_messages_per_group=max_messages_per_group,
        )

        # Создаём менеджер инструкций и SessionSummarizer с системой оценки качества
        self.instruction_manager = InstructionManager()
        self.session_summarizer = SessionSummarizer(
            self.embedding_client,
            self.reports_path,
            instruction_manager=self.instruction_manager,
            enable_quality_check=enable_quality_check,
            enable_iterative_refinement=enable_iterative_refinement,
            min_quality_score=min_quality_score,
        )

        self.entity_extractor = EntityExtractor(
            enable_learning=enable_entity_learning,
            enable_natasha=True,
        )
        
        # Инициализация TimeProcessor для анализа временных паттернов
        self.time_processor = TimeProcessor() if enable_time_analysis else None
        
        # Инициализация словаря сущностей для автоматического обучения
        self.entity_dictionary = get_entity_dictionary() if enable_entity_learning else None
        
        self.markdown_renderer = MarkdownRenderer(self.reports_path)

        # Кластеризация (инициализируется после коллекций)
        self.session_clusterer = None
        self.cluster_summarizer = None

        # Коллекции
        self.sessions_collection = None
        self.messages_collection = None
        self.tasks_collection = None
        self.clusters_collection = None
        self.progress_collection = None  # Новая коллекция для отслеживания прогресса

        self._initialize_collections()

        # Инициализируем кластеризацию после коллекций
        if self.enable_clustering:
            self.session_clusterer = SessionClusterer(
                similarity_threshold=self.clustering_threshold,
                min_cluster_size=self.min_cluster_size,
                use_hdbscan=False,  # Используем threshold-based для детерминизма
            )
            self.cluster_summarizer = ClusterSummarizer(
                embedding_client=self.embedding_client
            )

        # Инициализируем умный агрегатор
        if self.enable_smart_aggregation:
            self.smart_aggregator = SmartRollingAggregator(
                chats_dir=Path("chats"),
                use_smart_strategy=(self.aggregation_strategy == "smart"),
            )
        else:
            self.smart_aggregator = None

        # Инициализация графа памяти для синхронизации
        self.graph = graph
        if self.graph:
            from ..memory.ingest import MemoryIngestor
            self.ingestor = MemoryIngestor(self.graph)
            logger.info("TwoLevelIndexer: граф памяти подключен, записи будут синхронизироваться")
        else:
            self.ingestor = None
            logger.debug("TwoLevelIndexer: граф памяти не подключен, записи будут только в ChromaDB")

    def _get_embedding_dimension(self) -> Optional[int]:
        """Получить размерность эмбеддингов из клиента."""
        if not self.embedding_client:
            return None
        
        # Пытаемся получить размерность из клиента
        if hasattr(self.embedding_client, '_embedding_dimension') and self.embedding_client._embedding_dimension:
            return self.embedding_client._embedding_dimension
        
        # Если размерность ещё не определена, делаем тестовый запрос
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если цикл уже запущен, создаём новый
                import nest_asyncio
                try:
                    nest_asyncio.apply()
                except ImportError:
                    pass
            
            async def _get_dim():
                async with self.embedding_client:
                    test_embedding = await self.embedding_client.get_embedding("test")
                    return len(test_embedding) if test_embedding else None
            
            try:
                dimension = asyncio.run(_get_dim())
                if dimension:
                    self.embedding_client._embedding_dimension = dimension
                return dimension
            except RuntimeError:
                # Если цикл уже запущен, возвращаем None - размерность определится позже
                return None
        except Exception as e:
            logger.debug(f"Не удалось определить размерность эмбеддингов: {e}")
            return None

    def _check_and_recreate_collection(self, collection_name: str, description: str, force_recreate: bool = False):
        """Проверить коллекцию и пересоздать при несоответствии размерности."""
        expected_dimension = self._get_embedding_dimension()
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Проверяем размерность коллекции
            if expected_dimension:
                try:
                    # Получаем информацию о коллекции
                    collection_info = collection.get()
                    # Проверяем размерность по первому эмбеддингу, если есть
                    if collection_info.get("embeddings") and len(collection_info["embeddings"]) > 0:
                        existing_dimension = len(collection_info["embeddings"][0])
                        if existing_dimension != expected_dimension:
                            logger.warning(
                                f"Коллекция {collection_name} имеет размерность {existing_dimension}, "
                                f"ожидается {expected_dimension}. Пересоздаём коллекцию..."
                            )
                            self.chroma_client.delete_collection(collection_name)
                            collection = None
                        else:
                            logger.info(
                                f"Найдена коллекция {collection_name} с {collection.count()} записями "
                                f"(размерность: {existing_dimension})"
                            )
                    else:
                        # Если коллекция пустая, оставляем как есть
                        logger.info(
                            f"Найдена пустая коллекция {collection_name}"
                        )
                except Exception as e:
                    logger.debug(f"Не удалось проверить размерность коллекции {collection_name}: {e}")
                    # Если не удалось проверить, оставляем коллекцию как есть
                    logger.info(
                        f"Найдена коллекция {collection_name} с {collection.count()} записями"
                    )
            else:
                logger.info(
                    f"Найдена коллекция {collection_name} с {collection.count()} записями"
                )
            
            if collection is None or force_recreate:
                # Пересоздаём коллекцию
                if collection:
                    self.chroma_client.delete_collection(collection_name)
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": description},
                )
                logger.info(f"Создана новая коллекция {collection_name}")
            
            return collection
            
        except Exception:
            # Коллекция не существует, создаём новую
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": description},
            )
            logger.info(f"Создана новая коллекция {collection_name}")
            return collection

    def _initialize_collections(self):
        """Инициализация коллекций ChromaDB с проверкой размерности эмбеддингов"""
        try:
            # L1: Sessions
            self.sessions_collection = self._check_and_recreate_collection(
                "chat_sessions",
                "Саммаризации сессий для векторного поиска (L1)",
                force_recreate=self.force
            )

            # L2: Messages
            self.messages_collection = self._check_and_recreate_collection(
                "chat_messages",
                "Сообщения с контекстом для уточняющего поиска (L2)",
                force_recreate=self.force
            )

            # Tasks
            self.tasks_collection = self._check_and_recreate_collection(
                "chat_tasks",
                "Action Items из сессий",
                force_recreate=self.force
            )

            # Clusters
            self.clusters_collection = self._check_and_recreate_collection(
                "session_clusters",
                "Тематические кластеры сессий",
                force_recreate=self.force
            )

            # Progress tracking (не требует эмбеддингов, поэтому без проверки размерности)
            try:
                self.progress_collection = self.chroma_client.get_collection(
                    "indexing_progress"
                )
                logger.info(
                    f"Найдена коллекция indexing_progress с {self.progress_collection.count()} записями"
                )
            except Exception:
                self.progress_collection = self.chroma_client.create_collection(
                    name="indexing_progress",
                    metadata={
                        "description": "Отслеживание прогресса индексации для инкрементальных обновлений"
                    },
                )
                logger.info("Создана новая коллекция indexing_progress")

        except Exception as e:
            logger.error(f"Ошибка при инициализации коллекций: {e}")
            raise

    def _expand_day_groups(
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
                self._merge_small_sessions(
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
                
                # Добавляем анализ временных паттернов если включено
                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(session.get("messages", []))
                        session["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(f"Ошибка анализа временных паттернов для сессии {base_id}: {e}")
                
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
                
                # Добавляем анализ временных паттернов если включено
                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(session_data.get("messages", []))
                        session_data["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(f"Ошибка анализа временных паттернов для сессии {base_id}: {e}")
                
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
                
                # Добавляем анализ временных паттернов если включено
                if self.time_processor and self.enable_time_analysis:
                    try:
                        activity_patterns = self.time_processor.analyze_activity_patterns(session_copy.get("messages", []))
                        session_copy["activity_patterns"] = activity_patterns
                    except Exception as e:
                        logger.warning(f"Ошибка анализа временных паттернов для сессии {session_copy['session_id']}: {e}")
                
                sessions.append(session_copy)

        return sessions

    def _merge_small_sessions(
        self,
        segments: List[Dict[str, Any]],
        chat_name: Optional[str],
        min_messages: int,
    ) -> List[Dict[str, Any]]:
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

            # Логируем объединение сессий
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

    def _get_last_indexed_date(self, chat_name: str) -> Optional[datetime]:
        """
        Получить дату последнего проиндексированного сообщения для чата

        Args:
            chat_name: Название чата

        Returns:
            Дата последнего проиндексированного сообщения или None
        """
        try:
            result = self.progress_collection.get(
                ids=[f"progress_{slugify(chat_name)}"], include=["metadatas"]
            )

            if result["ids"] and result["metadatas"]:
                last_date_str = result["metadatas"][0].get("last_indexed_date")
                if last_date_str:
                    from ..utils.datetime_utils import parse_datetime_utc

                    return parse_datetime_utc(last_date_str, use_zoneinfo=True)
        except Exception as e:
            logger.debug(f"Не удалось получить прогресс для {chat_name}: {e}")

        return None

    def _save_indexing_progress(
        self,
        chat_name: str,
        last_message_date: datetime,
        messages_count: int,
        sessions_count: int,
    ):
        """
        Сохранить прогресс индексации для чата

        Args:
            chat_name: Название чата
            last_message_date: Дата последнего проиндексированного сообщения
            messages_count: Количество проиндексированных сообщений
            sessions_count: Количество созданных сессий
        """
        try:
            progress_id = f"progress_{slugify(chat_name)}"
            now = datetime.now(ZoneInfo("UTC"))

            # Форматируем даты в ISO формат
            last_date_iso = last_message_date.isoformat()
            now_iso = now.isoformat()

            metadata = {
                "chat_name": chat_name,
                "last_indexed_date": last_date_iso,
                "last_indexing_time": now_iso,
                "total_messages": messages_count,
                "total_sessions": sessions_count,
            }

            # Используем пустой эмбеддинг (не нужен для метаданных)
            dummy_embedding = [0.0] * 1024  # BGE-M3 размерность

            self.progress_collection.upsert(
                ids=[progress_id],
                documents=[f"Progress for {chat_name}"],
                embeddings=[dummy_embedding],
                metadatas=[metadata],
            )

            logger.info(
                f"Сохранён прогресс для {chat_name}: "
                f"последнее сообщение {last_date_iso}, "
                f"всего сообщений {messages_count}, сессий {sessions_count}"
            )
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогресса для {chat_name}: {e}")

    def _parse_session_start_time(self, session: Dict[str, Any]) -> datetime:
        """
        Парсит время начала сессии для хронологической сортировки (использует общую утилиту).

        Args:
            session: Словарь с данными сессии

        Returns:
            datetime: Время начала сессии или минимальная дата, если не удалось распарсить
        """
        from ..utils.datetime_utils import parse_datetime_utc

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

    async def build_index(
        self,
        scope: str = "all",
        chat: Optional[str] = None,
        force_full: bool = False,
        recent_days: int = 7,
        adapter: Optional[Any] = None,  # MemoryServiceAdapter, но избегаем циклического импорта
    ) -> Dict[str, Any]:
        """
        Построение индекса

        Args:
            scope: "all" или "chat"
            chat: Название чата (для scope="chat")
            force_full: Полная пересборка
            recent_days: Пересаммаризировать последние N дней

        Returns:
            Статистика индексации
        """
        logger.info(
            f"Начало индексации: scope={scope}, chat={chat}, force_full={force_full}"
        )

        stats = {
            "indexed_chats": [],
            "sessions_indexed": 0,
            "messages_indexed": 0,
            "tasks_indexed": 0,
        }

        # Получаем список чатов для индексации
        chats_path = Path("chats")
        if scope == "chat" and chat:
            chat_dirs = [chats_path / chat]
        else:
            chat_dirs = [d for d in chats_path.iterdir() if d.is_dir()]

        # Индексируем каждый чат
        total_chats = len(chat_dirs)
        for chat_idx, chat_dir in enumerate(chat_dirs, 1):
            try:
                chat_name = chat_dir.name
                logger.info(f"📁 Чат {chat_idx}/{total_chats}: {chat_name}")

                # Очистка старых данных при полной переиндексации
                if force_full and adapter is not None:
                    logger.info(f"🧹 Очистка старых данных чата {chat_name} перед переиндексацией...")
                    try:
                        cleanup_stats = adapter.clear_chat_data(chat_name)
                        logger.info(
                            f"✅ Очистка завершена: "
                            f"узлов={cleanup_stats.get('nodes_deleted', 0)}, "
                            f"векторов={cleanup_stats.get('vectors_deleted', 0)}, "
                            f"ChromaDB={cleanup_stats.get('chromadb_deleted', 0)}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Ошибка при очистке данных чата {chat_name}: {e}. "
                            f"Продолжаем индексацию...",
                            exc_info=True,
                        )

                # Загружаем сообщения из JSON файлов
                messages = await self._load_messages_from_chat(chat_dir)

                if not messages:
                    logger.warning(f"Нет сообщений в чате {chat_name}")
                    continue

                # Определяем, какие сообщения нужно переиндексировать
                if not force_full:
                    # Инкрементальная индексация: индексируем только новые сообщения
                    last_indexed_date = self._get_last_indexed_date(chat_name)

                    if last_indexed_date:
                        # Индексируем только сообщения новее последнего проиндексированного
                        messages_to_index = [
                            m
                            for m in messages
                            if self._parse_message_time(m) > last_indexed_date
                        ]
                        logger.info(
                            f"📊 Инкрементальная индексация: последнее проиндексированное "
                            f"сообщение от {last_indexed_date.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    else:
                        # Первая индексация чата - индексируем все или последние N дней
                        if recent_days > 0:
                            recent_cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(
                                days=recent_days
                            )
                            messages_to_index = [
                                m
                                for m in messages
                                if self._parse_message_time(m) >= recent_cutoff
                            ]
                            logger.info(
                                f"📊 Первая индексация: обрабатываем последние {recent_days} дней"
                            )
                        else:
                            messages_to_index = messages
                            logger.info(
                                "📊 Первая индексация: обрабатываем все сообщения"
                            )
                else:
                    messages_to_index = messages
                    logger.info("📊 Полная переиндексация: обрабатываем все сообщения")

                logger.info(
                    f"Сообщений для индексации: {len(messages_to_index)} из {len(messages)}"
                )

                # Выбираем стратегию группировки
                if self.enable_smart_aggregation and self.smart_aggregator:
                    logger.info("🧠 Используем умную группировку с скользящими окнами")
                    # Используем умную агрегацию
                    aggregation_result = await self.smart_aggregator.aggregate_chat(
                        chat_name, dry_run=False
                    )
                    sessions = aggregation_result.get("sessions", [])
                    logger.info(f"Умная агрегация создала {len(sessions)} сессий")
                else:
                    logger.info("📊 Используем классическую группировку")
                    # Классическая группировка с умной стратегией окон
                    sessions = self._group_messages_by_smart_strategy(
                        messages_to_index, chat_name
                    )
                    logger.info(f"Создано {len(sessions)} сессий с умной стратегией")

                # Получаем список уже проиндексированных сессий
                existing_session_ids = set()
                existing_summaries = []
                if not force_full:
                    try:
                        # Получаем все session_id для данного чата
                        result = self.sessions_collection.get(
                            where={"chat": chat_name}, include=["metadatas"]
                        )
                        if result and result.get("ids"):
                            existing_session_ids = set(result["ids"])
                            logger.info(
                                f"📋 Найдено {len(existing_session_ids)} уже проиндексированных сессий"
                            )

                            # Загружаем существующие саммаризации из JSON файлов
                            reports_dir = self.reports_path
                            chat_slug = slugify(chat_name)
                            sessions_dir = reports_dir / chat_slug / "sessions"

                            if sessions_dir.exists():
                                import json

                                for session_id in existing_session_ids:
                                    json_file = sessions_dir / f"{session_id}.json"
                                    if json_file.exists():
                                        try:
                                            with open(json_file, encoding="utf-8") as f:
                                                summary = json.load(f)
                                                existing_summaries.append(summary)
                                        except Exception as e:
                                            logger.debug(
                                                f"Не удалось загрузить {json_file}: {e}"
                                            )
                    except Exception as e:
                        logger.debug(f"Не удалось получить существующие сессии: {e}")

                # Сортируем сессии сначала по типу окна, затем по времени начала
                # Порядок: old -> recent -> fresh -> now для правильного формирования контекста
                window_priority = {"old": 0, "recent": 1, "fresh": 2, "now": 3}

                def sort_key(session):
                    window = session.get("window", "unknown")
                    priority = window_priority.get(
                        window, 999
                    )  # Неизвестные окна в конец
                    start_time = self._parse_session_start_time(session)
                    return (priority, start_time)

                sessions_sorted = sorted(sessions, key=sort_key)
                logger.info(
                    "📅 Сессии отсортированы по типу окна и времени (old -> recent -> fresh -> now)"
                )

                # Саммаризируем и индексируем каждую сессию в хронологическом порядке
                processed_summaries = []
                total_messages_in_chat = len(messages)
                processed_messages_count = 0
                skipped_sessions = 0

                for session_idx, session in enumerate(sessions_sorted, 1):
                    try:
                        if session is None:
                            logger.warning(
                                f"Пропускаем None сессию в позиции {session_idx}"
                            )
                            continue

                        session_id = session.get("session_id")
                        session_messages_count = len(session.get("messages", []))

                        # Проверяем, не была ли эта сессия уже проиндексирована
                        if session_id in existing_session_ids:
                            logger.debug(
                                f"⏭️  Пропускаем уже проиндексированную сессию: {session_id}"
                            )
                            skipped_sessions += 1
                            processed_messages_count += session_messages_count
                            continue

                        processed_messages_count += session_messages_count

                        # Показываем прогресс
                        progress_pct = (
                            processed_messages_count / total_messages_in_chat
                        ) * 100
                        logger.info(
                            f"📊 Прогресс чата {chat_name}: "
                            f"{processed_messages_count}/{total_messages_in_chat} сообщений "
                            f"({progress_pct:.1f}%) | "
                            f"Сессия {session_idx}/{len(sessions)}: {session_id}"
                        )

                        # Саммаризация
                        summary = await self.session_summarizer.summarize_session(
                            session
                        )
                        processed_summaries.append(summary)

                        # L1: Индексация саммари сессии
                        await self._index_session_l1(summary)
                        stats["sessions_indexed"] += 1

                        # L2: Индексация сообщений с контекстом
                        messages_count = await self._index_messages_l2(session)
                        stats["messages_indexed"] += messages_count

                        # Индексация tasks
                        tasks_count = await self._index_tasks(summary)
                        stats["tasks_indexed"] += tasks_count

                        # Рендеринг Markdown
                        self.markdown_renderer.render_session_summary(
                            summary, force=self.force
                        )
                        self.markdown_renderer.render_snippets(
                            session, force=self.force
                        )

                        # Небольшая задержка
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.error(
                            f"Ошибка при обработке сессии {session['session_id']}: {e}"
                        )
                        # Логируем дополнительную информацию для отладки
                        if "Invalid IPv6 URL" in str(e):
                            logger.error(
                                f"Обнаружена ошибка IPv6 URL в сессии {session['session_id']}. "
                                f"Проверьте сообщения на наличие некорректных URL."
                            )
                        continue

                # Объединяем существующие и новые саммаризации для отчетов
                all_summaries = existing_summaries + processed_summaries

                # Создаём главную сводку чата из всех саммаризаций
                if all_summaries:
                    # Фильтруем сессии за последние 30 дней для раздела "Актуально"
                    now = datetime.now(ZoneInfo("UTC"))
                    thirty_days_ago = now - timedelta(days=30)

                    recent_sessions = [
                        s
                        for s in all_summaries
                        if self._parse_message_time(
                            {"date_utc": s.get("meta", {}).get("end_time_utc", "")}
                        )
                        >= thirty_days_ago
                    ]

                    # Сортируем по качеству (score) для топ-сессий
                    top_sessions = sorted(
                        recent_sessions,
                        key=lambda s: s.get("quality", {}).get("score", 0),
                        reverse=True,
                    )

                    self.markdown_renderer.render_chat_summary(
                        chat_name,
                        all_summaries,
                        top_sessions=top_sessions,
                        force=self.force,
                    )
                    # Создаём файл накапливающегося контекста
                    self.markdown_renderer.render_cumulative_context(
                        chat_name, all_summaries, force=self.force
                    )
                    # Создаём индекс сессий чата
                    self.markdown_renderer.render_chat_index(
                        chat_name, all_summaries, force=self.force
                    )

                    # Кластеризация сессий чата (только если есть новые сессии)
                    if (
                        self.enable_clustering
                        and len(processed_summaries) >= self.min_cluster_size
                    ):
                        logger.info(f"Запуск кластеризации для чата {chat_name}")
                        try:
                            # Кластеризуем все сессии чата, включая существующие
                            cluster_stats = await self._cluster_chat_sessions(
                                chat_name, all_summaries
                            )
                            stats["clusters_created"] = cluster_stats.get(
                                "clusters_count", 0
                            )
                            stats["sessions_clustered"] = cluster_stats.get(
                                "sessions_clustered", 0
                            )
                            logger.info(f"Кластеризация завершена: {cluster_stats}")
                        except Exception as e:
                            logger.error(
                                f"Ошибка при кластеризации чата {chat_name}: {e}"
                            )

                # Сохраняем прогресс индексации для инкрементальных обновлений
                if messages_to_index:
                    # Находим дату последнего проиндексированного сообщения
                    last_message_date = max(
                        self._parse_message_time(m) for m in messages_to_index
                    )
                    self._save_indexing_progress(
                        chat_name=chat_name,
                        last_message_date=last_message_date,
                        messages_count=len(messages_to_index),
                        sessions_count=len(processed_summaries),
                    )

                # Сохраняем словари сущностей если включено обучение
                if self.entity_dictionary and self.enable_entity_learning:
                    try:
                        self.entity_dictionary.save_dictionaries()
                        logger.info(f"Словари сущностей сохранены для чата {chat_name}")
                    except Exception as e:
                        logger.warning(f"Ошибка сохранения словарей сущностей: {e}")

                stats["indexed_chats"].append(chat_name)

                # Итоговая статистика по чату
                if skipped_sessions > 0:
                    logger.info(
                        f"✅ Чат {chat_name} завершен: "
                        f"{len(processed_summaries)} новых сессий, "
                        f"{skipped_sessions} пропущено (уже проиндексировано), "
                        f"{processed_messages_count} сообщений обработано"
                    )
                else:
                    logger.info(
                        f"✅ Чат {chat_name} завершен: "
                        f"{len(processed_summaries)} сессий, "
                        f"{processed_messages_count} сообщений обработано"
                    )

            except Exception as e:
                logger.error(f"Ошибка при индексации чата {chat_dir.name}: {e}")
                continue

        logger.info(f"Индексация завершена: {stats}")
        return stats

    async def _load_messages_from_chat(self, chat_dir: Path) -> List[Dict[str, Any]]:
        """
        Загрузка сообщений из JSON файлов чата (использует общую утилиту).

        Args:
            chat_dir: Директория чата

        Returns:
            Список сообщений
        """
        from ..utils.json_loader import load_json_or_jsonl

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

    async def _index_session_l1(self, summary: Dict[str, Any]):
        """
        Индексация сессии на уровне L1 (саммари + E1)

        Args:
            summary: Саммаризация сессии
        """
        session_id = summary["session_id"]

        meta = summary.get("meta", {})

        topics_text = "\n".join(
            f"{topic.get('title', '')}: {topic.get('summary', '')}"
            for topic in summary.get("topics", [])
        )
        claims_text = "\n".join(
            claim.get("summary", "") for claim in summary.get("claims", [])
        )
        discussion_text = "\n".join(
            item.get("quote", "") for item in summary.get("discussion", [])
        )
        entities_text = ", ".join(summary.get("entities", []))

        embedding_text = (
            f"Topics:\n{topics_text}\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Discussion:\n{discussion_text}\n\n"
            f"Entities: {entities_text}"
        )

        # Валидируем текст перед отправкой в эмбеддинг
        embedding_text, replaced_urls = validate_embedding_text(embedding_text)

        # Логируем замененные URL если они есть
        if replaced_urls:
            logger.warning(
                f"В сессии {session_id} заменены некорректные URL: {replaced_urls}"
            )

        # Генерируем эмбеддинг
        async with self.embedding_client:
            embeddings = await self.embedding_client.generate_embeddings([embedding_text])
            embedding = embeddings[0]

        # Подготавливаем метаданные
        metadata = {
            "session_id": session_id,
            "chat": meta.get("chat_name", ""),
            "profile": meta.get("profile", ""),
            "start_time_utc": meta.get("start_time_utc", ""),
            "end_time_utc": meta.get("end_time_utc", ""),
            "time_span": meta.get("time_span", ""),
            "message_count": meta.get("messages_total", 0),
            "dominant_language": meta.get("dominant_language", "unknown"),
            "chat_mode": meta.get("chat_mode", "group"),
            "topics_count": len(summary.get("topics", [])),
            "claims_count": len(summary.get("claims", [])),
            "quality_score": summary.get("quality", {}).get("score", 0),
            "replaced_urls": ",".join(replaced_urls)
            if replaced_urls
            else "",  # Сохраняем замененные URL как строку
        }

        # Добавляем в коллекцию
        self.sessions_collection.upsert(
            ids=[session_id],
            documents=[embedding_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        # Синхронизация с графом памяти
        if self.ingestor and self.graph:
            from ..indexing import MemoryRecord
            from ..utils.datetime_utils import parse_datetime_utc
            
            try:
                # Парсим timestamp из метаданных
                start_time_utc = meta.get("start_time_utc", "")
                timestamp = parse_datetime_utc(start_time_utc, default=None) if start_time_utc else None
                if not timestamp:
                    from datetime import datetime, timezone
                    timestamp = datetime.now(timezone.utc)
                
                # Создаём MemoryRecord для сессии
                record = MemoryRecord(
                    record_id=session_id,
                    source=meta.get("chat_name", "unknown"),
                    content=embedding_text,
                    timestamp=timestamp,
                    author=None,  # Сессии не имеют автора
                    tags=[],
                    entities=summary.get("entities", []),
                    attachments=[],
                    metadata={
                        "chat": meta.get("chat_name", ""),
                        "profile": meta.get("profile", ""),
                        "start_time_utc": start_time_utc,
                        "end_time_utc": meta.get("end_time_utc", ""),
                        "time_span": meta.get("time_span", ""),
                        "message_count": meta.get("messages_total", 0),
                        "dominant_language": meta.get("dominant_language", "unknown"),
                        "chat_mode": meta.get("chat_mode", "group"),
                        "topics_count": len(summary.get("topics", [])),
                        "claims_count": len(summary.get("claims", [])),
                        "quality_score": summary.get("quality", {}).get("score", 0),
                        "session_type": "session_summary",  # Помечаем как саммари сессии
                    },
                )
                
                # Сохраняем в граф
                self.ingestor.ingest([record])
                
                # Сохраняем эмбеддинг в граф
                if embedding:
                    try:
                        self.graph.update_node(
                            session_id,
                            embedding=embedding,
                        )
                    except Exception as e:
                        logger.debug(f"Ошибка при сохранении эмбеддинга сессии {session_id}: {e}")
                
                logger.debug(f"Синхронизирована сессия {session_id} с графом памяти")
            except Exception as e:
                logger.warning(f"Ошибка при синхронизации сессии {session_id} с графом: {e}")

        logger.info(f"L1: Проиндексирована сессия {session_id}")

    async def _index_messages_l2(self, session: Dict[str, Any]) -> int:
        """
        Индексация сообщений на уровне L2 (с симметричным контекстом)

        Args:
            session: Сессия

        Returns:
            Количество проиндексированных сообщений
        """
        messages = session["messages"]
        session_id = session["session_id"]
        chat = session["chat"]

        indexed_count = 0
        messages_to_index = []

        # Определим тип чата для подстройки контекста
        chat_mode = self._detect_chat_mode(messages)

        for i, msg in enumerate(messages):
            try:
                msg_text = msg.get("text", "").strip()
                if not msg_text or len(msg_text) < 10:
                    continue

                # Проверяем, существует ли уже это сообщение в базе
                msg_id = f"{session_id}-M{i+1:04d}"
                existing_msg = self.messages_collection.get(ids=[msg_id])
                if existing_msg and existing_msg.get("ids"):
                    # Сообщение уже существует, пропускаем
                    logger.debug(f"Сообщение {msg_id} уже существует, пропускаем")
                    continue

                # Добавляем симметричный контекст (до 10 сообщений, ≤ 1500 символов)
                # В каналах соседний контент менее релевантен — уменьшим контекст
                context_text = self._build_symmetric_context(
                    messages,
                    i,
                    max_messages=(3 if chat_mode == "channel" else 10),
                    max_chars=(500 if chat_mode == "channel" else 1500),
                )

                # Комбинируем текст + контекст
                embedding_text = f"{context_text}\n[CURRENT]: {msg_text}"

                # Проверяем и обрезаем текст, если он превышает лимит токенов
                # Оценка токенов: примерно 4 символа = 1 токен
                estimated_tokens = len(embedding_text) // 4
                max_tokens = 8192  # Уменьшаем до 8192 токенов для ускорения обработки

                if estimated_tokens > max_tokens:
                    # Сначала обрезаем контекст, если он слишком длинный
                    max_context_chars = (
                        2000  # Уменьшаем до 2000 символов (~500 токенов) для соответствия лимиту 8192
                    )
                    if len(context_text) > max_context_chars:
                        context_text = context_text[:max_context_chars] + "..."

                    # Затем обрезаем основное сообщение
                    # Учитываем длину контекста + "[CURRENT]: " (~3 токена)
                    remaining_chars = (max_tokens - len(context_text) // 4 - 3) * 4
                    if remaining_chars > 0:
                        msg_text = msg_text[:remaining_chars] + "..."
                        embedding_text = f"{context_text}\n[CURRENT]: {msg_text}"
                    else:
                        # Если контекст уже занимает почти весь лимит, используем только сообщение
                        msg_text = msg_text[: max_tokens * 4 - 10] + "..."
                        embedding_text = f"[CURRENT]: {msg_text}"

                    final_tokens = len(embedding_text) // 4
                    original_tokens = estimated_tokens
                    logger.warning(
                        f"Текст эмбеддинга обрезан до ~{final_tokens} токенов "
                        f"(исходная оценка: ~{original_tokens} токенов)"
                    )

                # Валидируем текст перед отправкой в эмбеддинг
                embedding_text, replaced_urls = validate_embedding_text(embedding_text)

                # Логируем замененные URL если они есть
                if replaced_urls:
                    logger.warning(
                        f"В сообщении {i+1} сессии {session_id} заменены некорректные URL: {replaced_urls}"
                    )

                # Сохраняем данные для батчевой обработки
                messages_to_index.append({
                    "msg_id": msg_id,
                    "msg_text": msg_text,
                    "embedding_text": embedding_text,
                    "msg_index": i,  # Сохраняем индекс для извлечения автора
                    "msg": msg,  # Сохраняем исходное сообщение для извлечения автора
                    "metadata": {
                        "msg_id": msg_id,
                        "session_id": session_id,
                        "chat": chat,
                        "date_utc": msg.get("date_utc") or msg.get("date", ""),
                        "has_context": len(context_text) > 0,
                        "context_length": len(context_text),
                        "chat_mode": chat_mode,
                        "replaced_urls": ",".join(replaced_urls)
                        if replaced_urls
                        else "",  # Сохраняем замененные URL как строку
                    }
                })

            except Exception as e:
                logger.error(
                    f"Ошибка при подготовке сообщения {i} в сессии {session_id} для индексации: {e}"
                )
                # Логируем дополнительную информацию для отладки
                if "Invalid IPv6 URL" in str(e):
                    logger.error(
                        f"Обнаружена ошибка IPv6 URL в сообщении {i} сессии {session_id}. "
                        f"Текст сообщения: {msg.get('text', '')[:100]}..."
                    )
                continue

        # Генерируем эмбеддинги батчами
        if messages_to_index:
            try:
                async with self.embedding_client:
                    # Собираем тексты для батча
                    batch_texts = [msg["embedding_text"] for msg in messages_to_index]
                    
                    # Генерируем эмбеддинги батчем
                    embeddings = await self.embedding_client.generate_embeddings(batch_texts, batch_size=32)
                    
                    # Добавляем в коллекцию батчем
                    ids = [msg["msg_id"] for msg in messages_to_index]
                    documents = [msg["msg_text"] for msg in messages_to_index]
                    metadatas = [msg["metadata"] for msg in messages_to_index]
                    
                    self.messages_collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                    
                    # Синхронизация с графом памяти
                    if self.ingestor and self.graph:
                        from ..indexing import MemoryRecord, Attachment
                        from ..utils.datetime_utils import parse_datetime_utc
                        
                        records_to_ingest = []
                        for idx, msg_data in enumerate(messages_to_index):
                            try:
                                msg_id = msg_data["msg_id"]
                                msg_text = msg_data["msg_text"]
                                metadata = msg_data["metadata"]
                                embedding = embeddings[idx] if idx < len(embeddings) else None
                                
                                # Парсим timestamp
                                date_utc = metadata.get("date_utc", "")
                                timestamp = parse_datetime_utc(date_utc, default=None) if date_utc else None
                                if not timestamp:
                                    # Используем текущее время как fallback
                                    from datetime import datetime, timezone
                                    timestamp = datetime.now(timezone.utc)
                                
                                # Извлекаем автора из исходного сообщения
                                author = None
                                msg_obj = msg_data.get("msg")
                                if msg_obj:
                                    from_data = msg_obj.get("from") or {}
                                    author = from_data.get("username") or from_data.get("display") or from_data.get("id")
                                
                                # Создаём MemoryRecord
                                record = MemoryRecord(
                                    record_id=msg_id,
                                    source=metadata.get("chat", "unknown"),
                                    content=msg_text,
                                    timestamp=timestamp,
                                    author=author,
                                    tags=[],
                                    entities=[],
                                    attachments=[],
                                    metadata={
                                        "chat": metadata.get("chat", ""),
                                        "session_id": metadata.get("session_id", ""),
                                        "has_context": metadata.get("has_context", False),
                                        "context_length": metadata.get("context_length", 0),
                                        "chat_mode": metadata.get("chat_mode", "group"),
                                        "date_utc": date_utc,
                                    },
                                )
                                records_to_ingest.append((record, embedding))
                            except Exception as e:
                                logger.warning(f"Ошибка при подготовке записи {msg_data.get('msg_id', 'unknown')} для графа: {e}")
                                continue
                        
                        # Сохраняем записи в граф батчем
                        if records_to_ingest:
                            try:
                                records_only = [r for r, _ in records_to_ingest]
                                self.ingestor.ingest(records_only)
                                
                                # Сохраняем эмбеддинги в граф
                                for record, embedding in records_to_ingest:
                                    if embedding:
                                        try:
                                            self.graph.update_node(
                                                record.record_id,
                                                embedding=embedding,
                                            )
                                        except Exception as e:
                                            logger.debug(f"Ошибка при сохранении эмбеддинга для {record.record_id}: {e}")
                                
                                logger.debug(f"Синхронизировано {len(records_to_ingest)} записей с графом памяти")
                            except Exception as e:
                                logger.warning(f"Ошибка при синхронизации записей с графом: {e}")
                    
                    indexed_count += len(messages_to_index)
            except Exception as e:
                logger.error(
                    f"Ошибка при индексации сообщений в сессии {session_id}: {e}"
                )

        logger.info(
            f"L2: Проиндексировано {indexed_count} сообщений из сессии {session_id}"
        )
        return indexed_count

    def _detect_chat_mode(self, messages: List[Dict[str, Any]]) -> str:
        """Локальная эвристика определения типа чата: 'channel' или 'group'."""
        authors = []
        for m in messages:
            fr = m.get("from") or {}
            name = fr.get("username") or fr.get("display") or fr.get("id") or "unknown"
            authors.append(str(name))
        total = len([a for a in authors if a != "unknown"])
        if total == 0:
            return "group"
        cnt = Counter(a for a in authors if a != "unknown")
        top, top_count = cnt.most_common(1)[0]
        top_share = top_count / total
        unique = len(cnt)
        if (top_share >= 0.85 and unique <= 3 and total >= 5) or unique == 1:
            return "channel"
        return "group"

    def _build_symmetric_context(
        self,
        messages: List[Dict[str, Any]],
        current_idx: int,
        max_messages: int = 10,
        max_chars: int = 1500,
    ) -> str:
        """
        Построение симметричного контекста для сообщения

        Args:
            messages: Список сообщений
            current_idx: Индекс текущего сообщения
            max_messages: Максимум сообщений в контексте (по умолчанию 10)
            max_chars: Максимум символов в контексте (по умолчанию 1500)

        Returns:
            Текст контекста
        """
        context_parts = []
        total_chars = 0

        # Добавляем парами: -1, +1, -2, +2, ...
        distance = 1
        while len(context_parts) < max_messages and distance <= max_messages // 2:
            # Предыдущее сообщение
            prev_idx = current_idx - distance
            if prev_idx >= 0:
                prev_text = messages[prev_idx].get("text", "").strip()
                if prev_text and total_chars + len(prev_text) <= max_chars:
                    context_parts.insert(0, prev_text)
                    total_chars += len(prev_text)

            # Следующее сообщение
            next_idx = current_idx + distance
            if next_idx < len(messages):
                next_text = messages[next_idx].get("text", "").strip()
                if next_text and total_chars + len(next_text) <= max_chars:
                    context_parts.append(next_text)
                    total_chars += len(next_text)

            distance += 1

        return " | ".join(context_parts)

    async def _index_tasks(self, summary: Dict[str, Any]) -> int:
        """
        Индексация Action Items в коллекцию tasks

        Args:
            summary: Саммаризация сессии

        Returns:
            Количество проиндексированных задач
        """
        actions = summary.get("actions", [])
        session_id = summary["session_id"]
        chat = summary.get("meta", {}).get("chat_name", "")

        indexed_count = 0

        for i, action in enumerate(actions):
            confidence = action.get("confidence", 0.8)
            if confidence < 0.6:
                continue

            try:
                # Текст задачи
                task_text = action.get("text", "")
                if not task_text:
                    continue

                # Генерируем эмбеддинг
                async with self.embedding_client:
                    embeddings = await self.embedding_client.generate_embeddings(
                        [task_text]
                    )
                    embedding = embeddings[0]

                # Подготавливаем метаданные
                task_id = f"{session_id}-T{i+1:02d}"
                metadata = {
                    "task_id": task_id,
                    "session_id": session_id,
                    "chat": chat,
                    "owner": action.get("owner", ""),
                    "due": action.get("due", ""),
                    "priority": action.get("priority", "normal"),
                    "confidence": confidence,
                    "msg_id": action.get("msg_id", ""),
                    "topic_title": action.get("topic_title", ""),
                }

                # Добавляем в коллекцию
                self.tasks_collection.upsert(
                    ids=[task_id],
                    documents=[task_text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )

                indexed_count += 1

            except Exception as e:
                logger.error(
                    f"Ошибка при индексации задачи {i} в сессии {session_id}: {e}"
                )
                continue

        logger.info(
            f"Tasks: Проиндексировано {indexed_count} задач из сессии {session_id}"
        )
        return indexed_count

    def _parse_message_time(self, msg: Dict[str, Any]) -> datetime:
        """Парсинг времени сообщения (использует общую утилиту)."""
        from ..utils.datetime_utils import parse_message_time

        return parse_message_time(msg, use_zoneinfo=True)

    async def _cluster_chat_sessions(
        self, chat_name: str, summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Кластеризация сессий чата и сохранение результатов

        Args:
            chat_name: Название чата
            summaries: Список саммаризаций сессий

        Returns:
            Статистика кластеризации
        """
        if not self.session_clusterer or not self.cluster_summarizer:
            return {"clusters_count": 0, "sessions_clustered": 0}

        # Получаем эмбеддинги из ChromaDB для сессий
        session_ids = [s["session_id"] for s in summaries]

        try:
            result = self.sessions_collection.get(
                ids=session_ids, include=["embeddings", "metadatas", "documents"]
            )

            if not result["ids"]:
                logger.warning(f"Не найдены эмбеддинги для сессий чата {chat_name}")
                return {"clusters_count": 0, "sessions_clustered": 0}

            # Преобразуем в формат для кластеризации
            sessions_data = []
            embeddings_list = []

            for i, session_id in enumerate(result["ids"]):
                sessions_data.append(
                    {
                        "session_id": session_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )
                embeddings_list.append(result["embeddings"][i])

            # Выполняем кластеризацию
            logger.info(f"Кластеризация {len(sessions_data)} сессий чата {chat_name}")
            clustering_result = self.session_clusterer.cluster_sessions(
                sessions_data, embeddings_list
            )

            clusters = clustering_result.get("clusters", [])
            logger.info(f"Найдено {len(clusters)} кластеров")

            # Сохраняем кластеры в ChromaDB и обновляем метаданные сессий
            clusters_saved = 0
            sessions_clustered = 0

            for cluster in clusters:
                cluster_id = f"{slugify(chat_name)}-cluster-{cluster['cluster_id']}"

                # Обновляем метаданные сессий, добавляя информацию о кластере
                for session_id in cluster["session_ids"]:
                    try:
                        # Получаем текущую метаданные сессии
                        session_data = self.sessions_collection.get(
                            ids=[session_id], include=["metadatas"]
                        )

                        if session_data["ids"]:
                            metadata = session_data["metadatas"][0]
                            metadata["cluster_id"] = cluster_id
                            metadata["cluster_label"] = cluster.get("label", "")

                            # Обновляем метаданные
                            self.sessions_collection.update(
                                ids=[session_id], metadatas=[metadata]
                            )
                            sessions_clustered += 1
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении метаданных сессии {session_id}: {e}"
                        )

                # Генерируем эмбеддинг для кластера (среднее эмбеддингов сессий)
                cluster_embedding = [0.0] * len(embeddings_list[0])
                for session_id in cluster["session_ids"]:
                    try:
                        idx = session_ids.index(session_id)
                        session_emb = embeddings_list[idx]
                        for i, val in enumerate(session_emb):
                            cluster_embedding[i] += val
                    except ValueError:
                        continue

                # Нормализуем
                n = len(cluster["session_ids"])
                if n > 0:
                    cluster_embedding = [val / n for val in cluster_embedding]

                # Создаём документ для кластера
                cluster_doc = (
                    f"Кластер: {cluster.get('label', 'Без названия')}\n"
                    f"Ключевые слова: {', '.join(cluster.get('keywords', []))}\n"
                    f"Топики: {', '.join(cluster.get('topics', []))}\n"
                    f"Сущности: {', '.join(cluster.get('entities', []))}"
                )

                # Метаданные кластера
                cluster_metadata = {
                    "cluster_id": cluster_id,
                    "chat": chat_name,
                    "label": cluster.get("label", ""),
                    "size": cluster.get("size", 0),
                    "coherence": cluster.get("coherence", 0.0),
                    "session_ids": ",".join(cluster["session_ids"][:10]),  # Первые 10
                }

                # Сохраняем кластер
                try:
                    self.clusters_collection.upsert(
                        ids=[cluster_id],
                        documents=[cluster_doc],
                        embeddings=[cluster_embedding],
                        metadatas=[cluster_metadata],
                    )
                    clusters_saved += 1
                    logger.info(
                        f"Сохранён кластер {cluster_id}: {cluster.get('label', '')}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка при сохранении кластера {cluster_id}: {e}")

            return {
                "clusters_count": clusters_saved,
                "sessions_clustered": sessions_clustered,
                "total_sessions": len(sessions_data),
                "noise_sessions": clustering_result.get("noise_count", 0),
            }

        except Exception as e:
            logger.error(f"Ошибка при кластеризации чата {chat_name}: {e}")
            return {"clusters_count": 0, "sessions_clustered": 0}

    def get_clusters(
        self, chat: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Получить список кластеров

        Args:
            chat: Фильтр по чату (опционально)
            limit: Максимальное количество кластеров

        Returns:
            Список кластеров
        """
        if not self.clusters_collection:
            return []

        try:
            where_filter = {"chat": chat} if chat else None

            result = self.clusters_collection.get(
                where=where_filter, limit=limit, include=["metadatas", "documents"]
            )

            clusters = []
            for i, cluster_id in enumerate(result["ids"]):
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )

            return clusters
        except Exception as e:
            logger.error(f"Ошибка при получении кластеров: {e}")
            return []

    def get_cluster_sessions(self, cluster_id: str) -> List[Dict[str, Any]]:
        """
        Получить сессии, принадлежащие кластеру

        Args:
            cluster_id: ID кластера

        Returns:
            Список сессий
        """
        if not self.sessions_collection:
            return []

        try:
            result = self.sessions_collection.get(
                where={"cluster_id": cluster_id}, include=["metadatas", "documents"]
            )

            sessions = []
            for i, session_id in enumerate(result["ids"]):
                sessions.append(
                    {
                        "session_id": session_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )

            return sessions
        except Exception as e:
            logger.error(f"Ошибка при получении сессий кластера {cluster_id}: {e}")
            return []

    def _count_indexed_messages_in_chat(self, chat_name: str) -> int:
        """
        Подсчитывает количество уже проиндексированных сообщений в чате

        Returns:
            int: количество проиндексированных сообщений
        """
        try:
            # Получаем все сообщения из коллекции chat_messages для данного чата
            messages_collection = self.chroma_client.get_collection("chat_messages")
            existing_messages = messages_collection.get(where={"chat": chat_name})

            if existing_messages and existing_messages.get("ids") is not None:
                message_count = len(existing_messages["ids"])
                logger.info(
                    f"Найдено {message_count} уже проиндексированных сообщений в чате {chat_name}"
                )
                return message_count
            else:
                logger.info(f"В чате {chat_name} нет проиндексированных сообщений")
                return 0

        except Exception as e:
            logger.warning(
                f"Ошибка при подсчете проиндексированных сообщений для чата {chat_name}: {e}"
            )
            return 0

    def _group_messages_by_smart_strategy(
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
        indexed_messages_count = self._count_indexed_messages_in_chat(chat_name)

        # Определяем стратегию на основе количества уже проиндексированных сообщений
        if indexed_messages_count >= self.strategy_threshold:
            logger.info(
                f"🔄 Переход стратегии для чата {chat_name}: "
                f"уже проиндексировано {indexed_messages_count} сообщений "
                f"(порог: {self.strategy_threshold})"
            )

            # Определяем текущую стратегию на основе существующих сессий
            current_strategy = self._determine_current_strategy(chat_name)
            logger.info(f"📊 Текущая стратегия для чата {chat_name}: {current_strategy}")

            # Переходим к следующей стратегии
            next_strategy = self._get_next_strategy(current_strategy)
            logger.info(f"➡️  Переход к стратегии: {next_strategy}")

            # Применяем новую стратегию
            return self._apply_strategy_transition(
                messages, chat_name, next_strategy, current_date
            )

        # Определяем начальный номер сессии на основе существующих сессий
        existing_session_ids = set()
        try:
            existing_sessions = self.sessions_collection.get(where={"chat": chat_name})
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
        return self._apply_strategy_transition(
            messages, chat_name, "fresh", current_date
        )

    def _determine_current_strategy(self, chat_name: str) -> str:
        """
        Определяет текущую стратегию на основе существующих сессий

        Returns:
            str: текущая стратегия (fresh, recent, old)
        """
        try:
            existing_sessions = self.sessions_collection.get(where={"chat": chat_name})

            if not existing_sessions or not existing_sessions.get("ids"):
                return "fresh"  # По умолчанию начинаем с fresh

            # Анализируем существующие сессии для определения стратегии
            strategies_found = set()
            for session_id in existing_sessions["ids"]:
                if f"{slugify(chat_name)}-" in session_id:
                    parts = session_id.split("-")
                    if len(parts) >= 3:
                        window_name = parts[1]
                        if window_name in ["fresh", "recent", "old"]:
                            strategies_found.add(window_name)

            # Определяем текущую стратегию на основе найденных окон
            if "old" in strategies_found:
                return "old"
            elif "recent" in strategies_found:
                return "recent"
            elif "fresh" in strategies_found:
                return "fresh"
            else:
                return "fresh"  # По умолчанию

        except Exception as e:
            logger.warning(
                f"Ошибка при определении текущей стратегии для чата {chat_name}: {e}"
            )
            return "fresh"

    def _get_next_strategy(self, current_strategy: str) -> str:
        """
        Определяет следующую стратегию в последовательности

        Args:
            current_strategy: текущая стратегия

        Returns:
            str: следующая стратегия
        """
        strategy_sequence = ["fresh", "recent", "old"]

        try:
            current_index = strategy_sequence.index(current_strategy)
            if current_index < len(strategy_sequence) - 1:
                return strategy_sequence[current_index + 1]
            else:
                return "old"  # Остаемся на последней стратегии
        except ValueError:
            logger.warning(
                f"Неизвестная стратегия: {current_strategy}, возвращаем fresh"
            )
            return "fresh"

    def _apply_strategy_transition(
        self,
        messages: List[Dict[str, Any]],
        chat_name: str,
        strategy: str,
        current_date,
    ) -> List[Dict[str, Any]]:
        """
        Применяет переход к новой стратегии

        Args:
            messages: список сообщений
            chat_name: название чата
            strategy: новая стратегия (fresh, recent, old)
            current_date: текущая дата

        Returns:
            List[Dict[str, Any]]: список сессий
        """
        logger.info(f"🎯 Применение стратегии '{strategy}' для чата {chat_name}")

        # Определяем возраст сообщений и группируем по окнам
        now_messages = []
        fresh_messages = []
        recent_messages = []
        old_messages = []

        for msg in messages:
            if "date_utc" not in msg:
                continue

            try:
                from ..utils.datetime_utils import parse_datetime_utc

                msg_date = parse_datetime_utc(msg["date_utc"], use_zoneinfo=True)
                age_days = (current_date - msg_date).days

                if age_days <= 1:
                    now_messages.append(msg)
                elif age_days <= 14:
                    fresh_messages.append(msg)
                elif age_days <= 30:
                    recent_messages.append(msg)
                else:
                    old_messages.append(msg)
            except Exception as e:
                logger.warning(f"Ошибка парсинга даты сообщения: {e}")
                continue

        # Применяем стратегию в зависимости от выбранной
        sessions = []

        if strategy == "fresh":
            # Обрабатываем только NOW и FRESH окна
            # Если в этих окнах нет сообщений, но есть в OLD, автоматически переключаемся на OLD
            if not now_messages and not fresh_messages and old_messages:
                logger.info(
                    f"⚠️  В окнах NOW и FRESH нет сообщений, но есть {len(old_messages)} в OLD. "
                    "Автоматически переключаемся на стратегию OLD."
                )
                strategy = "old"
                window_strategies = [
                    ("now", now_messages, "session"),
                    ("fresh", fresh_messages, "day"),
                    ("recent", recent_messages, "week"),
                    ("old", old_messages, "month"),
                ]
            else:
                window_strategies = [
                    ("now", now_messages, "session"),
                    ("fresh", fresh_messages, "day"),
                ]
        elif strategy == "recent":
            # Обрабатываем NOW, FRESH и RECENT окна
            window_strategies = [
                ("now", now_messages, "session"),
                ("fresh", fresh_messages, "day"),
                ("recent", recent_messages, "week"),
            ]
        else:  # strategy == "old"
            # Обрабатываем все окна
            window_strategies = [
                ("now", now_messages, "session"),
                ("fresh", fresh_messages, "day"),
                ("recent", recent_messages, "week"),
                ("old", old_messages, "month"),
            ]

        # Определяем максимальный номер сессии для каждого окна
        existing_session_ids = set()
        try:
            existing_sessions = self.sessions_collection.get(where={"chat": chat_name})
            if existing_sessions and existing_sessions.get("ids"):
                existing_session_ids = set(existing_sessions["ids"])
        except Exception as e:
            logger.warning(f"Ошибка при получении существующих сессий: {e}")

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

        # Группируем каждое окно по своей стратегии
        for window_name, window_messages, window_strategy in window_strategies:
            if not window_messages:
                continue

            logger.info(
                f"Окно '{window_name}': {len(window_messages)} сообщений, стратегия: {window_strategy}"
            )

            # Используем новый метод группировки
            window_sessions = (
                self.day_grouping_segmenter.group_messages_by_window_strategy(
                    window_messages, chat_name, window_strategy
                )
            )

            # Добавляем информацию об окне к сессиям
            window_counter = window_max_numbers.get(window_name, 0) + 1
            for session in window_sessions:
                session["window"] = window_name
                session[
                    "session_id"
                ] = f"{slugify(chat_name)}-{window_name}-S{window_counter:04d}"
                window_counter += 1

            sessions.extend(window_sessions)

        logger.info(
            f"✅ Стратегия '{strategy}' применена: создано {len(sessions)} сессий"
        )
        return sessions


if __name__ == "__main__":
    # Тест модуля
    async def test():
        indexer = TwoLevelIndexer()
        stats = await indexer.build_index(scope="all", recent_days=7)
        print(f"Статистика индексации: {stats}")

    asyncio.run(test())
