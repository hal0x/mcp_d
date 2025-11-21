#!/usr/bin/env python3
"""Двухуровневая индексация: L1 (sessions с саммари) и L2 (messages с контекстом)."""

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..memory.qdrant_collections import QdrantCollectionsManager

from ..analysis.adaptive_message_grouper import AdaptiveMessageGrouper
from ..analysis.cluster_summarizer import ClusterSummarizer
from ..analysis.day_grouping import DayGroupingSegmenter
from ..analysis.entity_extraction import EntityExtractor
from ..analysis.entity_dictionary import get_entity_dictionary
from ..analysis.instruction_manager import InstructionManager
from ..analysis.markdown_renderer import MarkdownRenderer
from ..analysis.semantic_regrouper import SemanticRegrouper
from ..analysis.session_clustering import SessionClusterer
from ..analysis.session_segmentation import SessionSegmenter
from ..analysis.session_summarizer import SessionSummarizer
from ..analysis.time_processor import TimeProcessor
from ..utils.naming import slugify
from ..utils.url_validator import validate_embedding_text
from .lmstudio_client import LMStudioEmbeddingClient

logger = logging.getLogger(__name__)

MIN_SESSION_MESSAGES = 15


class TwoLevelIndexer:
    """Двухуровневая индексация: L1 (sessions с саммари) и L2 (messages с контекстом)."""

    def __init__(
        self,
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
        graph: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        enable_message_grouping: bool = True,
        message_grouping_strategy: str = "session",
        min_group_size: int = 3,
        max_group_size: int = 50,
        max_group_tokens: int = 8000,
    ):
        """Инициализирует индексатор с указанными параметрами.

        Args:
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
            progress_callback: Callback функция для отслеживания прогресса (job_id, event, data)
            enable_message_grouping: Включить группировку сообщений для эмбеддингов
            message_grouping_strategy: Стратегия группировки ("session"/"semantic"/"adaptive")
            min_group_size: Минимальный размер группы сообщений
            max_group_size: Максимальный размер группы сообщений
            max_group_tokens: Максимальное количество токенов в группе
        """
        self.progress_callback = progress_callback
        
        # Инициализируем embedding_client ПЕРЕД использованием
        self.artifacts_path = Path(artifacts_path).expanduser()
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.reports_path = self.artifacts_path / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client or LMStudioEmbeddingClient()
        
        # Инициализируем Qdrant для векторного хранилища
        from ..config import get_settings
        settings = get_settings()
        qdrant_url = settings.get_qdrant_url()
        if qdrant_url:
            # Получаем размерность эмбеддингов
            embedding_dimension = self.embedding_client.dimension if self.embedding_client else 1024
            self.qdrant_manager = QdrantCollectionsManager(url=qdrant_url, vector_size=embedding_dimension)
            if not self.qdrant_manager.available():
                logger.warning("Qdrant недоступен, коллекции не будут созданы")
                self.qdrant_manager = None
        else:
            logger.warning("QDRANT_URL не установлен, коллекции не будут созданы")
            self.qdrant_manager = None
        
        # Qdrant используется для векторного хранилища

        self.enable_clustering = enable_clustering
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
        self.max_messages_per_group = max_messages_per_group
        self.max_session_hours = max_session_hours
        self.gap_minutes = gap_minutes
        self.enable_smart_aggregation = enable_smart_aggregation
        self.aggregation_strategy = aggregation_strategy
        self.now_window_hours = now_window_hours
        self.fresh_window_days = fresh_window_days
        self.recent_window_days = recent_window_days
        self.strategy_threshold = strategy_threshold
        self.force = force
        self.enable_entity_learning = enable_entity_learning
        self.enable_time_analysis = enable_time_analysis
        self.enable_message_grouping = enable_message_grouping
        self.message_grouping_strategy = message_grouping_strategy
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_group_tokens = max_group_tokens

        self.session_segmenter = SessionSegmenter(
            gap_minutes=gap_minutes,
            max_session_hours=max_session_hours,
            enable_time_analysis=enable_time_analysis,
        )
        self.day_grouping_segmenter = DayGroupingSegmenter(
            max_messages_per_group=max_messages_per_group,
        )

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
            enable_llm_validation=True,
        )
        
        self.time_processor = TimeProcessor() if enable_time_analysis else None
        
        # Инициализируем словарь сущностей с настройками из config
        if enable_entity_learning:
            from ..config import get_settings
            settings = get_settings()
            self.entity_dictionary = get_entity_dictionary(
                enable_llm_validation=True,
                enable_description_generation=settings.entity_description_enabled,
                graph=graph,
            )
        else:
            self.entity_dictionary = None
        
        self.markdown_renderer = MarkdownRenderer(self.reports_path)

        self.session_clusterer = None
        self.cluster_summarizer = None
        self.sessions_collection = None
        self.messages_collection = None
        self.tasks_collection = None
        self.clusters_collection = None
        self.progress_collection = None

        self._initialize_collections()

        if self.enable_clustering:
            self.session_clusterer = SessionClusterer(
                similarity_threshold=self.clustering_threshold,
                min_cluster_size=self.min_cluster_size,
                use_hdbscan=False,
            )
            self.cluster_summarizer = ClusterSummarizer(
                embedding_client=self.embedding_client
            )

        if self.enable_smart_aggregation:
            from ..analysis.smart_rolling_aggregator import SmartRollingAggregator
            self.smart_aggregator = SmartRollingAggregator(
                chats_dir=Path("chats"),
                use_smart_strategy=(self.aggregation_strategy == "smart"),
            )
        else:
            self.smart_aggregator = None

        # Инициализируем компоненты группировки сообщений для эмбеддингов
        self.semantic_regrouper = None
        self.adaptive_grouper = None
        if self.enable_message_grouping:
            if self.message_grouping_strategy == "semantic":
                self.semantic_regrouper = SemanticRegrouper(embedding_client=self.embedding_client)
            elif self.message_grouping_strategy == "adaptive":
                self.adaptive_grouper = AdaptiveMessageGrouper(
                    max_tokens=max_group_tokens,
                    strategy="hybrid"
                )

        self.graph = graph
        if self.graph:
            from ..memory.ingest import MemoryIngestor
            self.ingestor = MemoryIngestor(self.graph)
            logger.info("TwoLevelIndexer: граф памяти подключен, записи будут синхронизироваться")
        else:
            self.ingestor = None
            logger.debug("TwoLevelIndexer: граф памяти не подключен, записи будут только в Qdrant")
        
        # Инициализируем VectorStore для сохранения эмбеддингов в Qdrant
        from ..memory.vector_store import build_vector_store_from_env
        self.vector_store = build_vector_store_from_env()
        if self.vector_store and self.vector_store.available():
            logger.info("VectorStore (Qdrant) инициализирован для векторного поиска")
            # Убеждаемся, что коллекция создана с правильной размерностью
            if self.embedding_client:
                try:
                    dimension = self.embedding_client.dimension
                    if dimension:
                        self.vector_store.ensure_collection(dimension)
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать коллекцию Qdrant: {e}")
        else:
            logger.warning("VectorStore (Qdrant) недоступен - векторный поиск не будет работать")
            self.vector_store = None
        
        # Инициализируем EntityVectorStore для индексации сущностей
        if enable_entity_learning:
            from ..memory.vector_store import build_entity_vector_store_from_env
            self.entity_vector_store = build_entity_vector_store_from_env()
            if self.entity_vector_store:
                logger.info("EntityVectorStore инициализирован для индексации сущностей")
            else:
                logger.debug("EntityVectorStore недоступен (Qdrant не настроен)")
        else:
            self.entity_vector_store = None

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
                return None
        except Exception as e:
            logger.debug(f"Не удалось определить размерность эмбеддингов: {e}")
            return None

    def _check_and_recreate_collection(self, collection_name: str, description: str, force_recreate: bool = False):
        """Проверить коллекцию Qdrant и пересоздать при несоответствии размерности."""
        if not self.qdrant_manager or not self.qdrant_manager.available():
            logger.warning(f"Qdrant недоступен, коллекция {collection_name} не будет создана")
            return None
        
        # Обновляем размерность в менеджере, если нужно
        expected_dimension = self._get_embedding_dimension()
        if expected_dimension and expected_dimension != self.qdrant_manager.vector_size:
            logger.info(f"Обновление размерности Qdrant менеджера: {self.qdrant_manager.vector_size} -> {expected_dimension}")
            self.qdrant_manager.vector_size = expected_dimension
        
        # Создаем/проверяем коллекцию через менеджер
        if self.qdrant_manager.ensure_collection(collection_name, force_recreate=force_recreate):
            count = self.qdrant_manager.count(collection_name)
            logger.info(f"Коллекция Qdrant {collection_name} готова ({count} записей)")
            return collection_name  # Возвращаем имя коллекции вместо объекта
        else:
            logger.error(f"Не удалось создать коллекцию Qdrant {collection_name}")
            return None

    def _initialize_collections(self):
        """Инициализирует коллекции Qdrant с проверкой размерности эмбеддингов."""
        if not self.qdrant_manager or not self.qdrant_manager.available():
            logger.warning("Qdrant недоступен, коллекции не будут инициализированы")
            self.sessions_collection = None
            self.messages_collection = None
            self.tasks_collection = None
            self.clusters_collection = None
            self.progress_collection = None
            return
        
        try:
            self.sessions_collection = self._check_and_recreate_collection(
                "chat_sessions",
                "Саммаризации сессий для векторного поиска (L1)",
                force_recreate=self.force
            )

            self.messages_collection = self._check_and_recreate_collection(
                "chat_messages",
                "Сообщения с контекстом для уточняющего поиска (L2)",
                force_recreate=self.force
            )

            self.tasks_collection = self._check_and_recreate_collection(
                "chat_tasks",
                "Action Items из сессий",
                force_recreate=self.force
            )

            self.clusters_collection = self._check_and_recreate_collection(
                "session_clusters",
                "Тематические кластеры сессий",
                force_recreate=self.force
            )

            # indexing_progress теперь хранится в SQLite через IndexingJobTracker
            # Qdrant не используется для прогресса
            self.progress_collection = None
            logger.info("Коллекции Qdrant инициализированы")

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

    def _call_progress_callback(
        self, job_id: Optional[str], event: str, data: Dict[str, Any]
    ) -> None:
        """Вызвать callback прогресса, если он установлен."""
        if self.progress_callback and job_id:
            try:
                self.progress_callback(job_id, event, data)
            except Exception as e:
                logger.warning(f"Ошибка при вызове progress_callback: {e}")

    async def build_index(
        self,
        scope: str = "all",
        chat: Optional[str] = None,
        force_full: bool = False,
        recent_days: int = 7,
        adapter: Optional[Any] = None,  # MemoryServiceAdapter, но избегаем циклического импорта
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Построение индекса

        Args:
            scope: "all" или "chat"
            chat: Название чата (для scope="chat")
            force_full: Полная пересборка
            recent_days: Пересаммаризировать последние N дней
            adapter: Адаптер памяти для синхронизации
            job_id: Идентификатор задачи для отслеживания прогресса

        Returns:
            Статистика индексации
        """
        logger.info(
            f"Начало индексации: scope={scope}, chat={chat}, force_full={force_full}"
        )
        if scope == "chat" and chat:
            logger.info(f"🎯 Режим индексации одного чата: '{chat}'")
        elif scope == "all":
            logger.info("🌐 Режим индексации всех чатов")
        else:
            logger.warning(f"⚠️ Неизвестный scope={scope}, будет использован режим 'all'")

        stats = {
            "indexed_chats": [],
            "sessions_indexed": 0,
            "messages_indexed": 0,
            "tasks_indexed": 0,
        }

        # Получаем список чатов для индексации
        chats_path = Path("chats")
        if scope == "chat" and chat:
            chat_dir = chats_path / chat
            if not chat_dir.exists() or not chat_dir.is_dir():
                logger.error(f"❌ Чат '{chat}' не найден в {chats_path}")
                return {
                    "indexed_chats": [],
                    "sessions_indexed": 0,
                    "messages_indexed": 0,
                    "tasks_indexed": 0,
                    "error": f"Чат '{chat}' не найден",
                }
            chat_dirs = [chat_dir]
        else:
            chat_dirs = [d for d in chats_path.iterdir() if d.is_dir()]

        # Индексируем каждый чат
        total_chats = len(chat_dirs)
        for chat_idx, chat_dir in enumerate(chat_dirs, 1):
            try:
                chat_name = chat_dir.name
                logger.info(f"📁 Чат {chat_idx}/{total_chats}: {chat_name}")
                
                # Callback: начало обработки чата
                self._call_progress_callback(
                    job_id,
                    "chat_started",
                    {
                        "chat": chat_name,
                        "chat_index": chat_idx,
                        "total_chats": total_chats,
                    },
                )

                # Очистка старых данных при полной переиндексации
                if force_full and adapter is not None:
                    logger.info(f"🧹 Очистка старых данных чата {chat_name} перед переиндексацией...")
                    try:
                        cleanup_stats = adapter.clear_chat_data(chat_name)
                        logger.info(
                            f"✅ Очистка завершена: "
                            f"узлов={cleanup_stats.get('nodes_deleted', 0)}, "
                            f"векторов={cleanup_stats.get('vectors_deleted', 0)}, "
                            f"Qdrant={cleanup_stats.get('qdrant_deleted', 0)}"
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
                    try:
                        aggregation_result = await self.smart_aggregator.aggregate_chat(
                            chat_name, dry_run=False
                        )
                        # Проверяем, что результат - словарь
                        if isinstance(aggregation_result, dict):
                            sessions = aggregation_result.get("sessions", [])
                        else:
                            logger.error(
                                f"Неожиданный тип результата агрегации: {type(aggregation_result)}"
                            )
                            sessions = []
                        logger.info(f"Умная агрегация создала {len(sessions)} сессий")
                    except Exception as e:
                        logger.error(f"Ошибка при умной агрегации: {e}", exc_info=True)
                        # Fallback на классическую группировку
                        logger.info("Переключаемся на классическую группировку")
                        sessions = self._group_messages_by_smart_strategy(
                            messages_to_index, chat_name
                        )
                        logger.info(f"Создано {len(sessions)} сессий с умной стратегией")
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
                        result = None
                        if self.qdrant_manager and self.sessions_collection:
                            result = self.qdrant_manager.get(
                                collection_name=self.sessions_collection,
                                where={"chat": chat_name}
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
                        
                        # Callback: обработка сессий
                        self._call_progress_callback(
                            job_id,
                            "sessions_processing",
                            {
                                "chat": chat_name,
                                "session_index": session_idx,
                                "total_sessions": len(sessions),
                                "sessions_count": len(processed_summaries),
                                "messages_count": processed_messages_count,
                                "total_messages": total_messages_in_chat,
                                "progress_pct": progress_pct,
                            },
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

                    # Определяем, есть ли новые данные для обновления файлов
                    has_new_data = len(processed_summaries) > 0
                    
                    self.markdown_renderer.render_chat_summary(
                        chat_name,
                        all_summaries,
                        top_sessions=top_sessions,
                        force=self.force,
                        has_new_data=has_new_data,
                    )
                    # Создаём файл накапливающегося контекста
                    self.markdown_renderer.render_cumulative_context(
                        chat_name, all_summaries, force=self.force, has_new_data=has_new_data
                    )
                    # Создаём индекс сессий чата
                    self.markdown_renderer.render_chat_index(
                        chat_name, all_summaries, force=self.force, has_new_data=has_new_data
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
                
                # Callback: завершение обработки чата
                self._call_progress_callback(
                    job_id,
                    "chat_completed",
                    {
                        "chat": chat_name,
                        "chat_index": chat_idx,
                        "total_chats": total_chats,
                        "stats": {
                            "sessions_indexed": len(processed_summaries),
                            "messages_indexed": processed_messages_count,
                            "tasks_indexed": stats.get("tasks_indexed", 0),
                        },
                    },
                )

                # Сохраняем словари сущностей если включено обучение
                if self.entity_dictionary and self.enable_entity_learning:
                    try:
                        # Обрабатываем очередь валидации перед сохранением
                        self.entity_dictionary.flush_validation_queue()
                        self.entity_dictionary.save_dictionaries()
                        logger.info(f"Словари сущностей сохранены для чата {chat_name}")
                        
                        # Обновляем EntityNode в графе с описаниями
                        if self.graph:
                            await self._update_entity_nodes_with_descriptions()
                        
                        # Строим и индексируем профили сущностей
                        await self._build_and_index_entities(chat_name)
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
                # Callback: ошибка при обработке чата
                self._call_progress_callback(
                    job_id,
                    "error",
                    {
                        "chat": chat_dir.name,
                        "error": str(e),
                    },
                )
                continue

        logger.info(f"Индексация завершена: {stats}")
        
        # Callback: завершение всей индексации
        self._call_progress_callback(
            job_id,
            "completed",
            {
                "stats": stats,
            },
        )
        
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

        # Добавляем в коллекцию Qdrant
        if self.qdrant_manager and self.sessions_collection:
            try:
                self.qdrant_manager.upsert(
                    collection_name=self.sessions_collection,
                    ids=[session_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[embedding_text],
                )
            except Exception as e:
                logger.error(f"Ошибка при добавлении сессии {session_id} в Qdrant: {e}")
        else:
            logger.warning("Qdrant недоступен, сессия не будет сохранена в векторное хранилище")
        
        # Qdrant используется для векторного хранилища (см. код выше)
        try:
            pass  # Заглушка для сохранения структуры try-except
        except Exception as e:
            error_msg = str(e)
            if "embedding with dimension" in error_msg or "dimension" in error_msg.lower():
                logger.warning(
                    f"Ошибка размерности эмбеддингов в коллекции chat_sessions: {error_msg}. "
                    "Пересоздаём коллекцию..."
                )
                # Пересоздаём коллекцию
                if self.qdrant_manager and self.sessions_collection:
                    self.qdrant_manager.delete_collection(self.sessions_collection)
                    self.sessions_collection = self._check_and_recreate_collection(
                        "chat_sessions",
                        "Саммаризации сессий для векторного поиска (L1)",
                        force_recreate=True
                    )
                    if self.sessions_collection:
                        self.qdrant_manager.upsert(
                            collection_name=self.sessions_collection,
                            ids=[session_id],
                            embeddings=[embedding],
                            metadatas=[metadata],
                            documents=[embedding_text],
                        )
                        logger.info("Коллекция chat_sessions пересоздана и запись добавлена")
            else:
                raise

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
                
                # Создаём теги для сессии
                tags = []
                chat_name = meta.get("chat_name", "")
                if chat_name:
                    # Добавляем тег на основе имени чата
                    chat_tag = chat_name.lower().replace(" ", "_")
                    tags.append(chat_tag)
                
                # Создаём MemoryRecord для сессии
                record = MemoryRecord(
                    record_id=session_id,
                    source=meta.get("chat_name", "unknown"),
                    content=embedding_text,
                    timestamp=timestamp,
                    author=None,  # Сессии не имеют автора
                    tags=tags,  # Применяем теги
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
                
                # Сохраняем эмбеддинг в граф и Qdrant
                if embedding:
                    try:
                        # Сохраняем в граф
                        self.graph.update_node(
                            session_id,
                            embedding=embedding,
                        )
                        
                        # Сохраняем в Qdrant для векторного поиска (коллекция memory-records)
                        # ВАЖНО: Это основная коллекция для поиска, должна содержать все записи, включая сессии
                        if self.vector_store and self.vector_store.available():
                            try:
                                payload_data = {
                                    "record_id": session_id,
                                    "source": meta.get("chat_name", ""),
                                    "tags": tags,  # Сохраняем теги в Qdrant
                                    "timestamp": start_time_utc.timestamp() if start_time_utc else 0,
                                    "timestamp_iso": start_time_utc.isoformat() if start_time_utc else "",
                                    "content_preview": summary.get("context", "")[:200],
                                    "session_type": "session_summary",
                                    "chat": meta.get("chat_name", ""),
                                }
                                chat_name = meta.get("chat_name")
                                if isinstance(chat_name, str):
                                    payload_data["chat"] = chat_name
                                
                                self.vector_store.upsert(session_id, embedding, payload_data)
                                logger.debug(f"Эмбеддинг сессии сохранен в Qdrant (memory-records) для {session_id}")
                            except Exception as e:
                                logger.warning(f"Ошибка при сохранении эмбеддинга сессии {session_id} в Qdrant (memory-records): {e}")
                    except Exception as e:
                        logger.debug(f"Ошибка при сохранении эмбеддинга сессии {session_id}: {e}")
                
                # Создаем связи с предыдущими сессиями того же чата
                # Получаем имя чата из meta, с fallback на chat_id из summary
                chat = meta.get("chat_name") or summary.get("chat_id") or ""
                self._link_session_to_previous_sessions(session_id, chat, start_time_utc)
                
                logger.debug(f"Синхронизирована сессия {session_id} с графом памяти")
            except Exception as e:
                logger.warning(f"Ошибка при синхронизации сессии {session_id} с графом: {e}")

        logger.info(f"L1: Проиндексирована сессия {session_id}")

    def _format_group_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Форматирует группу сообщений в единый текст для создания эмбеддинга.

        Args:
            messages: Список сообщений группы
            max_tokens: Максимальное количество токенов (опционально)

        Returns:
            Отформатированный текст группы
        """
        if not messages:
            return ""

        parts = []
        total_chars = 0
        max_chars = (max_tokens * 4) if max_tokens else None  # ~4 символа на токен

        for i, msg in enumerate(messages, 1):
            msg_text = msg.get("text", "").strip()
            if not msg_text:
                continue

            # Извлекаем автора
            author = "Unknown"
            from_field = msg.get("from") or {}
            if isinstance(from_field, dict):
                author = from_field.get("display") or from_field.get("username") or from_field.get("id") or "Unknown"
            elif isinstance(from_field, str):
                author = from_field

            # Форматируем сообщение
            formatted_msg = f"[MSG_{i}] {author}: {msg_text}"

            # Проверяем лимит токенов
            if max_chars and total_chars + len(formatted_msg) > max_chars:
                # Обрезаем последнее сообщение, если нужно
                remaining = max_chars - total_chars - len(f"[MSG_{i}] {author}: ")
                if remaining > 20:  # Минимум 20 символов
                    formatted_msg = f"[MSG_{i}] {author}: {msg_text[:remaining]}..."
                else:
                    break

            parts.append(formatted_msg)
            total_chars += len(formatted_msg)

        return "\n".join(parts)

    async def _group_messages_for_embedding(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        chat_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Группирует сообщения по смыслу для создания эмбеддингов.

        Args:
            messages: Список сообщений для группировки
            session_id: ID сессии
            chat_name: Название чата

        Returns:
            Список групп сообщений с метаданными
        """
        if not self.enable_message_grouping:
            # Если группировка отключена, возвращаем каждое сообщение как отдельную группу
            return [
                {
                    "group_id": f"{session_id}-M{i+1:04d}",
                    "messages": [msg],
                    "message_ids": [msg.get("id") or msg.get("message_id") or f"msg_{i}"],
                    "strategy": "none",
                }
                for i, msg in enumerate(messages)
            ]

        strategy = self.message_grouping_strategy
        groups = []

        if strategy == "session":
            # Стратегия "session": используем всю сессию как одну группу (если размер подходит)
            if len(messages) >= self.min_group_size and len(messages) <= self.max_group_size:
                # Вся сессия как одна группа
                message_ids = [
                    msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    for i, msg in enumerate(messages)
                ]
                groups.append({
                    "group_id": f"{session_id}-G001",
                    "messages": messages,
                    "message_ids": message_ids,
                    "strategy": "session",
                })
            else:
                # Разбиваем на подгруппы по max_group_size
                for i in range(0, len(messages), self.max_group_size):
                    group_messages = messages[i:i + self.max_group_size]
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id") or msg.get("message_id") or f"msg_{i+j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append({
                            "group_id": f"{session_id}-G{i//self.max_group_size + 1:03d}",
                            "messages": group_messages,
                            "message_ids": message_ids,
                            "strategy": "session",
                        })

        elif strategy == "semantic" and self.semantic_regrouper:
            # Стратегия "semantic": семантическая перегруппировка через LLM
            try:
                # Создаем временные сессии из сообщений для перегруппировки
                temp_sessions = [{
                    "session_id": f"{session_id}-temp",
                    "messages": messages,
                    "chat": chat_name,
                }]
                
                regrouped_sessions = await self.semantic_regrouper.regroup_sessions(
                    temp_sessions, chat_name
                )
                
                # Преобразуем перегруппированные сессии в группы
                for i, regrouped_session in enumerate(regrouped_sessions, 1):
                    group_messages = regrouped_session.get("messages", [])
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id") or msg.get("message_id") or f"msg_{j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append({
                            "group_id": f"{session_id}-G{i:03d}",
                            "messages": group_messages,
                            "message_ids": message_ids,
                            "strategy": "semantic",
                            "theme": regrouped_session.get("theme"),
                            "rationale": regrouped_session.get("regroup_rationale"),
                        })
            except Exception as e:
                logger.warning(f"Ошибка семантической перегруппировки, используем стратегию session: {e}")
                # Fallback на стратегию session
                if len(messages) >= self.min_group_size:
                    message_ids = [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                        for i, msg in enumerate(messages)
                    ]
                    groups.append({
                        "group_id": f"{session_id}-G001",
                        "messages": messages,
                        "message_ids": message_ids,
                        "strategy": "session",
                    })

        elif strategy == "adaptive" and self.adaptive_grouper:
            # Стратегия "adaptive": адаптивная группировка с учетом размера контекста
            try:
                message_groups = self.adaptive_grouper.group_messages_adaptively(
                    messages, chat_name
                )
                
                for i, group_messages in enumerate(message_groups, 1):
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id") or msg.get("message_id") or f"msg_{j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append({
                            "group_id": f"{session_id}-G{i:03d}",
                            "messages": group_messages,
                            "message_ids": message_ids,
                            "strategy": "adaptive",
                        })
            except Exception as e:
                logger.warning(f"Ошибка адаптивной группировки, используем стратегию session: {e}")
                # Fallback на стратегию session
                if len(messages) >= self.min_group_size:
                    message_ids = [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                        for i, msg in enumerate(messages)
                    ]
                    groups.append({
                        "group_id": f"{session_id}-G001",
                        "messages": messages,
                        "message_ids": message_ids,
                        "strategy": "session",
                    })

        else:
            # Неизвестная стратегия или компонент недоступен - используем session
            logger.warning(f"Стратегия {strategy} недоступна, используем session")
            if len(messages) >= self.min_group_size:
                message_ids = [
                    msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    for i, msg in enumerate(messages)
                ]
                groups.append({
                    "group_id": f"{session_id}-G001",
                    "messages": messages,
                    "message_ids": message_ids,
                    "strategy": "session",
                })

        # Если не получилось создать группы, возвращаем каждое сообщение отдельно
        if not groups:
            logger.warning(f"Не удалось создать группы для сессии {session_id}, используем отдельные сообщения")
            return [
                {
                    "group_id": f"{session_id}-M{i+1:04d}",
                    "messages": [msg],
                    "message_ids": [msg.get("id") or msg.get("message_id") or f"msg_{i}"],
                    "strategy": "none",
                }
                for i, msg in enumerate(messages)
            ]

        logger.info(
            f"Создано {len(groups)} групп из {len(messages)} сообщений "
            f"для сессии {session_id} (стратегия: {strategy})"
        )
        return groups

    async def _index_messages_l2_grouped(
        self,
        session: Dict[str, Any],
        messages: List[Dict[str, Any]],
        session_id: str,
        chat: str,
        chat_mode: str,
    ) -> int:
        """
        Индексация сообщений на уровне L2 с группировкой по смыслу.

        Args:
            session: Сессия
            messages: Список сообщений
            session_id: ID сессии
            chat: Название чата
            chat_mode: Тип чата (channel/group)

        Returns:
            Количество проиндексированных сообщений
        """
        indexed_count = 0
        skipped_duplicates_count = 0

        # Группируем сообщения
        groups = await self._group_messages_for_embedding(messages, session_id, chat)

        groups_to_index = []

        for group in groups:
            try:
                group_id = group["group_id"]
                group_messages = group["messages"]
                message_ids = group["message_ids"]
                strategy = group.get("strategy", "session")

                # Проверяем дубликаты группы
                skipped_duplicate = False

                # Проверяем, существует ли уже эта группа в базе
                if not self.force and self.qdrant_manager and self.messages_collection:
                    try:
                        existing_group = self.qdrant_manager.get(
                            collection_name=self.messages_collection,
                            ids=[group_id]
                        )
                        if existing_group and existing_group.get("ids"):
                            logger.debug(f"Группа {group_id} уже существует, пропускаем")
                            skipped_duplicate = True
                    except Exception as e:
                        logger.debug(f"Не удалось проверить дубликаты группы: {e}")

                if skipped_duplicate:
                    skipped_duplicates_count += len(group_messages)
                    continue

                # Форматируем текст группы
                group_text = self._format_group_text(
                    group_messages,
                    max_tokens=self.max_group_tokens
                )

                if not group_text or len(group_text.strip()) < 10:
                    logger.warning(f"Группа {group_id} пуста, пропускаем")
                    continue

                # Валидируем текст перед отправкой в эмбеддинг
                group_text, replaced_urls = validate_embedding_text(group_text)

                # Сохраняем данные для батчевой обработки
                groups_to_index.append({
                    "group_id": group_id,
                    "group_text": group_text,
                    "group_messages": group_messages,
                    "message_ids": message_ids,
                    "strategy": strategy,
                    "message_count": len(group_messages),
                    "metadata": {
                        "group_id": group_id,
                        "message_ids": message_ids,
                        "message_count": len(group_messages),
                        "group_strategy": strategy,
                        "is_group_embedding": True,
                        "session_id": session_id,
                        "chat": chat,
                        "chat_mode": chat_mode,
                        "replaced_urls": ",".join(replaced_urls) if replaced_urls else "",
                        "theme": group.get("theme"),
                        "rationale": group.get("rationale"),
                    }
                })

            except Exception as e:
                logger.error(
                    f"Ошибка при подготовке группы {group.get('group_id', 'unknown')} "
                    f"в сессии {session_id} для индексации: {e}"
                )
                continue

        # Генерируем эмбеддинги батчами
        if groups_to_index:
            try:
                async with self.embedding_client:
                    # Собираем тексты для батча
                    batch_texts = [group["group_text"] for group in groups_to_index]

                    # Генерируем эмбеддинги батчем
                    embeddings = await self.embedding_client.generate_embeddings(batch_texts, batch_size=32)

                    # Добавляем в коллекцию батчем
                    ids = [group["group_id"] for group in groups_to_index]
                    documents = [group["group_text"] for group in groups_to_index]
                    metadatas = [group["metadata"] for group in groups_to_index]

                    # Сохраняем в Qdrant коллекцию chat_messages
                    if self.qdrant_manager and self.messages_collection:
                        try:
                            self.qdrant_manager.upsert(
                                collection_name=self.messages_collection,
                                ids=ids,
                                embeddings=embeddings,
                                metadatas=metadatas,
                                documents=documents,
                            )
                            logger.debug(f"Сохранено {len(ids)} групп в Qdrant коллекцию {self.messages_collection}")
                        except Exception as e:
                            error_msg = str(e)
                            if "dimension" in error_msg.lower():
                                logger.warning(
                                    f"Ошибка размерности эмбеддингов в коллекции chat_messages: {error_msg}. "
                                    "Пересоздаём коллекцию..."
                                )
                                if self.qdrant_manager:
                                    self.qdrant_manager.delete_collection(self.messages_collection)
                                    self.messages_collection = self._check_and_recreate_collection(
                                        "chat_messages",
                                        "Сообщения с контекстом для уточняющего поиска (L2)",
                                        force_recreate=True
                                    )
                                    if self.messages_collection:
                                        self.qdrant_manager.upsert(
                                            collection_name=self.messages_collection,
                                            ids=ids,
                                            embeddings=embeddings,
                                            metadatas=metadatas,
                                            documents=documents,
                                        )
                                        logger.info("Коллекция chat_messages пересоздана и записи добавлены")
                            else:
                                logger.error(f"Ошибка при добавлении групп в Qdrant: {e}")
                    else:
                        logger.warning("Qdrant недоступен, группы не будут сохранены в векторное хранилище")

                    # Синхронизация с графом памяти
                    if self.ingestor and self.graph:
                        logger.info(f"Синхронизация с графом: обрабатываем {len(groups_to_index)} групп")
                        from ..indexing import MemoryRecord, Attachment
                        from ..utils.datetime_utils import parse_datetime_utc

                        records_to_ingest = []
                        for idx, group_data in enumerate(groups_to_index):
                            try:
                                group_id = group_data["group_id"]
                                group_text = group_data["group_text"]
                                metadata = group_data["metadata"]
                                embedding = embeddings[idx] if idx < len(embeddings) else None

                                # Используем время первого сообщения группы
                                first_msg = group_data["group_messages"][0]
                                date_utc = first_msg.get("date_utc") or first_msg.get("date", "")
                                timestamp = parse_datetime_utc(date_utc, default=None) if date_utc else None
                                if not timestamp:
                                    from datetime import datetime, timezone
                                    timestamp = datetime.now(timezone.utc)

                                # Создаем запись для группы
                                record = MemoryRecord(
                                    record_id=group_id,
                                    source=chat,
                                    content=group_text,
                                    timestamp=timestamp,
                                    author=None,  # Группа может содержать сообщения от разных авторов
                                    tags=[],
                                    entities=[],
                                    attachments=[],
                                    metadata=metadata,
                                )

                                if embedding is not None and len(embedding) > 0:
                                    records_to_ingest.append((record, embedding))

                            except Exception as e:
                                logger.error(f"Ошибка при подготовке записи группы {group_data.get('group_id', 'unknown')}: {e}")
                                continue

                        # Инжестим записи в граф
                        if records_to_ingest:
                            try:
                                for record, embedding in records_to_ingest:
                                    try:
                                        if hasattr(embedding, 'tolist'):
                                            embedding = embedding.tolist()
                                        elif not isinstance(embedding, list):
                                            embedding = list(embedding)

                                        self.ingestor.ingest([record], embeddings=[embedding])

                                        # Также сохраняем в vector_store для поиска
                                        if self.vector_store and self.vector_store.available():
                                            payload_data = {
                                                "record_id": record.record_id,
                                                "source": record.source,
                                                "content": record.content,
                                                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                                                "tags": record.tags,
                                                "entities": record.entities,
                                                "metadata": record.metadata,
                                            }
                                            self.vector_store.upsert(record.record_id, embedding, payload_data)

                                    except Exception as e:
                                        logger.error(f"Ошибка при индексации группы {record.record_id}: {e}")
                                        continue

                                logger.info(
                                    f"Синхронизировано {len(records_to_ingest)} групп с графом памяти"
                                )
                            except Exception as e:
                                logger.error(f"Ошибка при синхронизации групп с графом: {e}")

                    indexed_count = sum(group["message_count"] for group in groups_to_index)

            except Exception as e:
                logger.error(
                    f"Ошибка при индексации групп в сессии {session_id}: {e}"
                )

        logger.info(
            f"L2 (группировка): Проиндексировано {indexed_count} сообщений "
            f"в {len(groups_to_index)} группах из сессии {session_id} "
            f"(пропущено дубликатов: {skipped_duplicates_count})"
        )
        return indexed_count

    async def _index_messages_l2(self, session: Dict[str, Any]) -> int:
        """
        Индексация сообщений на уровне L2 (с симметричным контекстом или группировкой)

        Args:
            session: Сессия

        Returns:
            Количество проиндексированных сообщений
        """
        messages = session["messages"]
        session_id = session["session_id"]
        chat = session["chat"]

        indexed_count = 0
        skipped_duplicates_count = 0
        processed_count = 0

        # Определим тип чата для подстройки контекста
        chat_mode = self._detect_chat_mode(messages)

        # Если включена группировка, используем групповые эмбеддинги
        if self.enable_message_grouping:
            return await self._index_messages_l2_grouped(session, messages, session_id, chat, chat_mode)

        # Иначе используем старую логику (отдельные сообщения с контекстом)
        messages_to_index = []
        queued_count = 0

        for i, msg in enumerate(messages):
            try:
                processed_count += 1
                msg_text = msg.get("text", "").strip()
                if not msg_text or len(msg_text) < 10:
                    continue

                # Используем уникальный ID сообщения из Telegram для предотвращения дублирования
                # Приоритет: id -> message_id -> fallback на session-based ID
                telegram_msg_id = msg.get("id") or msg.get("message_id")
                if telegram_msg_id:
                    # Используем формат: chat:telegram_msg_id для уникальности
                    msg_id = f"{chat}:{telegram_msg_id}"
                else:
                    # Fallback: используем session-based ID, но с хешем текста для уникальности
                    import hashlib
                    text_hash = hashlib.md5(msg_text.encode("utf-8")).hexdigest()[:8]
                    msg_id = f"{session_id}-M{i+1:04d}-{text_hash}"
                
                # Проверяем, существует ли уже это сообщение в базе
                # ВАЖНО: даже при force_full проверяем дубликаты по telegram_id, чтобы предотвратить
                # дублирование одного сообщения в разных сессиях
                skipped_duplicate = False
                
                # Всегда проверяем дубликаты по telegram_id (даже при force_full)
                if telegram_msg_id:
                    # 1. Проверяем в графе памяти по точному ID
                    if self.graph:
                        try:
                            cursor = self.graph.conn.cursor()
                            cursor.execute("SELECT id FROM nodes WHERE id = ? LIMIT 1", (msg_id,))
                            if cursor.fetchone():
                                logger.debug(f"Дубликат сообщения найден в графе по msg_id={msg_id}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в графе: {e}")
                    
                    # 2. Проверяем в графе по telegram_id глобально (в любом чате)
                    if not skipped_duplicate and self.graph:
                        try:
                            cursor = self.graph.conn.cursor()
                            cursor.execute("""
                                SELECT id FROM nodes 
                                WHERE type = 'DocChunk' 
                                AND properties IS NOT NULL
                                AND json_extract(properties, '$.telegram_id') = ?
                                LIMIT 1
                            """, (str(telegram_msg_id),))
                            if cursor.fetchone():
                                logger.debug(f"Дубликат сообщения найден в графе по telegram_id={telegram_msg_id}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в графе по telegram_id: {e}")
                    
                    # 3. Проверяем по telegram ID в Qdrant (коллекция chat_messages)
                    if not skipped_duplicate and self.qdrant_manager and self.messages_collection:
                        try:
                            existing_by_id = self.qdrant_manager.get(
                                collection_name=self.messages_collection,
                                where={"$or": [
                                    {"msg_id": {"$eq": f"{chat}:{telegram_msg_id}"}},
                                    {"telegram_id": {"$eq": str(telegram_msg_id)}},
                                ]},
                                limit=1
                            )
                            if existing_by_id and existing_by_id.get("ids"):
                                logger.debug(f"Дубликат сообщения найден в chat_messages по telegram_id={telegram_msg_id}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в chat_messages по telegram_id: {e}")
                    
                    # 4. Проверяем в основной коллекции memory-records (для поиска)
                    if not skipped_duplicate and self.vector_store and self.vector_store.available():
                        try:
                            # Проверяем, существует ли запись с таким telegram_id в payload
                            # Используем поиск по фильтру (если поддерживается) или проверяем через get
                            # Пока используем простую проверку по record_id (msg_id)
                            # TODO: Добавить проверку по telegram_id в payload, если Qdrant поддерживает фильтрацию
                            pass  # Пока пропускаем, так как vector_store не имеет метода get с фильтром
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в memory-records: {e}")
                
                # Дополнительная проверка по content_hash (даже при force_full)
                if not skipped_duplicate:
                    import hashlib
                    content_hash = hashlib.md5(msg_text.encode("utf-8")).hexdigest()[:16]
                    
                    # Проверяем в графе памяти по content_hash в метаданных
                    if self.graph:
                        try:
                            cursor = self.graph.conn.cursor()
                            cursor.execute("""
                                SELECT id FROM nodes 
                                WHERE type = 'DocChunk' 
                                AND properties IS NOT NULL
                                AND json_extract(properties, '$.content_hash') = ?
                                AND (
                                    json_extract(properties, '$.chat') = ?
                                    OR json_extract(properties, '$.source') = ?
                                )
                                LIMIT 1
                            """, (content_hash, chat, chat))
                            if cursor.fetchone():
                                logger.debug(f"Дубликат сообщения найден в графе по content_hash={content_hash} в чате {chat}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в графе по content_hash: {e}")
                    
                    # Проверяем в Qdrant
                    if not skipped_duplicate and self.qdrant_manager and self.messages_collection:
                        try:
                            existing_by_hash = self.qdrant_manager.get(
                                collection_name=self.messages_collection,
                                where={"content_hash": {"$eq": content_hash}},
                                limit=1
                            )
                            if existing_by_hash and existing_by_hash.get("ids"):
                                # Проверяем, что это действительно то же сообщение (из того же чата)
                                existing_metadata = existing_by_hash.get("metadatas", [])
                                if existing_metadata and len(existing_metadata) > 0:
                                    existing_chat = existing_metadata[0].get("chat", "")
                                    if existing_chat == chat:
                                        logger.debug(f"Дубликат сообщения найден по content_hash={content_hash} в чате {chat}, пропускаем")
                                        skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты по content_hash: {e}")
                
                # Дополнительная проверка по точному ID (только если не force_full)
                # ВАЖНО: даже при force_full проверяем дубликаты по telegram_id и content_hash
                # чтобы предотвратить дублирование одного сообщения в разных сессиях
                if not self.force and not skipped_duplicate and self.qdrant_manager and self.messages_collection:
                    # Сначала проверяем по точному ID
                    existing_msg = self.qdrant_manager.get(
                        collection_name=self.messages_collection,
                        ids=[msg_id]
                    )
                    if existing_msg and existing_msg.get("ids"):
                        # Сообщение уже существует, пропускаем
                        logger.debug(f"Сообщение {msg_id} уже существует, пропускаем")
                        skipped_duplicate = True
                
                # Глобальная проверка дубликатов по telegram_id и content_hash (всегда, даже при force_full)
                # Это предотвращает дублирование, когда одно сообщение попадает в разные сессии
                if not skipped_duplicate:
                    import hashlib
                    content_hash = hashlib.md5(msg_text.encode("utf-8")).hexdigest()[:16]
                    
                    # Проверяем в графе по content_hash глобально (в любом чате)
                    if self.graph:
                        try:
                            cursor = self.graph.conn.cursor()
                            cursor.execute("""
                                SELECT id FROM nodes 
                                WHERE type = 'DocChunk' 
                                AND properties IS NOT NULL
                                AND json_extract(properties, '$.content_hash') = ?
                                LIMIT 1
                            """, (content_hash,))
                            if cursor.fetchone():
                                logger.debug(f"Дубликат сообщения найден в графе по content_hash={content_hash}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            logger.debug(f"Не удалось проверить дубликаты в графе по content_hash: {e}")
                    
                    # Проверяем в Qdrant коллекции chat_messages (глобально по чату)
                    if not skipped_duplicate and self.qdrant_manager and self.messages_collection:
                        try:
                            if telegram_msg_id:
                                where_conditions = {
                                    "$and": [
                                        {"chat": {"$eq": chat}},
                                        {"$or": [
                                            {"telegram_id": {"$eq": str(telegram_msg_id)}},
                                            {"content_hash": {"$eq": content_hash}}
                                        ]}
                                    ]
                                }
                            else:
                                where_conditions = {
                                    "$and": [
                                        {"chat": {"$eq": chat}},
                                        {"content_hash": {"$eq": content_hash}}
                                    ]
                                }
                            
                            existing_by_id = self.qdrant_manager.get(
                                collection_name=self.messages_collection,
                                where=where_conditions,
                                limit=1
                            )
                            if existing_by_id and existing_by_id.get("ids"):
                                logger.debug(f"Дубликат сообщения найден в chat_messages по telegram_id={telegram_msg_id} или content_hash={content_hash} в чате {chat}, пропускаем")
                                skipped_duplicate = True
                        except Exception as e:
                            # Если поиск по метаданным не работает, продолжаем
                            logger.debug(f"Не удалось проверить дубликаты в chat_messages: {e}")
                
                if skipped_duplicate:
                    skipped_duplicates_count += 1
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
                max_tokens = 131072  # Для gpt-oss-20b (максимальный лимит)

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

                # Вычисляем content_hash для предотвращения дублирования
                import hashlib
                content_hash = hashlib.md5(msg_text.encode("utf-8")).hexdigest()[:16]
                
                # Сохраняем данные для батчевой обработки
                queued_count += 1
                messages_to_index.append({
                    "msg_id": msg_id,
                    "msg_text": msg_text,
                    "embedding_text": embedding_text,
                    "msg_index": i,  # Сохраняем индекс для извлечения автора
                    "msg": msg,  # Сохраняем исходное сообщение для извлечения автора
                    "telegram_id": str(telegram_msg_id) if telegram_msg_id else None,
                    "content_hash": content_hash,
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
                        "telegram_id": str(telegram_msg_id) if telegram_msg_id else None,
                        "content_hash": content_hash,
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
                    
                    # Сохраняем в Qdrant коллекцию chat_messages (для инкрементальной индексации и других целей)
                    # ВАЖНО: Основная коллекция для поиска - memory-records, она заполняется через vector_store.upsert
                    if self.qdrant_manager and self.messages_collection:
                        try:
                            self.qdrant_manager.upsert(
                                collection_name=self.messages_collection,
                                ids=ids,
                                embeddings=embeddings,
                                metadatas=metadatas,
                                documents=documents,
                            )
                            logger.debug(f"Сохранено {len(ids)} сообщений в Qdrant коллекцию {self.messages_collection}")
                        except Exception as e:
                            error_msg = str(e)
                            if "dimension" in error_msg.lower():
                                logger.warning(
                                    f"Ошибка размерности эмбеддингов в коллекции chat_messages: {error_msg}. "
                                    "Пересоздаём коллекцию..."
                                )
                                # Пересоздаём коллекцию
                                if self.qdrant_manager:
                                    self.qdrant_manager.delete_collection(self.messages_collection)
                                    self.messages_collection = self._check_and_recreate_collection(
                                        "chat_messages",
                                        "Сообщения с контекстом для уточняющего поиска (L2)",
                                        force_recreate=True
                                    )
                                    if self.messages_collection:
                                        self.qdrant_manager.upsert(
                                            collection_name=self.messages_collection,
                                            ids=ids,
                                            embeddings=embeddings,
                                            metadatas=metadatas,
                                            documents=documents,
                                        )
                                        logger.info("Коллекция chat_messages пересоздана и записи добавлены")
                            else:
                                logger.error(f"Ошибка при добавлении сообщений в Qdrant: {e}")
                    else:
                        logger.warning("Qdrant недоступен, сообщения не будут сохранены в векторное хранилище")
                    
                    # Синхронизация с графом памяти
                    logger.debug(f"Проверка синхронизации с графом: ingestor={self.ingestor is not None}, graph={self.graph is not None}")
                    if self.ingestor and self.graph:
                        logger.info(f"Синхронизация с графом: обрабатываем {len(messages_to_index)} сообщений")
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
                                
                                # Извлекаем теги
                                tags = []
                                chat_name = metadata.get("chat", "")
                                if chat_name:
                                    tags.append(chat_name.lower().replace(" ", "_"))
                                msg_tags = msg_obj.get("tags", []) if msg_obj else []
                                if not msg_tags:
                                    msg_tags = metadata.get("tags", [])
                                if isinstance(msg_tags, list):
                                    tags.extend([str(t).lower() for t in msg_tags if t])
                                tags = list(dict.fromkeys(tags))  # Убираем дубликаты, сохраняя порядок
                                
                                # Извлекаем сущности
                                entities = []
                                if self.entity_extractor:
                                    try:
                                        extracted = self.entity_extractor.extract_entities(msg_text)
                                        if extracted:
                                            entities.extend([
                                                e.get("text") or e.get("value") 
                                                for e in extracted 
                                                if e.get("text") or e.get("value")
                                            ])
                                    except Exception as e:
                                        logger.debug(f"Ошибка извлечения сущностей для сообщения {msg_id}: {e}")
                                metadata_entities = metadata.get("entities", [])
                                if isinstance(metadata_entities, list):
                                    entities.extend(metadata_entities)
                                entities = list(dict.fromkeys([str(e) for e in entities if e]))  # Убираем дубликаты
                                
                                # Создаём MemoryRecord
                                record = MemoryRecord(
                                    record_id=msg_id,
                                    source=metadata.get("chat", "unknown"),
                                    content=msg_text,
                                    timestamp=timestamp,
                                    author=author,
                                    tags=tags,
                                    entities=entities,
                                    attachments=[],
                                    metadata={
                                        "chat": metadata.get("chat", ""),
                                        "session_id": metadata.get("session_id", ""),
                                        "has_context": metadata.get("has_context", False),
                                        "context_length": metadata.get("context_length", 0),
                                        "chat_mode": metadata.get("chat_mode", "group"),
                                        "date_utc": date_utc,
                                        "content_hash": metadata.get("content_hash", ""),  # Сохраняем для проверки дубликатов
                                        "telegram_id": metadata.get("telegram_id"),  # Сохраняем для проверки дубликатов
                                    },
                                )
                                records_to_ingest.append((record, embedding))
                            except Exception as e:
                                logger.warning(f"Ошибка при подготовке записи {msg_data.get('msg_id', 'unknown')} для графа: {e}")
                                continue
                        
                        # Сохраняем записи в граф батчем
                        logger.debug(f"Подготовлено {len(records_to_ingest)} записей для сохранения в граф")
                        if records_to_ingest:
                            try:
                                records_only = [r for r, _ in records_to_ingest]
                                logger.info(f"Сохранение {len(records_only)} записей в граф через ingestor.ingest()")
                                ingest_result = self.ingestor.ingest(records_only)
                                logger.info(f"Результат ingest: records_ingested={ingest_result.records_ingested}, attachments_ingested={ingest_result.attachments_ingested}")
                                
                                if ingest_result.records_ingested == 0:
                                    logger.warning(
                                        f"⚠️ Внимание: 0 записей добавлено в граф для сессии {session_id}. "
                                        f"Возможно, все записи уже существуют или произошла ошибка."
                                    )
                                
                                # Сохраняем эмбеддинги в граф
                                embeddings_saved = 0
                                embeddings_failed = 0
                                for record, embedding in records_to_ingest:
                                    # Проверяем, что эмбеддинг существует и не пустой
                                    if embedding is not None and len(embedding) > 0:
                                        try:
                                            # Преобразуем numpy массив в список, если нужно
                                            if hasattr(embedding, 'tolist'):
                                                embedding = embedding.tolist()
                                            elif not isinstance(embedding, list):
                                                embedding = list(embedding)
                                            
                                            # Проверяем, что узел существует в графе
                                            if record.record_id not in self.graph.graph:
                                                logger.warning(
                                                    f"Узел {record.record_id} не найден в графе, "
                                                    f"нельзя обновить эмбеддинг"
                                                )
                                                embeddings_failed += 1
                                                continue
                                            
                                            # Сохраняем эмбеддинг в граф
                                            success = self.graph.update_node(
                                                record.record_id,
                                                embedding=embedding,
                                            )
                                            if success:
                                                embeddings_saved += 1
                                                logger.debug(
                                                    f"Эмбеддинг сохранен в граф для {record.record_id}: "
                                                    f"размер={len(embedding)}"
                                                )
                                                
                                                # Сохраняем эмбеддинг в Qdrant для векторного поиска (коллекция memory-records)
                                                # ВАЖНО: Это основная коллекция для поиска, должна содержать все записи
                                                if self.vector_store and self.vector_store.available():
                                                    try:
                                                        payload_data = {
                                                            "record_id": record.record_id,
                                                            "source": record.source,
                                                            "tags": record.tags,
                                                            "timestamp": record.timestamp.timestamp(),
                                                            "timestamp_iso": record.timestamp.isoformat(),
                                                            "content_preview": record.content[:200],
                                                            "chat": record.metadata.get("chat", ""),
                                                            "session_id": record.metadata.get("session_id", ""),
                                                            "telegram_id": record.metadata.get("telegram_id"),
                                                            "content_hash": record.metadata.get("content_hash", ""),
                                                        }
                                                        chat_name = record.metadata.get("chat")
                                                        if isinstance(chat_name, str):
                                                            payload_data["chat"] = chat_name
                                                        
                                                        self.vector_store.upsert(record.record_id, embedding, payload_data)
                                                        logger.debug(f"Эмбеддинг сохранен в Qdrant (memory-records) для {record.record_id}")
                                                    except Exception as e:
                                                        logger.warning(
                                                            f"Не удалось сохранить эмбеддинг в Qdrant (memory-records) для {record.record_id}: {e}"
                                                        )
                                            else:
                                                embeddings_failed += 1
                                                logger.warning(
                                                    f"Не удалось сохранить эмбеддинг в граф для {record.record_id}"
                                                )
                                        except Exception as e:
                                            embeddings_failed += 1
                                            logger.warning(
                                                f"Ошибка при сохранении эмбеддинга для {record.record_id}: {e}",
                                                exc_info=True
                                            )
                                    else:
                                        logger.debug(f"Эмбеддинг отсутствует или пустой для {record.record_id}")
                                
                                logger.info(
                                    f"Эмбеддинги сохранены: {embeddings_saved}, "
                                    f"ошибок: {embeddings_failed} из {len(records_to_ingest)} записей"
                                )
                                
                                logger.debug(f"Синхронизировано {len(records_to_ingest)} записей с графом памяти")
                            except Exception as e:
                                logger.warning(f"Ошибка при синхронизации записей с графом: {e}")
                    
                    indexed_count += len(messages_to_index)
            except Exception as e:
                logger.error(
                    f"Ошибка при индексации сообщений в сессии {session_id}: {e}"
                )

        logger.info(
            f"L2: Проиндексировано {indexed_count} сообщений из сессии {session_id} "
            f"(обработано: {processed_count}, пропущено дубликатов: {skipped_duplicates_count}, "
            f"добавлено в очередь: {queued_count})"
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
                if self.qdrant_manager and self.tasks_collection:
                    self.qdrant_manager.upsert(
                        collection_name=self.tasks_collection,
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

        # Получаем эмбеддинги из Qdrant для сессий
        session_ids = [s["session_id"] for s in summaries]

        try:
            result = None
            if self.qdrant_manager and self.sessions_collection:
                result = self.qdrant_manager.get(
                    collection_name=self.sessions_collection,
                    ids=session_ids
                )
            else:
                logger.warning("Qdrant недоступен, кластеризация невозможна")
                return {"clusters_count": 0, "sessions_clustered": 0}

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

            # Сохраняем кластеры в Qdrant и обновляем метаданные сессий
            clusters_saved = 0
            sessions_clustered = 0

            for cluster in clusters:
                cluster_id = f"{slugify(chat_name)}-cluster-{cluster['cluster_id']}"

                # Обновляем метаданные сессий, добавляя информацию о кластере
                for session_id in cluster["session_ids"]:
                    try:
                        # Получаем текущую метаданные сессии
                        session_data = None
                        if self.qdrant_manager and self.sessions_collection:
                            session_data = self.qdrant_manager.get(
                                collection_name=self.sessions_collection,
                                ids=[session_id]
                            )

                        if session_data and session_data.get("ids"):
                            metadata = session_data["metadatas"][0].copy()
                            metadata["cluster_id"] = cluster_id
                            metadata["cluster_label"] = cluster.get("label", "")

                            # Обновляем метаданные через upsert (Qdrant не имеет отдельного update)
                            if self.qdrant_manager and self.sessions_collection:
                                # Получаем текущие данные для обновления
                                current_data = self.qdrant_manager.get(
                                    collection_name=self.sessions_collection,
                                    ids=[session_id]
                                )
                                if current_data and current_data.get("ids"):
                                    # Обновляем через upsert с новыми метаданными
                                    self.qdrant_manager.upsert(
                                        collection_name=self.sessions_collection,
                                        ids=[session_id],
                                        embeddings=current_data.get("embeddings", [[]])[:1] or [[]],
                                        metadatas=[metadata],
                                        documents=current_data.get("documents", [""])[:1] or [""],
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
                    if self.qdrant_manager and self.clusters_collection:
                        self.qdrant_manager.upsert(
                            collection_name=self.clusters_collection,
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
        if not self.qdrant_manager or not self.clusters_collection:
            return []

        try:
            where_filter = {"chat": chat} if chat else None

            result = self.qdrant_manager.get(
                collection_name=self.clusters_collection,
                where=where_filter,
                limit=limit
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
        if not self.qdrant_manager or not self.sessions_collection:
            return []

        try:
            result = self.qdrant_manager.get(
                collection_name=self.sessions_collection,
                where={"cluster_id": cluster_id}
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
            # Используем Qdrant для векторного хранилища
            existing_messages = None
            if self.qdrant_manager and self.messages_collection:
                existing_messages = self.qdrant_manager.get(
                    collection_name=self.messages_collection,
                    where={"chat": chat_name}
                )

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
            existing_sessions = None
            if self.qdrant_manager and self.sessions_collection:
                existing_sessions = self.qdrant_manager.get(
                    collection_name=self.sessions_collection,
                    where={"chat": chat_name}
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
            existing_sessions = None
            if self.qdrant_manager and self.sessions_collection:
                existing_sessions = self.qdrant_manager.get(
                    collection_name=self.sessions_collection,
                    where={"chat": chat_name}
                )

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
            existing_sessions = None
            if self.qdrant_manager and self.sessions_collection:
                existing_sessions = self.qdrant_manager.get(
                    collection_name=self.sessions_collection,
                    where={"chat": chat_name}
                )
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

    def _link_session_to_previous_sessions(
        self, session_id: str, chat: str, session_timestamp: datetime
    ) -> None:
        """Создает связи между текущей сессией и предыдущими сессиями того же чата."""
        if not self.graph:
            return
        
        try:
            from ..memory.graph_types import GraphEdge, EdgeType
            cursor = self.graph.conn.cursor()
            
            # Нормализуем имя чата для поиска (сессии могут иметь формат "semya-old-S0001")
            # Используем функцию slugify для получения нормализованного имени
            from ..utils.naming import slugify
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
                )
                ORDER BY json_extract(properties, '$.timestamp') DESC
                LIMIT 5
            """
            
            # Ищем сессии с различными форматами:
            # 1. С нормализованным именем: "semya-old-S%", "semya-S%"
            # 2. С оригинальным именем: "Семья-old-S%", "Семья-S%"
            # 3. С любым префиксом, если chat_slug пустой
            pattern1 = f"{chat_slug}-%-S%" if chat_slug else "%"
            pattern2 = f"{chat}-%-S%" if chat else "%"
            pattern3 = f"{chat_slug}-S%" if chat_slug else "%"
            
            cursor.execute(query, (session_id, chat, chat, pattern1, pattern2, pattern3, chat_slug))
            existing_sessions = cursor.fetchall()
            
            for row in existing_sessions:
                try:
                    props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                    if not props:
                        continue
                    
                    # Получаем timestamp предыдущей сессии
                    prev_timestamp_str = props.get("timestamp") or props.get("start_time_utc")
                    if not prev_timestamp_str:
                        continue
                    
                    from ..utils.datetime_utils import parse_datetime_utc
                    prev_timestamp = parse_datetime_utc(prev_timestamp_str, default=None)
                    if not prev_timestamp:
                        continue
                    
                    # Создаем связь только если сессии близки по времени (в пределах 7 дней)
                    time_diff = abs((session_timestamp - prev_timestamp).total_seconds())
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
                                "relation_type": "session_sequence"
                            },
                        )
                        try:
                            self.graph.add_edge(edge)
                            logger.debug(f"Создана связь между сессиями {source_id} -> {target_id}")
                        except Exception as e:
                            # Игнорируем ошибки, если связь уже существует
                            logger.debug(f"Не удалось создать связь между сессиями {source_id} и {target_id}: {e}")
                except Exception as e:
                    logger.debug(f"Ошибка при создании связи с сессией {row['id']}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Ошибка при связывании сессии {session_id} с предыдущими: {e}")

    async def _update_entity_nodes_with_descriptions(self) -> None:
        """Обновление EntityNode в графе с описаниями из словаря сущностей"""
        if not self.graph or not self.entity_dictionary:
            return
        
        try:
            from ..memory.graph_types import NodeType
            
            # Получаем все EntityNode из графа
            entity_nodes = self.graph.get_nodes_by_type(NodeType.ENTITY, limit=10000)
            
            updated_count = 0
            for node_data in entity_nodes:
                node_id = node_data.get("id")
                if not node_id:
                    continue
                
                # Получаем тип и имя сущности из узла
                entity_type = node_data.get("entity_type", "term")
                label = node_data.get("label", "")
                
                if not label:
                    continue
                
                # Получаем описание из словаря
                description = self.entity_dictionary.get_entity_description(entity_type, label)
                
                if description:
                    # Обновляем узел с описанием
                    # Проверяем, нужно ли обновлять (если описание изменилось или отсутствует)
                    current_description = node_data.get("description")
                    if current_description != description:
                        # Обновляем через update_node или напрямую через SQL
                        try:
                            # Обновляем узел через метод update_node графа
                            # Сначала получаем текущие свойства
                            if node_id in self.graph.graph:
                                node_data = self.graph.graph.nodes[node_id]
                                current_properties = node_data.get("properties", {})
                                
                                # Обновляем описание в свойствах
                                current_properties["description"] = description
                                
                                # Обновляем узел через метод графа
                                self.graph.update_node(
                                    node_id,
                                    properties=current_properties,
                                )
                                
                                # Также обновляем поле description напрямую, если узел это EntityNode
                                node_data["description"] = description
                                
                                updated_count += 1
                                logger.debug(f"Обновлено описание для EntityNode {node_id}: {description[:50]}...")
                        except Exception as e:
                            logger.debug(f"Ошибка при обновлении описания для EntityNode {node_id}: {e}")
            
            if updated_count > 0:
                logger.info(f"Обновлено {updated_count} EntityNode с описаниями")
        except Exception as e:
            logger.warning(f"Ошибка при обновлении EntityNode с описаниями: {e}")

    async def _build_and_index_entities(self, chat_name: str) -> None:
        """
        Построение и индексация профилей сущностей в векторное хранилище
        
        Args:
            chat_name: Название чата (для логирования)
        """
        if not self.entity_dictionary or not self.entity_vector_store or not self.graph:
            return
        
        try:
            from ..memory.graph_types import NodeType, EntityNode
            
            # Собираем все сущности из словаря
            all_entities = []
            for entity_type in self.entity_dictionary.learned_dictionaries:
                for normalized_value in self.entity_dictionary.learned_dictionaries[entity_type]:
                    # Получаем оригинальное значение (из entity_counts или из графа)
                    # Пробуем найти в графе для получения оригинального значения
                    entity_id = f"entity-{normalized_value.replace(' ', '-')}"
                    original_value = normalized_value
                    
                    if entity_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[entity_id]
                        original_value = node_data.get("label", normalized_value)
                    
                    all_entities.append((entity_type, normalized_value, original_value))
            
            if not all_entities:
                logger.debug(f"Нет сущностей для индексации в чате {chat_name}")
                return
            
            logger.info(f"Начинаем построение профилей для {len(all_entities)} сущностей из чата {chat_name}")
            
            indexed_count = 0
            failed_count = 0
            
            for entity_type, normalized_value, original_value in all_entities:
                try:
                    # Строим полный профиль сущности
                    profile = self.entity_dictionary.build_entity_profile(entity_type, original_value)
                    
                    if not profile:
                        continue
                    
                    # Формируем текст для эмбеддинга из полного описания
                    # Включаем: описание, алиасы, связанные сущности
                    embedding_text_parts = []
                    
                    if profile.get("description"):
                        embedding_text_parts.append(profile["description"])
                    
                    # Добавляем алиасы
                    aliases = profile.get("aliases", [])
                    if aliases:
                        embedding_text_parts.append(f"Также известен как: {', '.join(aliases[:5])}")
                    
                    # Добавляем информацию о связанных сущностях
                    related = profile.get("related_entities", [])
                    if related:
                        related_names = [r.get("label", "") for r in related[:3] if r.get("label")]
                        if related_names:
                            embedding_text_parts.append(f"Связан с: {', '.join(related_names)}")
                    
                    embedding_text = " ".join(embedding_text_parts)
                    
                    if not embedding_text:
                        # Если нет описания, используем имя и тип
                        embedding_text = f"{entity_type} {original_value}"
                    
                    # Генерируем эмбеддинг
                    async with self.embedding_client:
                        embedding = await self.embedding_client.embed(embedding_text)
                    
                    if not embedding or len(embedding) == 0:
                        logger.debug(f"Не удалось сгенерировать эмбеддинг для {entity_type}={normalized_value}")
                        failed_count += 1
                        continue
                    
                    # Формируем payload для Qdrant
                    entity_id = f"entity-{normalized_value.replace(' ', '-')}"
                    payload = {
                        "entity_type": entity_type,
                        "value": original_value,
                        "normalized_value": normalized_value,
                        "description": profile.get("description", ""),
                        "aliases": profile.get("aliases", []),
                        "importance": profile.get("importance", 0.5),
                        "mention_count": profile.get("mention_count", 0),
                        "chats": profile.get("chats", []),
                        "first_seen": profile.get("first_seen"),
                        "last_seen": profile.get("last_seen"),
                    }
                    
                    # Сохраняем в EntityVectorStore
                    self.entity_vector_store.upsert_entity(entity_id, embedding, payload)
                    
                    # Обновляем EntityNode в графе с эмбеддингом и полным описанием
                    if entity_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[entity_id]
                        current_properties = node_data.get("properties", {})
                        
                        # Обновляем описание и эмбеддинг
                        current_properties["description"] = profile.get("description", "")
                        current_properties["entity_profile"] = {
                            "mention_count": profile.get("mention_count", 0),
                            "chats": profile.get("chats", []),
                            "importance": profile.get("importance", 0.5),
                        }
                        
                        self.graph.update_node(
                            entity_id,
                            properties=current_properties,
                            embedding=embedding,
                        )
                    else:
                        # Создаем новый EntityNode, если его нет в графе
                        entity_node = EntityNode(
                            id=entity_id,
                            label=original_value,
                            entity_type=entity_type,
                            aliases=profile.get("aliases", []),
                            description=profile.get("description"),
                            importance=profile.get("importance", 0.5),
                            properties={
                                "normalized_value": normalized_value,
                                "entity_profile": {
                                    "mention_count": profile.get("mention_count", 0),
                                    "chats": profile.get("chats", []),
                                    "importance": profile.get("importance", 0.5),
                                },
                            },
                            embedding=embedding,
                        )
                        self.graph.add_node(entity_node)
                    
                    indexed_count += 1
                    
                    if indexed_count % 10 == 0:
                        logger.debug(f"Проиндексировано {indexed_count} сущностей...")
                        
                except Exception as e:
                    logger.warning(f"Ошибка при индексации сущности {entity_type}={normalized_value}: {e}")
                    failed_count += 1
                    continue
            
            logger.info(
                f"Завершена индексация сущностей из чата {chat_name}: "
                f"{indexed_count} успешно, {failed_count} ошибок"
            )
            
        except Exception as e:
            logger.warning(f"Ошибка при построении и индексации профилей сущностей: {e}")


if __name__ == "__main__":
    # Тест модуля
    async def test():
        indexer = TwoLevelIndexer()
        stats = await indexer.build_index(scope="all", recent_days=7)
        print(f"Статистика индексации: {stats}")

    asyncio.run(test())
