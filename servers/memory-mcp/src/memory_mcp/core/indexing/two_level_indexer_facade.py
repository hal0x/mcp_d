#!/usr/bin/env python3
"""Фасадный класс TwoLevelIndexer, использующий все менеджеры."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from ...analysis.segmentation import (
    AdaptiveMessageGrouper,
    DayGroupingSegmenter,
    SemanticRegrouper,
    SessionClusterer,
    SessionSegmenter,
)
from ...analysis.entities import EntityExtractor, EntityDictionary
from ...analysis.entities.entity_dictionary import get_entity_dictionary
from ...analysis.utils import InstructionManager, TimeProcessor
from ...analysis.rendering import MarkdownRenderer
from ...analysis.summarization import ClusterSummarizer
from ...analysis.summarization.session.summarizer import SessionSummarizer
from ...memory.storage.vector.qdrant_collections import QdrantCollectionsManager
from ...utils.system.naming import slugify
from ..adapters.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory

from .collections_manager import CollectionsManager
from .data_loader import DataLoader
from .l1_indexer import L1Indexer
from .l2_indexer import L2Indexer
from .tasks_indexer import TasksIndexer
from .clustering_manager import ClusteringManager
from .smart_aggregation import SmartAggregationManager
from .entities_indexer import EntitiesIndexer
from .progress_manager import ProgressManager

logger = logging.getLogger(__name__)


class TwoLevelIndexer:
    """Двухуровневая индексация: L1 (sessions с саммари) и L2 (messages с контекстом)."""

    def __init__(
        self,
        artifacts_path: str = "./artifacts",
        embedding_client: Optional[LangChainLLMAdapter] = None,
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
        """Инициализирует индексатор с указанными параметрами."""
        self.progress_callback = progress_callback
        
        self.artifacts_path = Path(artifacts_path).expanduser()
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.reports_path = self.artifacts_path / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        if embedding_client is None:
            embedding_client = get_llm_client_factory()
            if embedding_client is None:
                raise ValueError(
                    "Не удалось инициализировать LangChain LLM клиент. "
                    "Убедитесь, что LangChain установлен и MEMORY_MCP_LMSTUDIO_LLM_MODEL настроен."
                )
        self.embedding_client = embedding_client
        
        from ...config import get_settings
        settings = get_settings()
        qdrant_url = settings.get_qdrant_url()
        if qdrant_url:
            embedding_dimension = self.embedding_client.dimension if self.embedding_client else 1024
            self.qdrant_manager = QdrantCollectionsManager(url=qdrant_url, vector_size=embedding_dimension)
            if not self.qdrant_manager.available():
                logger.warning("Qdrant недоступен, коллекции не будут созданы")
                self.qdrant_manager = None
        else:
            logger.warning("QDRANT_URL не установлен, коллекции не будут созданы")
            self.qdrant_manager = None
        
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
        
        # Инициализируем словарь сущностей
        if enable_entity_learning:
            self.entity_dictionary = get_entity_dictionary(
                enable_llm_validation=True,
                enable_description_generation=settings.entity_description_enabled,
                graph=graph,
            )
        else:
            self.entity_dictionary = None
        
        self.markdown_renderer = MarkdownRenderer(self.reports_path)
        
        # Инициализируем граф памяти
        self.graph = graph
        if self.graph:
            from ...memory.ingest import MemoryIngestor
            self.ingestor = MemoryIngestor(self.graph)
            logger.info("TwoLevelIndexer: граф памяти подключен, записи будут синхронизироваться")
        else:
            self.ingestor = None
            logger.debug("TwoLevelIndexer: граф памяти не подключен, записи будут только в Qdrant")
        
        # Инициализируем VectorStore
        from ...memory.storage.vector.vector_store import build_vector_store_from_env
        self.vector_store = build_vector_store_from_env()
        if self.vector_store and self.vector_store.available():
            logger.info("VectorStore (Qdrant) инициализирован для векторного поиска")
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
        
        # Инициализируем EntityVectorStore
        if enable_entity_learning:
            from ...memory.storage.vector.vector_store import build_entity_vector_store_from_env
            self.entity_vector_store = build_entity_vector_store_from_env()
            if self.entity_vector_store:
                logger.info("EntityVectorStore инициализирован для индексации сущностей")
            else:
                logger.debug("EntityVectorStore недоступен (Qdrant не настроен)")
        else:
            self.entity_vector_store = None
        
        # Инициализируем менеджеры
        self.collections_manager = CollectionsManager(
            self.qdrant_manager,
            self.embedding_client,
            force_recreate=force,
        )
        
        self.data_loader = DataLoader(
            max_messages_per_group=max_messages_per_group,
            max_session_hours=max_session_hours,
            gap_minutes=gap_minutes,
            enable_time_analysis=enable_time_analysis,
        )
        
        # Инициализируем компоненты группировки сообщений
        self.semantic_regrouper = None
        self.adaptive_grouper = None
        if self.enable_message_grouping:
            if self.message_grouping_strategy == "semantic":
                self.semantic_regrouper = SemanticRegrouper(embedding_client=self.embedding_client)
            elif self.message_grouping_strategy == "adaptive":
                self.adaptive_grouper = AdaptiveMessageGrouper(
                    max_tokens=max_group_tokens,
                )
        
        # Инициализируем индексаторы
        self.l1_indexer = L1Indexer(
            embedding_client=self.embedding_client,
            session_summarizer=self.session_summarizer,
            qdrant_manager=self.qdrant_manager,
            sessions_collection=self.collections_manager.sessions_collection,
            ingestor=self.ingestor,
            graph=self.graph,
            vector_store=self.vector_store,
            force_recreate=force,
            collections_manager=self.collections_manager,
        )
        
        self.l2_indexer = L2Indexer(
            qdrant_manager=self.qdrant_manager,
            messages_collection=self.collections_manager.messages_collection,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            ingestor=self.ingestor,
            graph=self.graph,
            entity_extractor=self.entity_extractor,
            enable_message_grouping=enable_message_grouping,
            message_grouping_strategy=message_grouping_strategy,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            max_group_tokens=max_group_tokens,
            semantic_regrouper=self.semantic_regrouper,
            adaptive_grouper=self.adaptive_grouper,
            force=force,
            collections_manager=self.collections_manager,
        )
        
        self.tasks_indexer = TasksIndexer(
            embedding_client=self.embedding_client,
            qdrant_manager=self.qdrant_manager,
            tasks_collection=self.collections_manager.tasks_collection,
        )
        
        # Инициализируем менеджеры кластеризации и агрегации
        if self.enable_clustering:
            self.session_clusterer = SessionClusterer(
                similarity_threshold=self.clustering_threshold,
                min_cluster_size=self.min_cluster_size,
                use_hdbscan=False,
            )
            self.cluster_summarizer = ClusterSummarizer(
                embedding_client=self.embedding_client
            )
        else:
            self.session_clusterer = None
            self.cluster_summarizer = None
        
        self.clustering_manager = ClusteringManager(
            enable_clustering=enable_clustering,
            clustering_threshold=clustering_threshold,
            min_cluster_size=min_cluster_size,
            session_clusterer=self.session_clusterer,
            cluster_summarizer=self.cluster_summarizer,
            qdrant_manager=self.qdrant_manager,
            sessions_collection=self.collections_manager.sessions_collection,
            clusters_collection=self.collections_manager.clusters_collection,
            collections_manager=self.collections_manager,
        )
        
        self.smart_aggregation_manager = SmartAggregationManager(
            qdrant_manager=self.qdrant_manager,
            sessions_collection=self.collections_manager.sessions_collection,
            day_grouping_segmenter=self.day_grouping_segmenter,
            strategy_threshold=strategy_threshold,
            now_window_hours=now_window_hours,
            fresh_window_days=fresh_window_days,
            recent_window_days=recent_window_days,
        )
        
        self.entities_indexer = EntitiesIndexer(
            entity_dictionary=self.entity_dictionary,
            entity_vector_store=self.entity_vector_store,
            graph=self.graph,
            embedding_client=self.embedding_client,
        )
        
        # Инициализируем менеджер прогресса
        # Прогресс теперь хранится в SQLite через IndexingJobTracker, не в Qdrant
        self.progress_manager = ProgressManager(
            qdrant_manager=None,  # Не используем Qdrant для прогресса
            progress_collection=None,
            progress_callback=progress_callback,
        )
        
        logger.info("TwoLevelIndexer инициализирован со всеми менеджерами")

    async def build_index(
        self,
        scope: str = "all",
        chat: Optional[str] = None,
        force_full: bool = False,
        recent_days: int = 7,
        adapter: Optional[Any] = None,
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
                self.progress_manager.call_progress_callback(
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
                
                # Загружаем сообщения
                messages = await self.data_loader.load_messages_from_chat(chat_dir)
                
                if not messages:
                    logger.warning(f"Нет сообщений в чате {chat_name}")
                    continue
                
                # Определяем, какие сообщения нужно переиндексировать
                if not force_full:
                    last_indexed_date = self.progress_manager.get_last_indexed_date(chat_name)
                    
                    if last_indexed_date:
                        messages_to_index = [
                            m
                            for m in messages
                            if self.data_loader.parse_message_time(m) > last_indexed_date
                        ]
                        logger.info(
                            f"📊 Инкрементальная индексация: последнее проиндексированное "
                            f"сообщение от {last_indexed_date.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    else:
                        if recent_days > 0:
                            recent_cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(
                                days=recent_days
                            )
                            messages_to_index = [
                                m
                                for m in messages
                                if self.data_loader.parse_message_time(m) >= recent_cutoff
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
                
                # Группируем сообщения
                if self.enable_smart_aggregation:
                    logger.info("🧠 Используем умную группировку с скользящими окнами")
                    sessions = self.smart_aggregation_manager.group_messages_by_smart_strategy(
                        messages_to_index, chat_name
                    )
                else:
                    # Используем классическую группировку через DataLoader
                    day_groups = self.day_grouping_segmenter.group_messages_by_days(
                        messages_to_index, chat_name
                    )
                    sessions = self.data_loader.expand_day_groups(day_groups, chat_name)
                
                logger.info(f"Создано {len(sessions)} сессий")
                
                # Индексируем сессии
                chat_stats = await self._index_chat_sessions(
                    chat_name, sessions, force_full, job_id
                )
                
                stats["indexed_chats"].append(chat_name)
                stats["sessions_indexed"] += chat_stats.get("sessions_indexed", 0)
                stats["messages_indexed"] += chat_stats.get("messages_indexed", 0)
                stats["tasks_indexed"] += chat_stats.get("tasks_indexed", 0)
                
                # Сохраняем прогресс
                if messages_to_index:
                    last_message_date = self.data_loader.parse_message_time(messages_to_index[-1])
                    self.progress_manager.save_indexing_progress(
                        chat_name,
                        last_message_date,
                        len(messages_to_index),
                        len(sessions),
                    )
                
                # Кластеризация
                if self.enable_clustering:
                    summaries = await self._load_existing_summaries(chat_name, sessions)
                    if summaries:
                        clustering_stats = await self.clustering_manager.cluster_chat_sessions(
                            chat_name, summaries
                        )
                        logger.info(
                            f"Кластеризация: {clustering_stats.get('clusters_count', 0)} кластеров, "
                            f"{clustering_stats.get('sessions_clustered', 0)} сессий"
                        )
                
                # Индексация сущностей
                if self.enable_entity_learning:
                    await self.entities_indexer.build_and_index_entities(chat_name)
                
                # Callback: завершение обработки чата
                self.progress_manager.call_progress_callback(
                    job_id,
                    "chat_completed",
                    {
                        "chat": chat_name,
                        "stats": chat_stats,
                    },
                )
                
            except Exception as e:
                logger.error(f"Ошибка при индексации чата {chat_dir.name}: {e}", exc_info=True)
                continue
        
        logger.info(f"Индексация завершена: {stats}")
        return stats

    async def _index_chat_sessions(
        self,
        chat_name: str,
        sessions: List[Dict[str, Any]],
        force_full: bool,
        job_id: Optional[str],
    ) -> Dict[str, Any]:
        """Индексирует сессии чата."""
        stats = {
            "sessions_indexed": 0,
            "messages_indexed": 0,
            "tasks_indexed": 0,
        }
        
        # Получаем существующие сессии
        existing_session_ids = set()
        if not force_full and self.qdrant_manager and self.collections_manager.sessions_collection:
            try:
                result = self.qdrant_manager.get(
                    collection_name=self.collections_manager.sessions_collection,
                    where={"chat": chat_name}
                )
                if result and result.get("ids"):
                    existing_session_ids = set(result["ids"])
            except Exception as e:
                logger.debug(f"Не удалось получить существующие сессии: {e}")
        
        # Индексируем каждую сессию
        for session_idx, session in enumerate(sessions, 1):
            try:
                session_id = session.get("session_id")
                if not session_id:
                    logger.warning(f"Сессия без ID, пропускаем")
                    continue
                
                # Пропускаем существующие сессии при инкрементальной индексации
                if not force_full and session_id in existing_session_ids:
                    logger.debug(f"Сессия {session_id} уже проиндексирована, пропускаем")
                    continue
                
                logger.info(f"Индексация сессии {session_idx}/{len(sessions)}: {session_id}")
                
                # Callback: начало обработки сессии
                self.progress_manager.call_progress_callback(
                    job_id,
                    "session_started",
                    {
                        "chat": chat_name,
                        "session_id": session_id,
                        "session_index": session_idx,
                        "total_sessions": len(sessions),
                    },
                )
                
                # Саммаризация сессии (L1)
                summary = await self.session_summarizer.summarize_session(
                    session, chat_name
                )
                
                if not summary:
                    logger.warning(f"Не удалось создать саммаризацию для сессии {session_id}")
                    continue
                
                # Индексация L1
                await self.l1_indexer.index_session_l1(summary)
                stats["sessions_indexed"] += 1
                
                # Индексация L2 (сообщения)
                messages_count = await self.l2_indexer.index_messages_l2(session)
                stats["messages_indexed"] += messages_count
                
                # Индексация задач
                tasks_count = await self.tasks_indexer.index_tasks(summary)
                stats["tasks_indexed"] += tasks_count
                
                # Сохраняем саммаризацию в JSON
                await self._save_summary(summary, chat_name)
                
                # Callback: завершение обработки сессии
                self.progress_manager.call_progress_callback(
                    job_id,
                    "session_completed",
                    {
                        "chat": chat_name,
                        "session_id": session_id,
                        "messages_indexed": messages_count,
                        "tasks_indexed": tasks_count,
                    },
                )
                
            except Exception as e:
                logger.error(f"Ошибка при индексации сессии {session.get('session_id', 'unknown')}: {e}", exc_info=True)
                continue
        
        return stats

    async def _save_summary(self, summary: Dict[str, Any], chat_name: str) -> None:
        """Сохраняет саммаризацию в JSON файл."""
        try:
            chat_slug = slugify(chat_name)
            sessions_dir = self.reports_path / chat_slug / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            
            session_id = summary.get("session_id")
            if not session_id:
                logger.warning("Саммаризация без session_id, не сохраняем")
                return
            
            json_file = sessions_dir / f"{session_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Саммаризация сохранена: {json_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении саммаризации: {e}")

    async def _load_existing_summaries(
        self, chat_name: str, sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Загружает существующие саммаризации из JSON файлов."""
        summaries = []
        try:
            chat_slug = slugify(chat_name)
            sessions_dir = self.reports_path / chat_slug / "sessions"
            
            if not sessions_dir.exists():
                return summaries
            
            session_ids = {s.get("session_id") for s in sessions if s.get("session_id")}
            
            for session_id in session_ids:
                json_file = sessions_dir / f"{session_id}.json"
                if json_file.exists():
                    try:
                        with open(json_file, encoding="utf-8") as f:
                            summary = json.load(f)
                            summaries.append(summary)
                    except Exception as e:
                        logger.debug(f"Не удалось загрузить саммаризацию {session_id}: {e}")
        except Exception as e:
            logger.warning(f"Ошибка при загрузке саммаризаций: {e}")
        
        return summaries

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
        return self.clustering_manager.get_clusters(chat=chat, limit=limit)

    def get_cluster_sessions(
        self, cluster_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Получить сессии, принадлежащие кластеру

        Args:
            cluster_id: ID кластера
            limit: Максимальное количество сессий

        Returns:
            Список сессий
        """
        return self.clustering_manager.get_cluster_sessions(cluster_id, limit=limit)

