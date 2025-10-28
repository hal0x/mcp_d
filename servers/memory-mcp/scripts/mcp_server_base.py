#!/usr/bin/env python3
"""
🚀 MCP Сервер для поиска по Telegram дампам

Предоставляет доступ к индексированным данным через Model Context Protocol (MCP).
Поддерживает поиск по сообщениям, сессиям и задачам.
"""
# mypy: ignore-errors

import asyncio
import json
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импорты после изменения PYTHONPATH
from memory_mcp.core.indexer import TwoLevelIndexer  # noqa: E402
from memory_mcp.core.ollama_client import OllamaEmbeddingClient  # noqa: E402
from memory_mcp.utils.russian_tokenizer import normalize_word, tokenize_text  # noqa: E402

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TelegramDumpMCPBase:
    """MCP сервер для поиска по Telegram дампам"""

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        chats_path: str = "./chats",
        artifacts_path: str = "./artifacts",
    ):
        """
        Инициализация MCP сервера

        Args:
            chroma_path: Путь к базе ChromaDB
            chats_path: Путь к директории с чатами
            artifacts_path: Путь к директории с артефактами индексации
        """
        self.chroma_path = Path(chroma_path)
        self.chats_path = Path(chats_path)
        self.artifacts_path = Path(artifacts_path)
        self.server = Server("memory-mcp")

        # Клиенты (инициализируются при первом использовании)
        self._chroma_client: Optional[chromadb.Client] = None
        self._ollama_client: Optional[OllamaEmbeddingClient] = None

        # Коллекции
        self._collections: Dict[str, Any] = {}

        # Регистрируем обработчики
        self._register_handlers()

    @property
    def chroma_client(self) -> chromadb.Client:
        """Ленивая инициализация ChromaDB клиента"""
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            logger.info(f"✅ ChromaDB подключен: {self.chroma_path}")
        return self._chroma_client

    @property
    def ollama_client(self) -> OllamaEmbeddingClient:
        """Ленивая инициализация Ollama клиента"""
        if self._ollama_client is None:
            self._ollama_client = OllamaEmbeddingClient()
            logger.info("✅ Ollama клиент инициализирован")
        return self._ollama_client

    def _get_collection(self, collection_name: str):
        """Получить коллекцию с кешированием"""
        if collection_name not in self._collections:
            try:
                self._collections[collection_name] = self.chroma_client.get_collection(
                    collection_name
                )
                logger.info(f"✅ Коллекция загружена: {collection_name}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки коллекции {collection_name}: {e}")
                return None
        return self._collections[collection_name]

    async def _health_payload(self) -> Dict[str, Any]:
        """Базовая health-информация: пути и статистика индексов."""
        try:
            stats_raw = await self._get_stats()
            stats = json.loads(stats_raw) if isinstance(stats_raw, str) else stats_raw
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Не удалось собрать статистику для health: %s", exc)
            stats = {"error": str(exc)}

        return {
            "status": "healthy" if "error" not in stats else "degraded",
            "paths": {
                "chroma": str(self.chroma_path),
                "chats": str(self.chats_path),
                "artifacts": str(self.artifacts_path),
            },
            "stats": stats,
        }

    def _version_payload(self) -> Dict[str, Any]:
        """Информация о версии пакета memory-mcp."""
        try:  # pragma: no cover - metadata lookup
            version = metadata.version("memory_mcp")
        except metadata.PackageNotFoundError:
            version = "0.0.0"
        return {
            "name": "memory-mcp",
            "version": version,
        }

    def _config_snapshot(self) -> Dict[str, Any]:
        """Сводка основных путей конфигурации."""
        return {
            "chroma_path": str(self.chroma_path),
            "chats_path": str(self.chats_path),
            "artifacts_path": str(self.artifacts_path),
        }

    def _register_handlers(self):
        """Регистрация обработчиков MCP"""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """Список доступных ресурсов"""
            resources = []

            # Статистика
            resources.append(
                Resource(
                    uri="telegram://stats",
                    name="Статистика индексов",
                    mimeType="application/json",
                    description="Статистика по всем индексированным данным",
                )
            )

            # Список чатов
            resources.append(
                Resource(
                    uri="telegram://chats",
                    name="Список чатов",
                    mimeType="application/json",
                    description="Список всех доступных чатов с количеством сообщений",
                )
            )

            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Чтение ресурса"""
            if uri == "telegram://stats":
                return await self._get_stats()
            elif uri == "telegram://chats":
                return await self._get_chats_list()
            else:
                raise ValueError(f"Неизвестный ресурс: {uri}")

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Список доступных инструментов"""
            return [
                Tool(
                    name="health",
                    description="Проверка состояния MCP сервера, индексов и путей артефактов",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="version",
                    description="Информация о версии memory-mcp и поддерживаемых возможностях",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="search_messages",
                    description=(
                        "Поиск по сообщениям в Telegram чатах. "
                        "Использует семантический поиск с векторными эмбеддингами. "
                        "При глубине 'deep' возвращает полные артефакты сессий."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос",
                            },
                            "chat_filter": {
                                "type": "string",
                                "description": "Фильтр по имени чата (опционально)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимальное количество результатов",
                                "default": 10,
                            },
                            "depth": {
                                "type": "string",
                                "description": "Глубина поиска: 'shallow' (только результаты), 'medium' (с метаданными), 'deep' (с полными артефактами)",
                                "enum": ["shallow", "medium", "deep"],
                                "default": "shallow",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_sessions",
                    description=(
                        "Поиск по саммаризациям сессий разговоров. "
                        "Сессия = сгруппированные сообщения с кратким описанием. "
                        "При глубине 'deep' возвращает полные отчёты сессий."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос",
                            },
                            "chat_filter": {
                                "type": "string",
                                "description": "Фильтр по имени чата (опционально)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимальное количество результатов",
                                "default": 5,
                            },
                            "depth": {
                                "type": "string",
                                "description": "Глубина поиска: 'shallow' (только результаты), 'medium' (с метаданными), 'deep' (с полными артефактами)",
                                "enum": ["shallow", "medium", "deep"],
                                "default": "shallow",
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Включать ли метаданные (Risks, Actions, Attachments, Uncertainties) в результаты",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_tasks",
                    description=(
                        "Поиск по задачам и action items, извлечённым из чатов. "
                        "Включает информацию о приоритете, владельце и сроках."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос",
                            },
                            "chat_filter": {
                                "type": "string",
                                "description": "Фильтр по имени чата (опционально)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимальное количество результатов",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_chat_info",
                    description="Получить информацию о конкретном чате",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {"type": "string", "description": "Имя чата"}
                        },
                        "required": ["chat_name"],
                    },
                ),
                Tool(
                    name="get_stats",
                    description="Получить статистику по всем индексам",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="tokenize_text",
                    description=(
                        "Токенизация текста с улучшенной поддержкой русского языка. "
                        "Включает обработку чисел, валют, процентов и морфологический анализ."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Текст для токенизации",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                Tool(
                    name="search_numeric_data",
                    description=(
                        "Поиск по числовым данным в сообщениях. "
                        "Поддерживает поиск по валютам, суммам, процентам и большим числам."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос с числовыми данными",
                            },
                            "chat_filter": {
                                "type": "string",
                                "description": "Фильтр по имени чата (опционально)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимальное количество результатов",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="analyze_chat_content",
                    description=(
                        "Анализ содержимого чата с использованием улучшенной токенизации. "
                        "Извлекает числовые данные, валюты, проценты и ключевые термины."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {
                                "type": "string",
                                "description": "Имя чата для анализа",
                            },
                            "sample_size": {
                                "type": "integer",
                                "description": "Количество сообщений для анализа",
                                "default": 100,
                            },
                        },
                        "required": ["chat_name"],
                    },
                ),
                Tool(
                    name="read_session_report",
                    description=(
                        "Чтение детального отчёта сессии из артефактов индексации. "
                        "Возвращает полную структурированную информацию о сессии включая Topics, Discussion, Actions, Risks."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {
                                "type": "string",
                                "description": "Имя чата",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "ID сессии (например, Семья-S0001)",
                            },
                        },
                        "required": ["chat_name", "session_id"],
                    },
                ),
                Tool(
                    name="read_chat_context",
                    description=(
                        "Чтение контекстного файла чата из артефактов. "
                        "Содержит накапливающуюся информацию о чате и его участниках."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {
                                "type": "string",
                                "description": "Имя чата",
                            },
                        },
                        "required": ["chat_name"],
                    },
                ),
                Tool(
                    name="list_chat_artifacts",
                    description=(
                        "Получение списка доступных артефактов для чата. "
                        "Показывает доступные сессии, контексты и другие файлы."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {
                                "type": "string",
                                "description": "Имя чата",
                            },
                        },
                        "required": ["chat_name"],
                    },
                ),
                Tool(
                    name="get_chats_list",
                    description=(
                        "Получение списка всех доступных чатов для поиска. "
                        "Возвращает информацию о каждом чате включая количество сообщений и даты."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="index_chat",
                    description=(
                        "Индексация конкретного чата с отслеживанием прогресса и пересозданием всех артефактов. "
                        "Поддерживает полную переиндексацию, инкрементальную индексацию и создание новых артефактов."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chat_name": {
                                "type": "string",
                                "description": "Название чата для индексации",
                            },
                            "force_full": {
                                "type": "boolean",
                                "description": "Полная пересборка всех артефактов чата",
                                "default": False,
                            },
                            "recent_days": {
                                "type": "integer",
                                "description": "Количество последних дней для индексации (0 = все сообщения)",
                                "default": 0,
                            },
                            "enable_clustering": {
                                "type": "boolean",
                                "description": "Включить кластеризацию сессий",
                                "default": False,
                            },
                            "enable_smart_aggregation": {
                                "type": "boolean",
                                "description": "Включить умную группировку с скользящими окнами",
                                "default": True,
                            },
                            "max_messages_per_group": {
                                "type": "integer",
                                "description": "Максимальное количество сообщений в группе",
                                "default": 200,
                            },
                            "max_session_hours": {
                                "type": "integer",
                                "description": "Максимальная длительность сессии в часах",
                                "default": 12,
                            },
                            "gap_minutes": {
                                "type": "integer",
                                "description": "Максимальный разрыв между сообщениями в минутах",
                                "default": 120,
                            },
                        },
                        "required": ["chat_name"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Вызов инструмента"""
            try:
                if name == "health":
                    payload = await self._health_payload()
                    result = (
                        payload
                        if isinstance(payload, str)
                        else json.dumps(payload, ensure_ascii=False, indent=2)
                    )
                elif name == "version":
                    payload = self._version_payload()
                    result = (
                        payload
                        if isinstance(payload, str)
                        else json.dumps(payload, ensure_ascii=False, indent=2)
                    )
                elif name == "search_messages":
                    result = await self._search_collection(
                        collection_name="chat_messages",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 10),
                        depth=arguments.get("depth", "shallow"),
                    )
                elif name == "search_sessions":
                    result = await self._search_collection(
                        collection_name="chat_sessions",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 5),
                        depth=arguments.get("depth", "shallow"),
                        include_metadata=arguments.get("include_metadata", False),
                    )
                elif name == "search_tasks":
                    result = await self._search_collection(
                        collection_name="chat_tasks",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 5),
                    )
                elif name == "get_chat_info":
                    result = await self._get_chat_info(arguments["chat_name"])
                elif name == "get_stats":
                    result = await self._get_stats()
                elif name == "tokenize_text":
                    result = await self._tokenize_text(arguments["text"])
                elif name == "search_numeric_data":
                    result = await self._search_numeric_data(
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 10),
                    )
                elif name == "analyze_chat_content":
                    result = await self._analyze_chat_content(
                        chat_name=arguments["chat_name"],
                        sample_size=arguments.get("sample_size", 100),
                    )
                elif name == "read_session_report":
                    result = await self._read_session_report(
                        chat_name=arguments["chat_name"],
                        session_id=arguments["session_id"],
                    )
                elif name == "read_chat_context":
                    result = await self._read_chat_context(arguments["chat_name"])
                elif name == "list_chat_artifacts":
                    result = await self._list_chat_artifacts(arguments["chat_name"])
                elif name == "get_chats_list":
                    result = await self._get_chats_list()
                elif name == "index_chat":
                    result = await self._index_chat(
                        chat_name=arguments["chat_name"],
                        force_full=arguments.get("force_full", False),
                        recent_days=arguments.get("recent_days", 0),
                        enable_clustering=arguments.get("enable_clustering", False),
                        enable_smart_aggregation=arguments.get(
                            "enable_smart_aggregation", True
                        ),
                        max_messages_per_group=arguments.get(
                            "max_messages_per_group", 200
                        ),
                        max_session_hours=arguments.get("max_session_hours", 12),
                        gap_minutes=arguments.get("gap_minutes", 120),
                    )
                else:
                    result = json.dumps({"error": f"Неизвестный инструмент: {name}"})

                result_text = (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, ensure_ascii=False, indent=2)
                )
                return [TextContent(type="text", text=result_text)]

            except Exception as e:
                logger.error(f"Ошибка при вызове {name}: {e}", exc_info=True)
                error_result = json.dumps(
                    {"error": str(e), "tool": name, "arguments": arguments}
                )
                return [TextContent(type="text", text=error_result)]

    async def _search_collection(
        self,
        collection_name: str,
        query: str,
        chat_filter: Optional[str] = None,
        limit: int = 10,
        depth: str = "shallow",
        include_metadata: bool = False,
    ) -> str:
        """
        Поиск в коллекции

        Args:
            collection_name: Имя коллекции
            query: Поисковый запрос
            chat_filter: Фильтр по чату
            limit: Лимит результатов
            depth: Глубина поиска (shallow, medium, deep)
            include_metadata: Включать ли метаданные (Risks, Actions, Attachments)

        Returns:
            JSON строка с результатами
        """
        collection = self._get_collection(collection_name)
        if collection is None:
            return json.dumps(
                {
                    "error": f"Коллекция {collection_name} не найдена",
                    "hint": "Запустите 'memory_mcp index' для создания индексов",
                },
                ensure_ascii=False,
            )

        # Генерируем эмбеддинг запроса
        async with self.ollama_client:
            # Нормализуем поисковый запрос
            normalized_query = normalize_word(query)

            query_embedding = await self.ollama_client._generate_single_embedding(
                normalized_query
            )

            if not query_embedding:
                return json.dumps(
                    {"error": "Не удалось сгенерировать эмбеддинг для запроса"},
                    ensure_ascii=False,
                )

        # Выполняем поиск
        try:
            where_filter = {"chat": chat_filter} if chat_filter else None

            results = collection.query(
                query_embeddings=[query_embedding], n_results=limit, where=where_filter
            )

            if not results["documents"] or not results["documents"][0]:
                return json.dumps(
                    {
                        "query": query,
                        "collection": collection_name,
                        "results": [],
                        "total": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Форматируем результаты
            formatted_results = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # L2 distance: меньше = лучше (0 = идентичные векторы)
                # Показываем как есть, без конвертации

                result_item = {
                    "text": doc,
                    "distance": round(distance, 3),  # L2 расстояние
                    "metadata": metadata,
                }

                # Добавляем специфичные поля в зависимости от коллекции
                if collection_name == "chat_messages":
                    result_item["chat"] = metadata.get("chat", "Unknown")
                    result_item["date"] = metadata.get("date_utc", "Unknown")
                elif collection_name == "chat_sessions":
                    result_item["session_id"] = metadata.get("session_id", "Unknown")
                    result_item["chat"] = metadata.get("chat", "Unknown")
                    result_item["time_range"] = metadata.get("time_span", "Unknown")
                elif collection_name == "chat_tasks":
                    result_item["chat"] = metadata.get("chat", "Unknown")
                    result_item["priority"] = metadata.get("priority", "normal")
                    result_item["owner"] = metadata.get("owner", "N/A")
                    result_item["due_date"] = metadata.get("due_date", "N/A")

                formatted_results.append(result_item)

            # Добавляем артефакты при глубоком поиске
            artifacts = {}
            if depth == "deep":
                artifacts = await self._get_artifacts_for_results(
                    formatted_results, collection_name
                )

            result = {
                "query": query,
                "collection": collection_name,
                "chat_filter": chat_filter,
                "depth": depth,
                "include_metadata": include_metadata,
                "results": formatted_results,
                "total": len(formatted_results),
            }

            # Добавляем метаданные если запрошено
            if include_metadata and collection_name == "chat_sessions":
                metadata_results = await self._get_metadata_for_sessions(
                    formatted_results
                )
                if metadata_results:
                    result["metadata"] = metadata_results

            # Добавляем артефакты только при глубоком поиске
            if depth == "deep" and artifacts:
                result["artifacts"] = artifacts

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}", exc_info=True)
            return json.dumps(
                {"error": f"Ошибка при поиске: {str(e)}"}, ensure_ascii=False
            )

    async def _get_stats(self) -> str:
        """Получить статистику по индексам"""
        try:
            stats = {}

            for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                collection = self._get_collection(collection_name)
                if collection:
                    stats[collection_name] = collection.count()
                else:
                    stats[collection_name] = 0

            # Статистика по чатам
            chats = []
            if self.chats_path.exists():
                for chat_dir in sorted(self.chats_path.iterdir()):
                    if chat_dir.is_dir():
                        # Пробуем найти JSON файл (может быть result.json или unknown.json)
                        json_file = chat_dir / "result.json"
                        if not json_file.exists():
                            json_file = chat_dir / "unknown.json"
                        if json_file.exists():
                            chats.append(chat_dir.name)

            return json.dumps(
                {
                    "collections": stats,
                    "total_records": sum(stats.values()),
                    "total_chats": len(chats),
                    "chroma_path": str(self.chroma_path),
                    "chats_path": str(self.chats_path),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _get_chats_list(self) -> str:
        """Получить список всех чатов"""
        try:
            chats = []

            if not self.chats_path.exists():
                return json.dumps(
                    {
                        "error": "Директория с чатами не найдена",
                        "path": str(self.chats_path),
                    },
                    ensure_ascii=False,
                )

            for chat_dir in sorted(self.chats_path.iterdir()):
                if not chat_dir.is_dir():
                    continue

                # Пробуем найти JSON файл (может быть result.json или unknown.json)
                json_file = chat_dir / "result.json"
                if not json_file.exists():
                    json_file = chat_dir / "unknown.json"
                if not json_file.exists():
                    continue

                try:
                    messages = []
                    first_message_date = None
                    last_message_date = None

                    with open(json_file, encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                message = json.loads(line)
                                messages.append(message)

                                # Получаем дату сообщения
                                message_date = message.get("date_utc") or message.get(
                                    "date"
                                )
                                if message_date:
                                    if first_message_date is None:
                                        first_message_date = message_date
                                    last_message_date = message_date

                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Ошибка парсинга строки {line_num} в {json_file}: {e}"
                                )
                                continue

                    chat_info = {
                        "name": chat_dir.name,
                        "type": "unknown",
                        "message_count": len(messages),
                        "path": str(chat_dir),
                    }

                    # Добавляем даты первого и последнего сообщения
                    if first_message_date:
                        chat_info["first_message"] = first_message_date
                    if last_message_date:
                        chat_info["last_message"] = last_message_date

                    chats.append(chat_info)

                except Exception as e:
                    logger.warning(f"Ошибка чтения чата {chat_dir.name}: {e}")
                    continue

            return json.dumps(
                {"chats": chats, "total": len(chats)}, ensure_ascii=False, indent=2
            )

        except Exception as e:
            logger.error(f"Ошибка при получении списка чатов: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _get_chat_info(self, chat_name: str) -> str:
        """Получить информацию о конкретном чате"""
        try:
            # Ищем директорию чата
            chat_dir = None
            for d in self.chats_path.iterdir():
                if d.is_dir() and d.name == chat_name:
                    chat_dir = d
                    break

            if not chat_dir:
                return json.dumps(
                    {"error": f"Чат '{chat_name}' не найден"}, ensure_ascii=False
                )

            # Пробуем найти JSON файл (может быть result.json или unknown.json)
            json_file = chat_dir / "result.json"
            if not json_file.exists():
                json_file = chat_dir / "unknown.json"
            if not json_file.exists():
                return json.dumps(
                    {"error": f"JSON файл не найден для чата '{chat_name}'"},
                    ensure_ascii=False,
                )

            # Читаем JSON Lines формат (каждая строка - отдельный JSON объект)
            messages = []
            with open(json_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            message = json.loads(line)
                            messages.append(message)
                        except json.JSONDecodeError:
                            continue

            # Собираем статистику
            info = {
                "name": chat_name,
                "type": "unknown",
                "id": "unknown",
                "message_count": len(messages),
            }

            if messages:
                info["first_message"] = messages[0].get("date_utc", "Unknown")
                info["last_message"] = messages[-1].get("date_utc", "Unknown")

                # Статистика по типам сообщений
                text_count = sum(1 for m in messages if m.get("text"))
                info["text_messages"] = text_count

            return json.dumps(info, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Ошибка при получении информации о чате: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _tokenize_text(self, text: str) -> str:
        """Токенизация текста с улучшенной поддержкой русского языка"""
        try:
            tokens = tokenize_text(text)

            # Анализируем типы токенов
            money_tokens = [token for token in tokens if token.startswith("money_")]
            amount_tokens = [token for token in tokens if token.startswith("amount_")]
            value_tokens = [token for token in tokens if token.startswith("value_")]
            type_tokens = [
                token
                for token in tokens
                if token in ["billion", "million", "thousand", "percentage"]
            ]
            russian_tokens = [
                token
                for token in tokens
                if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in token.lower())
            ]
            english_tokens = [
                token
                for token in tokens
                if token.isalpha()
                and not any(
                    c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in token.lower()
                )
            ]

            result = {
                "original_text": text,
                "tokens": tokens,
                "token_count": len(tokens),
                "analysis": {
                    "money_tokens": money_tokens,
                    "amount_tokens": amount_tokens,
                    "value_tokens": value_tokens,
                    "type_tokens": type_tokens,
                    "russian_tokens": russian_tokens,
                    "english_tokens": english_tokens,
                },
                "statistics": {
                    "money_count": len(money_tokens),
                    "amount_count": len(amount_tokens),
                    "value_count": len(value_tokens),
                    "type_count": len(type_tokens),
                    "russian_count": len(russian_tokens),
                    "english_count": len(english_tokens),
                },
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Ошибка при токенизации: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _search_numeric_data(
        self,
        query: str,
        chat_filter: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Поиск по числовым данным с улучшенной токенизацией"""
        try:
            # Сначала токенизируем запрос
            tokens = tokenize_text(query)

            # Ищем числовые токены
            numeric_tokens = []
            for token in tokens:
                if (
                    token.startswith("money_")
                    or token.startswith("amount_")
                    or token.startswith("value_")
                    or token in ["billion", "million", "thousand", "percentage"]
                ):
                    numeric_tokens.append(token)

            if not numeric_tokens:
                return json.dumps(
                    {
                        "query": query,
                        "tokens": tokens,
                        "numeric_tokens": [],
                        "message": "В запросе не найдено числовых данных",
                        "results": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Выполняем поиск по коллекции сообщений
            collection = self._get_collection("chat_messages")
            if collection is None:
                return json.dumps(
                    {
                        "error": "Коллекция chat_messages не найдена",
                        "hint": "Запустите 'memory_mcp index' для создания индексов",
                    },
                    ensure_ascii=False,
                )

            # Генерируем эмбеддинг запроса
            async with self.ollama_client:
                query_embedding = await self.ollama_client._generate_single_embedding(
                    query
                )

                if not query_embedding:
                    return json.dumps(
                        {"error": "Не удалось сгенерировать эмбеддинг для запроса"},
                        ensure_ascii=False,
                    )

            # Выполняем поиск
            where_filter = {"chat": chat_filter} if chat_filter else None

            results = collection.query(
                query_embeddings=[query_embedding], n_results=limit, where=where_filter
            )

            if not results["documents"] or not results["documents"][0]:
                return json.dumps(
                    {
                        "query": query,
                        "tokens": tokens,
                        "numeric_tokens": numeric_tokens,
                        "results": [],
                        "total": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Анализируем результаты на предмет числовых данных
            analyzed_results = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Токенизируем найденный документ
                doc_tokens = tokenize_text(doc)
                doc_numeric_tokens = []

                for token in doc_tokens:
                    if (
                        token.startswith("money_")
                        or token.startswith("amount_")
                        or token.startswith("value_")
                        or token in ["billion", "million", "thousand", "percentage"]
                    ):
                        doc_numeric_tokens.append(token)

                result_item = {
                    "text": doc,
                    "distance": round(distance, 3),
                    "metadata": metadata,
                    "tokens": doc_tokens,
                    "numeric_tokens": doc_numeric_tokens,
                    "numeric_match": len(set(numeric_tokens) & set(doc_numeric_tokens))
                    > 0,
                }

                analyzed_results.append(result_item)

            return json.dumps(
                {
                    "query": query,
                    "tokens": tokens,
                    "numeric_tokens": numeric_tokens,
                    "chat_filter": chat_filter,
                    "results": analyzed_results,
                    "total": len(analyzed_results),
                    "numeric_matches": len(
                        [r for r in analyzed_results if r["numeric_match"]]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"Ошибка при поиске числовых данных: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _analyze_chat_content(
        self, chat_name: str, sample_size: int = 100
    ) -> str:
        """Анализ содержимого чата с использованием улучшенной токенизации"""
        try:
            # Ищем директорию чата
            chat_dir = None
            for d in self.chats_path.iterdir():
                if d.is_dir() and d.name == chat_name:
                    chat_dir = d
                    break

            if not chat_dir:
                return json.dumps(
                    {"error": f"Чат '{chat_name}' не найден"}, ensure_ascii=False
                )

            # Пробуем найти JSON файл
            json_file = chat_dir / "result.json"
            if not json_file.exists():
                json_file = chat_dir / "unknown.json"
            if not json_file.exists():
                return json.dumps(
                    {"error": f"JSON файл не найден для чата '{chat_name}'"},
                    ensure_ascii=False,
                )

            # Читаем JSON Lines формат (каждая строка - отдельный JSON объект)
            messages = []
            with open(json_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            message = json.loads(line)
                            messages.append(message)
                        except json.JSONDecodeError:
                            continue

            # Берем выборку сообщений
            sample_messages = (
                messages[:sample_size] if len(messages) > sample_size else messages
            )

            # Анализируем токены
            all_tokens = []
            money_tokens = []
            amount_tokens = []
            value_tokens = []
            type_tokens = []
            russian_tokens = []
            english_tokens = []

            for message in sample_messages:
                text = message.get("text", "")
                if text:
                    tokens = tokenize_text(text)
                    all_tokens.extend(tokens)

                    # Классифицируем токены
                    for token in tokens:
                        if token.startswith("money_"):
                            money_tokens.append(token)
                        elif token.startswith("amount_"):
                            amount_tokens.append(token)
                        elif token.startswith("value_"):
                            value_tokens.append(token)
                        elif token in ["billion", "million", "thousand", "percentage"]:
                            type_tokens.append(token)
                        elif any(
                            c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                            for c in token.lower()
                        ):
                            russian_tokens.append(token)
                        elif token.isalpha():
                            english_tokens.append(token)

            # Статистика
            analysis = {
                "chat_name": chat_name,
                "total_messages": len(messages),
                "analyzed_messages": len(sample_messages),
                "total_tokens": len(all_tokens),
                "unique_tokens": len(set(all_tokens)),
                "token_statistics": {
                    "money_tokens": len(money_tokens),
                    "amount_tokens": len(amount_tokens),
                    "value_tokens": len(value_tokens),
                    "type_tokens": len(type_tokens),
                    "russian_tokens": len(russian_tokens),
                    "english_tokens": len(english_tokens),
                },
                "top_tokens": dict(
                    sorted(
                        [
                            (token, count)
                            for token, count in [
                                (token, all_tokens.count(token))
                                for token in set(all_tokens)
                            ]
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                ),
                "unique_money_tokens": list(set(money_tokens)),
                "unique_amount_tokens": list(set(amount_tokens)),
                "unique_value_tokens": list(set(value_tokens)),
                "unique_type_tokens": list(set(type_tokens)),
            }

            return json.dumps(analysis, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Ошибка при анализе чата: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def run(self):
        """Запуск MCP сервера"""
        logger.info("🚀 Запуск MCP сервера...")
        logger.info(f"📁 Путь к ChromaDB: {self.chroma_path}")
        logger.info(f"💬 Путь к чатам: {self.chats_path}")

        # Проверяем доступность данных
        if not self.chroma_path.exists():
            logger.warning(f"⚠️  ChromaDB не найдена: {self.chroma_path}")

        if not self.chats_path.exists():
            logger.warning(f"⚠️  Директория чатов не найдена: {self.chats_path}")

        async with stdio_server() as (read_stream, write_stream):
            logger.info("✅ MCP сервер запущен и готов к работе")
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )

    async def _get_metadata_for_sessions(self, results: list) -> dict:
        """Получение метаданных (Risks, Actions, Attachments) для сессий"""
        metadata = {}

        try:
            for result in results:
                session_id = result.get("session_id", "")
                chat = result.get("chat", "Unknown")

                if not session_id:
                    continue

                # Пытаемся найти файл отчёта сессии
                chat_dir_name = self._normalize_chat_name(chat)
                report_paths = [
                    self.artifacts_path
                    / "reports"
                    / chat_dir_name
                    / "sessions"
                    / f"{session_id}.md",
                    self.artifacts_path
                    / "reports"
                    / chat_dir_name
                    / "sessions"
                    / f"{session_id}-needs-review.md",
                ]

                report_path = None
                for path in report_paths:
                    if path.exists():
                        report_path = path
                        break

                if not report_path:
                    continue

                try:
                    with open(report_path, encoding="utf-8") as f:
                        content = f.read()

                    # Парсим метаданные из markdown файла
                    session_metadata = self._parse_session_metadata(content)
                    if session_metadata:
                        metadata[f"{chat}_{session_id}"] = {
                            "session_id": session_id,
                            "chat": chat,
                            "file_path": str(report_path),
                            **session_metadata,
                        }

                except Exception as e:
                    logger.warning(
                        f"Не удалось прочитать метаданные для сессии {session_id}: {e}"
                    )

        except Exception as e:
            logger.error(f"Ошибка при получении метаданных сессий: {e}", exc_info=True)

        return metadata

    def _parse_session_metadata(self, content: str) -> dict:
        """Парсинг метаданных из markdown файла сессии"""
        metadata = {}

        try:
            lines = content.split("\n")
            current_section = None
            current_data = []

            for line in lines:
                line = line.strip()

                # Определяем секции
                if line.startswith("## Actions"):
                    current_section = "actions"
                    current_data = []
                elif line.startswith("## Risks"):
                    current_section = "risks"
                    current_data = []
                elif line.startswith("## Attachments"):
                    current_section = "attachments"
                    current_data = []
                elif line.startswith("## Uncertainties"):
                    current_section = "uncertainties"
                    current_data = []
                elif line.startswith("## ") and current_section:
                    # Новая секция - сохраняем предыдущую
                    if current_data:
                        metadata[current_section] = current_data
                    current_section = None
                    current_data = []
                elif current_section and line and not line.startswith("#"):
                    # Добавляем данные в текущую секцию
                    current_data.append(line)

            # Сохраняем последнюю секцию
            if current_section and current_data:
                metadata[current_section] = current_data

        except Exception as e:
            logger.warning(f"Ошибка при парсинге метаданных: {e}")

        return metadata

    async def _get_artifacts_for_results(
        self, results: list, collection_name: str
    ) -> dict:
        """Получение артефактов для результатов поиска при глубоком поиске"""
        artifacts = {}

        try:
            # Собираем уникальные чаты и сессии из результатов
            chats = set()
            sessions = set()

            for result in results:
                chat = result.get("chat", "Unknown")
                chats.add(chat)

                if collection_name == "chat_sessions":
                    session_id = result.get("session_id", "")
                    if session_id:
                        sessions.add((chat, session_id))

            # Загружаем контексты чатов
            for chat in chats:
                try:
                    context_result = await self._read_chat_context(chat)
                    context_data = json.loads(context_result)
                    if "content" in context_data:
                        artifacts[f"context_{self._normalize_chat_name(chat)}"] = {
                            "type": "chat_context",
                            "chat": chat,
                            "content": context_data["content"],
                        }
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить контекст для чата {chat}: {e}"
                    )

            # Загружаем отчёты сессий
            for chat, session_id in sessions:
                try:
                    session_result = await self._read_session_report(chat, session_id)
                    session_data = json.loads(session_result)
                    if "content" in session_data:
                        artifacts[
                            f"session_{self._normalize_chat_name(chat)}_{session_id}"
                        ] = {
                            "type": "session_report",
                            "chat": chat,
                            "session_id": session_id,
                            "content": session_data["content"],
                        }
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить отчёт сессии {session_id} для чата {chat}: {e}"
                    )

        except Exception as e:
            logger.error(f"Ошибка при получении артефактов: {e}", exc_info=True)

        return artifacts

    async def _read_session_report(self, chat_name: str, session_id: str) -> str:
        """Чтение детального отчёта сессии из артефактов"""
        try:
            # Преобразуем имя чата в формат файловой системы
            chat_dir_name = self._normalize_chat_name(chat_name)
            report_path = (
                self.artifacts_path
                / "reports"
                / chat_dir_name
                / "sessions"
                / f"{session_id}.md"
            )

            if not report_path.exists():
                return json.dumps(
                    {
                        "error": f"Отчёт сессии {session_id} не найден для чата {chat_name}",
                        "path": str(report_path),
                    },
                    ensure_ascii=False,
                )

            with open(report_path, encoding="utf-8") as f:
                content = f.read()

            return json.dumps(
                {
                    "chat_name": chat_name,
                    "session_id": session_id,
                    "content": content,
                    "path": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"Ошибка при чтении отчёта сессии: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _read_chat_context(self, chat_name: str) -> str:
        """Чтение контекстного файла чата из артефактов"""
        try:
            # Ищем контекстный файл в разных местах
            context_paths = [
                self.artifacts_path / "chat_contexts" / f"{chat_name}_context.md",
                self.artifacts_path
                / "reports"
                / self._normalize_chat_name(chat_name)
                / f"{self._normalize_chat_name(chat_name)}_context.md",
            ]

            context_path = None
            for path in context_paths:
                if path.exists():
                    context_path = path
                    break

            if not context_path:
                return json.dumps(
                    {
                        "error": f"Контекстный файл не найден для чата {chat_name}",
                        "searched_paths": [str(p) for p in context_paths],
                    },
                    ensure_ascii=False,
                )

            with open(context_path, encoding="utf-8") as f:
                content = f.read()

            return json.dumps(
                {"chat_name": chat_name, "content": content, "path": str(context_path)},
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"Ошибка при чтении контекста чата: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _list_chat_artifacts(self, chat_name: str) -> str:
        """Получение списка доступных артефактов для чата"""
        try:
            chat_dir_name = self._normalize_chat_name(chat_name)
            artifacts = {
                "chat_name": chat_name,
                "normalized_name": chat_dir_name,
                "reports": {},
                "contexts": [],
                "sessions": [],
            }

            # Проверяем отчёты
            reports_dir = self.artifacts_path / "reports" / chat_dir_name
            if reports_dir.exists():
                # Основной отчёт
                main_report = reports_dir / f"{chat_dir_name}.md"
                if main_report.exists():
                    artifacts["reports"]["main"] = str(main_report)

                # Сессии
                sessions_dir = reports_dir / "sessions"
                if sessions_dir.exists():
                    for session_file in sessions_dir.glob("*.md"):
                        artifacts["sessions"].append(
                            {"session_id": session_file.stem, "path": str(session_file)}
                        )

                # Контекст
                context_file = reports_dir / f"{chat_dir_name}_context.md"
                if context_file.exists():
                    artifacts["contexts"].append(
                        {"type": "report_context", "path": str(context_file)}
                    )

            # Проверяем контексты в chat_contexts
            context_file = (
                self.artifacts_path / "chat_contexts" / f"{chat_name}_context.md"
            )
            if context_file.exists():
                artifacts["contexts"].append(
                    {"type": "chat_context", "path": str(context_file)}
                )

            return json.dumps(artifacts, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Ошибка при получении списка артефактов: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _normalize_chat_name(self, chat_name: str) -> str:
        """Нормализация имени чата для файловой системы"""
        # Заменяем пробелы и специальные символы на подчеркивания
        normalized = chat_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        normalized = "".join(c for c in normalized if c.isalnum() or c in "_-")
        normalized = normalized.lower()

        # Специальные случаи для известных чатов
        if normalized == "семья":
            return "semya"

        return normalized

    async def _index_chat(
        self,
        chat_name: str,
        force_full: bool = False,
        recent_days: int = 0,
        enable_clustering: bool = False,
        enable_smart_aggregation: bool = True,
        max_messages_per_group: int = 200,
        max_session_hours: int = 12,
        gap_minutes: int = 120,
    ) -> str:
        """
        Индексация конкретного чата с отслеживанием прогресса

        Args:
            chat_name: Название чата для индексации
            force_full: Полная пересборка всех артефактов
            recent_days: Количество последних дней для индексации (0 = все)
            enable_clustering: Включить кластеризацию сессий
            enable_smart_aggregation: Включить умную группировку
            max_messages_per_group: Максимальное количество сообщений в группе
            max_session_hours: Максимальная длительность сессии в часах
            gap_minutes: Максимальный разрыв между сообщениями в минутах

        Returns:
            JSON строка с результатами индексации
        """
        try:
            logger.info(f"🚀 Начало индексации чата: {chat_name}")

            # Проверяем существование чата
            chat_dir = None
            for d in self.chats_path.iterdir():
                if d.is_dir() and d.name == chat_name:
                    chat_dir = d
                    break

            if not chat_dir:
                return json.dumps(
                    {
                        "error": f"Чат '{chat_name}' не найден",
                        "available_chats": [
                            d.name for d in self.chats_path.iterdir() if d.is_dir()
                        ],
                    },
                    ensure_ascii=False,
                )

            # Проверяем наличие JSON файла
            json_file = chat_dir / "result.json"
            if not json_file.exists():
                json_file = chat_dir / "unknown.json"
            if not json_file.exists():
                return json.dumps(
                    {
                        "error": f"JSON файл не найден для чата '{chat_name}'",
                        "path": str(chat_dir),
                    },
                    ensure_ascii=False,
                )

            # Создаем индексатор
            indexer = TwoLevelIndexer(
                chroma_path=str(self.chroma_path),
                artifacts_path=str(self.artifacts_path),
                enable_clustering=enable_clustering,
                enable_smart_aggregation=enable_smart_aggregation,
                max_messages_per_group=max_messages_per_group,
                max_session_hours=max_session_hours,
                gap_minutes=gap_minutes,
            )

            # Запускаем индексацию
            logger.info("📊 Параметры индексации:")
            logger.info(f"   - Полная пересборка: {force_full}")
            logger.info(
                f"   - Последние дни: {recent_days if recent_days > 0 else 'все'}"
            )
            logger.info(f"   - Кластеризация: {enable_clustering}")
            logger.info(f"   - Умная группировка: {enable_smart_aggregation}")

            stats = await indexer.build_index(
                scope="chat",
                chat=chat_name,
                force_full=force_full,
                recent_days=recent_days,
            )

            # Формируем результат
            result = {
                "success": True,
                "chat_name": chat_name,
                "parameters": {
                    "force_full": force_full,
                    "recent_days": recent_days,
                    "enable_clustering": enable_clustering,
                    "enable_smart_aggregation": enable_smart_aggregation,
                    "max_messages_per_group": max_messages_per_group,
                    "max_session_hours": max_session_hours,
                    "gap_minutes": gap_minutes,
                },
                "statistics": stats,
                "artifacts_created": {
                    "reports_path": str(
                        self.artifacts_path
                        / "reports"
                        / self._normalize_chat_name(chat_name)
                    ),
                    "chroma_collections": [
                        "chat_sessions",
                        "chat_messages",
                        "chat_tasks",
                    ],
                },
                "message": f"Чат '{chat_name}' успешно проиндексирован",
            }

            logger.info(f"✅ Индексация чата '{chat_name}' завершена успешно")
            logger.info(f"   - Сессий: {stats.get('sessions_indexed', 0)}")
            logger.info(f"   - Сообщений: {stats.get('messages_indexed', 0)}")
            logger.info(f"   - Задач: {stats.get('tasks_indexed', 0)}")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(
                f"❌ Ошибка при индексации чата '{chat_name}': {e}", exc_info=True
            )
            return json.dumps(
                {
                    "success": False,
                    "chat_name": chat_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                ensure_ascii=False,
                indent=2,
            )
