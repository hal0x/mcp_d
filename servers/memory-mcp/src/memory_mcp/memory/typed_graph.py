#!/usr/bin/env python3
"""
Типизированный граф знаний

Вдохновлено HALv1 Memory 2.0 Typed Graph Memory.
Хранит узлы и рёбра разных типов в SQLite + NetworkX.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from ..utils.russian_tokenizer import STOP_WORDS, get_tokenizer, get_word_variants
from .graph_types import EdgeType, GraphEdge, GraphNode, GraphPath, GraphStats, NodeType

logger = logging.getLogger(__name__)


class TypedGraphMemory:
    """Типизированный граф знаний с персистентным хранилищем"""

    def __init__(self, db_path: str = "./data/memory_graph.db"):
        """
        Инициализация графа

        Args:
            db_path: Путь к SQLite базе данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # NetworkX граф для быстрых операций
        self.graph = nx.DiGraph()

        # SQLite для персистентности
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self._initialize_schema()
        self._load_graph_from_db()

        logger.info(f"Инициализирован TypedGraphMemory: {db_path}")

    def _initialize_schema(self):
        """Создание схемы БД"""
        cursor = self.conn.cursor()

        # Таблица узлов
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT,
                embedding BLOB,
                created_at TEXT,
                updated_at TEXT
            )
        """
        )

        # Таблица рёбер
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                properties TEXT,
                created_at TEXT,
                bidirectional INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """
        )

        # Индексы для быстрого поиска
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")

        # Таблица полнотекстового поиска
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS node_search
            USING fts5(
                node_id UNINDEXED,
                content,
                source,
                tags,
                entities,
                tokenize = 'unicode61'
            )
        """
        )

        self.conn.commit()
        logger.info("Схема БД инициализирована")

    def _load_graph_from_db(self):
        """Загрузка графа из БД в память (NetworkX)"""
        cursor = self.conn.cursor()

        # Загружаем узлы (включая эмбеддинги)
        cursor.execute("SELECT id, type, label, properties, embedding FROM nodes")
        for row in cursor.fetchall():
            props = json.loads(row["properties"]) if row["properties"] else {}
            node_attrs = {
                "node_type": row["type"],
                "label": row["label"],
                "properties": props,
                **props,
            }
            # Загружаем эмбеддинг, если есть
            if row["embedding"]:
                try:
                    embedding = json.loads(row["embedding"].decode())
                    node_attrs["embedding"] = embedding
                except Exception:
                    # Если не удалось распарсить, пропускаем
                    pass
            self.graph.add_node(row["id"], **node_attrs)

        # Загружаем рёбра
        cursor.execute(
            "SELECT source_id, target_id, type, weight, properties FROM edges"
        )
        for row in cursor.fetchall():
            props = json.loads(row["properties"]) if row["properties"] else {}
            self.graph.add_edge(
                row["source_id"],
                row["target_id"],
                edge_type=row["type"],
                weight=row["weight"],
                **props,
            )

        logger.info(
            f"Граф загружен: {self.graph.number_of_nodes()} узлов, "
            f"{self.graph.number_of_edges()} рёбер"
        )

        # Обновляем FTS индекс из существующих данных
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") in (NodeType.DOC_CHUNK, NodeType.DOC_CHUNK.value):
                props = data.get("properties", {})
                self._fts_refresh_doc(
                    node_id=node_id,
                    content=data.get("content", ""),
                    source=data.get("source", ""),
                    tags=props.get("tags", []),
                    entities=props.get("entities", []),
                    properties=props,
                )

    def add_node(self, node: GraphNode) -> bool:
        """
        Добавление узла в граф

        Args:
            node: Узел для добавления

        Returns:
            True если узел добавлен, False если уже существует
        """
        if node.id in self.graph:
            logger.warning(f"Узел {node.id} уже существует")
            return False

        try:
            # Добавляем в NetworkX
            node_data = node.dict(exclude={"embedding"})
            self.graph.add_node(node.id, **node_data)

            # Сохраняем в БД
            cursor = self.conn.cursor()

            embedding_bytes = (
                json.dumps(node.embedding).encode() if node.embedding else None
            )

            cursor.execute(
                """
                INSERT INTO nodes (id, type, label, properties, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    node.id,
                    node.type.value if isinstance(node.type, NodeType) else node.type,
                    node.label,
                    json.dumps(node.properties),
                    embedding_bytes,
                    node.created_at or datetime.now().isoformat(),
                    node.updated_at or datetime.now().isoformat(),
                ),
            )

            self.conn.commit()

            if node.type == NodeType.DOC_CHUNK:
                self._fts_refresh_doc(
                    node_id=node.id,
                    content=getattr(node, "content", ""),
                    source=getattr(node, "source", ""),
                    tags=node.properties.get("tags", []),
                    entities=node.properties.get("entities", []),
                    properties=node.properties,
                )

            logger.info(f"Добавлен узел: {node.id} ({node.type})")
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления узла {node.id}: {e}")
            self.conn.rollback()
            return False

    def update_node(
        self,
        node_id: str,
        *,
        properties: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Обновление свойств узла

        Args:
            node_id: ID узла для обновления
            properties: Новые свойства (объединяются с существующими)
            content: Новый контент (для DOC_CHUNK)
            source: Новый источник
            tags: Новые теги
            entities: Новые сущности
            embedding: Новый эмбеддинг

        Returns:
            True если узел обновлён
        """
        if node_id not in self.graph:
            logger.error(f"Узел {node_id} не найден")
            return False

        try:
            cursor = self.conn.cursor()
            
            # Получаем текущие данные узла
            cursor.execute(
                "SELECT type, properties, embedding FROM nodes WHERE id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            if not row:
                logger.error(f"Узел {node_id} не найден в БД")
                return False

            node_type = row["type"]
            current_props = json.loads(row["properties"]) if row["properties"] else {}
            current_embedding = (
                json.loads(row["embedding"].decode()) if row["embedding"] else None
            )

            # Объединяем свойства
            updated_props = dict(current_props)
            if properties:
                updated_props.update(properties)
            
            # Обновляем специфичные поля
            if content is not None:
                updated_props["content"] = content
            if source is not None:
                updated_props["source"] = source
            if tags is not None:
                updated_props["tags"] = tags
            if entities is not None:
                updated_props["entities"] = entities

            # Обновляем эмбеддинг
            new_embedding = embedding if embedding is not None else current_embedding
            embedding_bytes = (
                json.dumps(new_embedding).encode() if new_embedding else None
            )

            # Обновляем в БД
            cursor.execute(
                """
                UPDATE nodes 
                SET properties = ?, embedding = ?, updated_at = ?
                WHERE id = ?
            """,
                (
                    json.dumps(updated_props),
                    embedding_bytes,
                    datetime.now().isoformat(),
                    node_id,
                ),
            )

            # Обновляем в NetworkX графе
            if node_id in self.graph:
                self.graph.nodes[node_id]["properties"] = updated_props
                if content is not None:
                    self.graph.nodes[node_id]["content"] = content
                if source is not None:
                    self.graph.nodes[node_id]["source"] = source
                if new_embedding is not None:
                    self.graph.nodes[node_id]["embedding"] = new_embedding

            self.conn.commit()

            # Обновляем FTS индекс для DOC_CHUNK
            if node_type == NodeType.DOC_CHUNK.value or node_type == NodeType.DOC_CHUNK:
                final_content = content or updated_props.get("content", "")
                final_source = source or updated_props.get("source", "")
                final_tags = tags or updated_props.get("tags", [])
                final_entities = entities or updated_props.get("entities", [])
                
                self._fts_refresh_doc(
                    node_id=node_id,
                    content=final_content,
                    source=final_source,
                    tags=final_tags,
                    entities=final_entities,
                    properties=updated_props,
                )

            logger.info(f"Обновлён узел: {node_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка обновления узла {node_id}: {e}")
            self.conn.rollback()
            return False

    def delete_node(self, node_id: str) -> bool:
        """
        Удаление узла из графа

        Args:
            node_id: ID узла для удаления

        Returns:
            True если узел удалён
        """
        if node_id not in self.graph:
            logger.warning(f"Узел {node_id} не найден в графе")
            return False

        try:
            cursor = self.conn.cursor()
            
            # Проверяем существование узла в БД
            cursor.execute("SELECT id FROM nodes WHERE id = ?", (node_id,))
            if not cursor.fetchone():
                logger.warning(f"Узел {node_id} не найден в БД")
                return False

            # Удаляем из FTS индекса
            cursor.execute("DELETE FROM node_search WHERE node_id = ?", (node_id,))

            # Удаляем все рёбра, связанные с узлом
            # Сначала удаляем исходящие рёбра
            cursor.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )

            # Удаляем узел из БД
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

            # Удаляем из NetworkX графа (это также удалит все связанные рёбра)
            if node_id in self.graph:
                self.graph.remove_node(node_id)

            self.conn.commit()
            logger.info(f"Удалён узел: {node_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id}: {e}")
            self.conn.rollback()
            return False

    def delete_nodes_by_chat(self, chat_name: str) -> int:
        """
        Удаление всех узлов конкретного чата из графа.

        Args:
            chat_name: Название чата для удаления

        Returns:
            Количество удалённых узлов
        """
        deleted_count = 0
        try:
            cursor = self.conn.cursor()

            # Находим все узлы, связанные с чатом
            # 1. Поиск через FTS индекс по tags
            node_ids_from_fts = set()
            try:
                # FTS поиск по тегам (tags хранятся как строка с пробелами)
                fts_rows = cursor.execute(
                    "SELECT node_id FROM node_search WHERE tags MATCH ?",
                    (chat_name,),
                ).fetchall()
                node_ids_from_fts.update(row["node_id"] for row in fts_rows)
            except Exception as e:
                logger.debug(f"FTS поиск по тегам не удался: {e}")

            # 2. Поиск через properties в nodes (chat в metadata или tags в списке)
            node_ids_from_props = set()
            try:
                # Ищем узлы, где chat_name входит в tags или metadata.chat
                # Используем JSON функции SQLite для поиска в JSON
                all_nodes = cursor.execute(
                    "SELECT id, properties FROM nodes WHERE properties IS NOT NULL"
                ).fetchall()

                for row in all_nodes:
                    try:
                        props = json.loads(row["properties"]) if row["properties"] else {}
                        # Проверяем metadata.chat
                        metadata = props.get("metadata", {})
                        if isinstance(metadata, dict) and metadata.get("chat") == chat_name:
                            node_ids_from_props.add(row["id"])
                        # Проверяем tags (может быть список или строка)
                        tags = props.get("tags", [])
                        if isinstance(tags, list) and chat_name in tags:
                            node_ids_from_props.add(row["id"])
                        elif isinstance(tags, str) and chat_name in tags:
                            node_ids_from_props.add(row["id"])
                        # Проверяем chat напрямую в properties
                        if props.get("chat") == chat_name:
                            node_ids_from_props.add(row["id"])
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Ошибка парсинга properties для узла {row['id']}: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Поиск по properties не удался: {e}")

            # 3. Поиск по record_id (формат: source:chat_name:record_id)
            node_ids_from_id = set()
            try:
                id_pattern = f"%:{chat_name}:%"
                id_rows = cursor.execute(
                    "SELECT id FROM nodes WHERE id LIKE ?",
                    (id_pattern,),
                ).fetchall()
                node_ids_from_id.update(row["id"] for row in id_rows)
            except Exception as e:
                logger.debug(f"Поиск по ID не удался: {e}")

            # Объединяем все найденные ID
            all_node_ids = node_ids_from_fts | node_ids_from_props | node_ids_from_id

            logger.info(
                f"Найдено {len(all_node_ids)} узлов для удаления (чат: {chat_name})"
            )

            # Удаляем каждый узел (delete_node уже удаляет рёбра и FTS записи)
            for node_id in all_node_ids:
                if self.delete_node(node_id):
                    deleted_count += 1

            logger.info(f"Удалено {deleted_count} узлов для чата {chat_name}")
            return deleted_count

        except Exception as e:
            logger.error(f"Ошибка удаления узлов чата {chat_name}: {e}", exc_info=True)
            self.conn.rollback()
            return deleted_count

    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Добавление ребра в граф

        Args:
            edge: Ребро для добавления

        Returns:
            True если ребро добавлено
        """
        # Проверяем существование узлов
        if edge.source_id not in self.graph or edge.target_id not in self.graph:
            logger.error(
                f"Узлы для ребра не существуют: {edge.source_id} -> {edge.target_id}"
            )
            return False

        try:
            # Добавляем в NetworkX
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.type.value
                if isinstance(edge.type, EdgeType)
                else edge.type,
                weight=edge.weight,
                **edge.properties,
            )

            # Добавляем обратное ребро если bidirectional
            if edge.bidirectional:
                self.graph.add_edge(
                    edge.target_id,
                    edge.source_id,
                    edge_type=edge.type.value
                    if isinstance(edge.type, EdgeType)
                    else edge.type,
                    weight=edge.weight,
                    **edge.properties,
                )

            # Сохраняем в БД
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO edges (id, source_id, target_id, type, weight, properties, created_at, bidirectional)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.type.value if isinstance(edge.type, EdgeType) else edge.type,
                    edge.weight,
                    json.dumps(edge.properties),
                    edge.created_at or datetime.now().isoformat(),
                    1 if edge.bidirectional else 0,
                ),
            )

            self.conn.commit()
            logger.info(
                f"Добавлено ребро: {edge.source_id} -> {edge.target_id} ({edge.type})"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления ребра {edge.id}: {e}")
            self.conn.rollback()
            return False

    # ------------------------------------------------------------------
    def _fts_refresh_doc(
        self,
        *,
        node_id: str,
        content: str,
        source: str,
        tags: Iterable[str],
        entities: Iterable[str],
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or update document chunk entry in FTS index.
        Включает метаданные в поле content для полнотекстового поиска.
        """
        # Извлекаем метаданные из properties или из узла графа
        metadata_parts = []
        if properties is None and node_id in self.graph:
            properties = self.graph.nodes[node_id].get("properties", {})
        
        if properties:
            # Добавляем username отправителя, если есть
            sender_username = properties.get("sender_username")
            if sender_username:
                metadata_parts.append(f"@{sender_username}")
            elif properties.get("author"):
                metadata_parts.append(properties.get("author"))
            
            # Добавляем реакции, если есть
            reactions = properties.get("reactions")
            if reactions and isinstance(reactions, list) and len(reactions) > 0:
                reaction_strs = []
                for reaction in reactions:
                    if isinstance(reaction, dict):
                        emoji = reaction.get("emoji", "")
                        count = reaction.get("count", 0)
                        if emoji and count > 0:
                            # Извлекаем emoji из строки вида "ReactionEmoji(emoticon='👍')"
                            if "emoticon=" in str(emoji):
                                try:
                                    emoji_value = str(emoji).split("emoticon=")[1].split("'")[1]
                                    reaction_strs.append(f"{emoji_value} x{count}")
                                except (IndexError, AttributeError):
                                    reaction_strs.append(f"{emoji} x{count}")
                            else:
                                reaction_strs.append(f"{emoji} x{count}")
                if reaction_strs:
                    metadata_parts.append(f"Реакции: {', '.join(reaction_strs)}")
            
            # Добавляем информацию о редактировании, если есть
            edited_utc = properties.get("edited_utc")
            if edited_utc:
                metadata_parts.append(f"Отредактировано: {edited_utc}")

        # Формируем расширенный контент для FTS5
        extended_content = content or ""
        
        # Добавляем нормализованные формы слов для улучшения поиска
        if extended_content:
            try:
                tokenizer = get_tokenizer()
                # Токенизируем контент
                tokens = tokenizer.tokenize(extended_content)
                # Собираем нормализованные формы
                normalized_forms = []
                for token in tokens:
                    # Получаем варианты слова (исходное + нормальная форма)
                    variants = get_word_variants(token)
                    # Добавляем нормализованные формы, отличные от исходного слова
                    for variant in variants:
                        if variant != token.lower():
                            normalized_forms.append(variant)
                
                # Добавляем нормализованные формы к контенту
                if normalized_forms:
                    extended_content = f"{extended_content} {' '.join(normalized_forms)}"
            except Exception as e:
                # Если морфологическая обработка не удалась, продолжаем без неё
                logger.debug(f"Ошибка при добавлении нормализованных форм для {node_id}: {e}")
        
        if metadata_parts:
            extended_content = f"{extended_content}\n{' '.join(metadata_parts)}"

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM node_search WHERE node_id = ?", (node_id,))
        cursor.execute(
            """
            INSERT INTO node_search (node_id, content, source, tags, entities)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                node_id,
                extended_content,
                source or "",
                " ".join(sorted({str(tag) for tag in tags if tag})),
                " ".join(sorted({str(entity) for entity in entities if entity})),
            ),
        )
        self.conn.commit()

    def _prepare_match_expression(self, query: str) -> str:
        """Подготовка выражения для FTS5 поиска с морфологическим расширением.
        
        Использует морфологию для расширения запросов нормальными формами слов.
        Для каждого слова добавляет его варианты (исходное + нормальная форма).
        Игнорирует стоп-слова (предлоги, союзы, местоимения).
        
        Для коротких запросов (<=3 значимых слова) использует AND логику.
        Для длинных запросов (>3 значимых слова) использует гибкую логику:
        - Первые 2-3 слова через AND (обязательные)
        - Остальные слова через OR (опциональные)
        Это позволяет находить релевантные записи даже если некоторые слова отсутствуют.
        """
        # Токенизируем запрос с помощью russian_tokenizer
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize(query)
        
        if not tokens:
            # Fallback на простую токенизацию, если russian_tokenizer не работает
            tokens = [token.strip().lower() for token in query.replace('"', "").split() if token.strip()]
            if not tokens:
                return ""
        
        # Фильтруем стоп-слова и короткие токены (меньше 2 символов)
        # Используем готовый список стоп-слов из russian_tokenizer
        significant_tokens = [
            token for token in tokens
            if token.lower() not in STOP_WORDS and len(token) >= 2
        ]
        
        # Если после фильтрации не осталось значимых токенов, используем все токены
        if not significant_tokens:
            significant_tokens = tokens
        
        # Расширяем каждый токен его вариантами (исходное слово + нормальная форма)
        expanded_tokens = []
        for token in significant_tokens:
            variants = get_word_variants(token)
            if len(variants) > 1:
                # Если есть варианты, создаем OR выражение
                variants_list = " OR ".join(f'"{v}"' for v in sorted(variants))
                expanded_tokens.append(f"({variants_list})")
            else:
                # Если вариантов нет, используем исходное слово
                expanded_tokens.append(f'"{token}"')
        
        if not expanded_tokens:
            return ""
        
        if len(expanded_tokens) == 1:
            return expanded_tokens[0]
        
        # Гибкая логика для длинных запросов
        if len(expanded_tokens) > 3:
            # Для длинных запросов: первые 2-3 слова через AND, остальные через OR
            # Это позволяет находить релевантные записи даже если некоторые слова отсутствуют
            and_tokens = expanded_tokens[:2]  # Первые 2 слова обязательны
            or_tokens = expanded_tokens[2:]   # Остальные опциональны
            
            and_expr = " AND ".join(and_tokens)
            or_expr = " OR ".join(or_tokens)
            
            # Комбинируем: (AND часть) AND (OR часть)
            return f"({and_expr}) AND ({or_expr})"
        else:
            # Для коротких запросов (<=3 слова) используем строгую AND логику
            return " AND ".join(expanded_tokens)

    def _get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        if node_id in self.graph:
            return self.graph.nodes[node_id]
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT properties FROM nodes WHERE id = ?",
            (node_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}
        props = json.loads(row["properties"]) if row["properties"] else {}
        return {"properties": props, **props}

    def search_text(
        self,
        query: str,
        *,
        limit: int = 5,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search doc chunks using FTS with optional filters."""

        if not query.strip():
            return [], 0

        match_expr = self._prepare_match_expression(query)
        if not match_expr:
            return [], 0

        cursor = self.conn.cursor()
        fetch_limit = max(limit * 5, 50)
        rows = cursor.execute(
            """
            SELECT node_id,
                   content,
                   snippet(node_search, 1, '<b>', '</b>', ' … ', 15) AS snippet,
                   bm25(node_search) AS score,
                   source,
                   tags,
                   entities
            FROM node_search
            WHERE node_search MATCH ?
            ORDER BY score
            LIMIT ?
        """,
            (match_expr, fetch_limit),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        # Собираем все BM25 scores для нормализации
        bm25_scores = []
        for row in rows:
            bm25_score = row["score"] if row["score"] is not None else 0.0
            bm25_scores.append(bm25_score)
        
        # Вычисляем min/max для нормализации
        if bm25_scores:
            min_bm25 = min(bm25_scores)
            max_bm25 = max(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
        else:
            min_bm25 = 0.0
            max_bm25 = 0.0
            bm25_range = 1.0
        
        result_count = 0  # Счетчик добавленных результатов (после фильтрации)
        for idx, row in enumerate(rows):
            node_id = row["node_id"]
            attrs = self._get_node_attrs(node_id)
            props = attrs.get("properties") if isinstance(attrs.get("properties"), dict) else {}
            node_source = (
                row["source"]
                or props.get("source")
                or attrs.get("source")
                or props.get("chat")
                or attrs.get("chat")
            )
            if source and node_source != source:
                continue

            node_tags = set(props.get("tags", attrs.get("tags", [])))
            if tags and not set(tags).issubset(node_tags):
                continue

            timestamp_raw = (
                props.get("timestamp")
                or attrs.get("timestamp")
                or props.get("created_at")
                or attrs.get("created_at")
            )
            timestamp_dt = None
            if timestamp_raw:
                try:
                    if isinstance(timestamp_raw, (int, float)):
                        timestamp_dt = datetime.fromtimestamp(float(timestamp_raw), tz=timezone.utc)
                    else:
                        ts = str(timestamp_raw)
                        from ..utils.datetime_utils import parse_datetime_utc

                        timestamp_dt = parse_datetime_utc(ts, return_none_on_error=True, use_zoneinfo=True)
                except Exception:
                    timestamp_dt = None

            if date_from and timestamp_dt and timestamp_dt < date_from:
                continue
            if date_to and timestamp_dt and timestamp_dt > date_to:
                continue

            # BM25 score нормализуем: отрицательные значения (лучше релевантность) -> выше score
            # BM25 обычно возвращает отрицательные значения, где более отрицательные = лучше
            bm25_score = row["score"] if row["score"] is not None else 0.0
            
            # Нормализуем BM25 с учетом диапазона scores для лучшего различения
            if bm25_range > 0.001:
                # Есть различия в scores - используем нормализацию по диапазону
                if bm25_score < 0:
                    # Отрицательные значения: более отрицательные = лучше
                    # Нормализуем к [0.5, 1.0] с учетом позиции в диапазоне
                    normalized = (bm25_score - min_bm25) / bm25_range
                    score = 0.5 + (1.0 - normalized) * 0.5  # Инвертируем, так как меньшие значения лучше
                else:
                    # Положительные значения (редко)
                    normalized = (bm25_score - min_bm25) / bm25_range
                    score = 0.5 + normalized * 0.5
            else:
                # Все scores одинаковые - используем позицию для различения
                # FTS5 уже отсортировал результаты, первые результаты более релевантны
                # Используем индекс в исходном списке для различения
                # Нормализуем индекс к [0, 1] и используем для создания различий в scores
                if len(rows) > 1:
                    position_factor = 1.0 - (idx / (len(rows) - 1))  # От 1.0 (первый) до 0.0 (последний)
                else:
                    position_factor = 1.0  # Если только один результат
                score = 0.5 + position_factor * 0.2  # Score от 0.7 (первый) до 0.5 (последний)

            snippet = row["snippet"] or ""
            content = row["content"] or props.get("content") or attrs.get("content") or ""

            results.append(
                {
                    "node_id": node_id,
                    "score": score,
                    "snippet": snippet if snippet.strip() else content[:200],
                    "content": content,
                    "source": node_source or "unknown",
                    "timestamp": timestamp_dt or datetime.now(timezone.utc),
                    "author": props.get("author") or attrs.get("author"),
                    "metadata": dict(props) if props else dict(attrs),
                }
            )
            result_count += 1  # Увеличиваем счетчик после добавления результата

        total = len(results)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit], total

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Получение узла по ID

        Args:
            node_id: ID узла

        Returns:
            Узел или None
        """
        if node_id not in self.graph:
            return None

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, type, label, properties, created_at, updated_at FROM nodes WHERE id = ?",
            (node_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        props = json.loads(row["properties"]) if row["properties"] else {}

        return GraphNode(
            id=row["id"],
            type=NodeType(row["type"]),
            label=row["label"],
            properties=props,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_nodes_by_type(
        self, node_type: NodeType, limit: int = 100
    ) -> List[GraphNode]:
        """
        Получение узлов по типу

        Args:
            node_type: Тип узлов
            limit: Максимум результатов

        Returns:
            Список узлов
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, type, label, properties, created_at, updated_at
            FROM nodes
            WHERE type = ?
            LIMIT ?
        """,
            (node_type.value, limit),
        )

        nodes = []
        for row in cursor.fetchall():
            props = json.loads(row["properties"]) if row["properties"] else {}
            nodes.append(
                GraphNode(
                    id=row["id"],
                    type=NodeType(row["type"]),
                    label=row["label"],
                    properties=props,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )

        return nodes

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Получение соседних узлов

        Args:
            node_id: ID узла
            edge_type: Фильтр по типу рёбер
            direction: Направление ("out", "in", "both")

        Returns:
            Список (neighbor_id, edge_data)
        """
        if node_id not in self.graph:
            return []

        neighbors = []

        if direction in ("out", "both"):
            for neighbor in self.graph.successors(node_id):
                edge_data = self.graph[node_id][neighbor]
                if edge_type is None or edge_data.get("edge_type") == edge_type.value:
                    neighbors.append((neighbor, edge_data))

        if direction in ("in", "both"):
            for neighbor in self.graph.predecessors(node_id):
                edge_data = self.graph[neighbor][node_id]
                if edge_type is None or edge_data.get("edge_type") == edge_type.value:
                    neighbors.append((neighbor, edge_data))

        return neighbors

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> Optional[GraphPath]:
        """
        Поиск кратчайшего пути между узлами

        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            max_length: Максимальная длина пути

        Returns:
            Путь или None
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None

        try:
            path_nodes = nx.shortest_path(
                self.graph, source=source_id, target=target_id, weight="weight"
            )

            if len(path_nodes) > max_length + 1:
                return None

            # Собираем узлы и рёбра
            nodes = [self.get_node(nid) for nid in path_nodes]
            nodes = [n for n in nodes if n is not None]

            edges = []
            total_weight = 0.0

            for i in range(len(path_nodes) - 1):
                src = path_nodes[i]
                tgt = path_nodes[i + 1]
                edge_data = self.graph[src][tgt]

                edges.append(
                    GraphEdge(
                        id=f"{src}-{tgt}",
                        source_id=src,
                        target_id=tgt,
                        type=EdgeType(edge_data.get("edge_type", "relates_to")),
                        weight=edge_data.get("weight", 1.0),
                        properties=edge_data,
                    )
                )

                total_weight += edge_data.get("weight", 1.0)

            return GraphPath(
                nodes=nodes,
                edges=edges,
                length=len(path_nodes) - 1,
                total_weight=total_weight,
            )

        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Ошибка поиска пути: {e}")
            return None

    def get_stats(self) -> GraphStats:
        """
        Получение статистики графа

        Returns:
            Статистика
        """
        node_types_count = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get("type", "unknown")
            node_types_count[node_type] = node_types_count.get(node_type, 0) + 1

        edge_types_count = {}
        for _, _, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get("edge_type", "unknown")
            edge_types_count[edge_type] = edge_types_count.get(edge_type, 0) + 1

        # Средняя степень узла
        total_degree = sum(dict(self.graph.degree()).values())
        avg_degree = (
            total_degree / self.graph.number_of_nodes()
            if self.graph.number_of_nodes() > 0
            else 0
        )

        # Плотность
        n = self.graph.number_of_nodes()
        max_edges = n * (n - 1) if n > 1 else 1
        density = self.graph.number_of_edges() / max_edges if max_edges > 0 else 0

        # Компоненты связности
        connected_components = nx.number_weakly_connected_components(self.graph)

        return GraphStats(
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            node_types_count=node_types_count,
            edge_types_count=edge_types_count,
            avg_node_degree=avg_degree,
            density=density,
            connected_components=connected_components,
        )

    def close(self):
        """Закрытие соединения с БД"""
        self.conn.close()
        logger.info("Соединение с БД закрыто")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
