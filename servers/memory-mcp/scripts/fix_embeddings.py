#!/usr/bin/env python3
"""Скрипт для исправления проблем с эмбеддингами в графе памяти."""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from memory_mcp.memory.typed_graph import TypedGraphMemory
from memory_mcp.memory.embeddings import build_embedding_service_from_env
from memory_mcp.memory.vector_store import build_vector_store_from_env
from memory_mcp.indexing import MemoryRecord
from memory_mcp.memory.ingest import MemoryIngestor
from memory_mcp.utils.datetime_utils import parse_datetime_utc
from datetime import datetime, timezone
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_embeddings_for_chat(chat_name: str, db_path: str = "data/memory_graph.db", chroma_path: str = "chroma_db"):
    """Исправляет эмбеддинги для записей из указанного чата."""
    
    # Инициализация
    graph = TypedGraphMemory(db_path=db_path)
    ingestor = MemoryIngestor(graph)
    embedding_service = build_embedding_service_from_env()
    vector_store = build_vector_store_from_env()
    
    if not embedding_service:
        logger.error("Сервис эмбеддингов недоступен!")
        return
    
    if vector_store and embedding_service.dimension:
        vector_store.ensure_collection(embedding_service.dimension)
        logger.info("✅ Векторное хранилище инициализировано")
    
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    messages = chroma_client.get_collection('chat_messages')
    
    # Получаем все записи из чата
    result = messages.get(
        where={'chat': chat_name},
        include=['documents', 'metadatas', 'embeddings']
    )
    
    logger.info(f"Найдено {len(result['ids'])} записей в ChromaDB для чата '{chat_name}'")
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, record_id in enumerate(result['ids']):
        try:
            # Проверяем, есть ли узел в графе
            if record_id not in graph.graph:
                logger.debug(f"Узел {record_id} не найден в графе, пропускаем")
                skipped_count += 1
                continue
            
            # Получаем эмбеддинг из ChromaDB
            embedding = None
            embeddings_list = result.get('embeddings')
            if embeddings_list is not None and i < len(embeddings_list):
                embedding = embeddings_list[i]
            
            # Проверяем, есть ли эмбеддинг в графе
            node_data = graph.graph.nodes[record_id]
            existing_emb = node_data.get('embedding')
            has_embedding = False
            if existing_emb is not None:
                try:
                    # Проверяем, что эмбеддинг валидный (не пустой)
                    if hasattr(existing_emb, '__len__'):
                        try:
                            has_embedding = len(existing_emb) > 0
                        except (TypeError, ValueError):
                            # Для numpy массивов используем другой способ проверки
                            has_embedding = True
                    else:
                        has_embedding = True
                except (TypeError, ValueError):
                    has_embedding = False
            
            if has_embedding:
                logger.debug(f"Узел {record_id} уже имеет эмбеддинг, пропускаем")
                skipped_count += 1
                continue
            
            # Если эмбеддинг есть в ChromaDB, сохраняем его в граф
            if embedding is not None:
                try:
                    # Проверяем, что эмбеддинг не пустой
                    emb_len = len(embedding) if hasattr(embedding, '__len__') else 0
                    if emb_len == 0:
                        embedding = None
                except (TypeError, ValueError):
                    # Если не можем проверить длину, считаем что эмбеддинг валидный
                    pass
            
            if embedding is not None:
                try:
                    # Преобразуем numpy массив в список
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)
                    
                    # Сохраняем в граф
                    graph.update_node(record_id, embedding=embedding)
                    
                    # Сохраняем в Qdrant, если доступен
                    if vector_store:
                        metadata = result['metadatas'][i] if result.get('metadatas') else {}
                        payload_data = {
                            "record_id": record_id,
                            "source": metadata.get("chat", chat_name),
                            "tags": metadata.get("tags", []),
                            "timestamp": metadata.get("date_utc", datetime.now(timezone.utc).isoformat()),
                            "timestamp_iso": metadata.get("date_utc", datetime.now(timezone.utc).isoformat()),
                            "content_preview": result['documents'][i][:200] if result.get('documents') else "",
                        }
                        if chat_name:
                            payload_data["chat"] = chat_name
                        
                        try:
                            vector_store.upsert(record_id, embedding, payload_data)
                        except Exception as e:
                            logger.debug(f"Ошибка при сохранении в Qdrant для {record_id}: {e}")
                    
                    fixed_count += 1
                    if fixed_count % 100 == 0:
                        logger.info(f"Исправлено {fixed_count} записей...")
                        
                except Exception as e:
                    logger.warning(f"Ошибка при сохранении эмбеддинга для {record_id}: {e}")
                    error_count += 1
            else:
                # Генерируем эмбеддинг, если его нет
                doc = result['documents'][i] if result.get('documents') else ""
                if doc:
                    try:
                        embedding = embedding_service.embed(doc)
                        if embedding is not None:
                            try:
                                # Проверяем, что эмбеддинг не пустой
                                emb_len = len(embedding) if hasattr(embedding, '__len__') else 0
                                if emb_len > 0:
                                    # Преобразуем numpy массив в список, если нужно
                                    if hasattr(embedding, 'tolist'):
                                        embedding = embedding.tolist()
                                    elif not isinstance(embedding, list):
                                        embedding = list(embedding)
                                    
                                    graph.update_node(record_id, embedding=embedding)
                                    fixed_count += 1
                                    if fixed_count % 100 == 0:
                                        logger.info(f"Сгенерировано и исправлено {fixed_count} записей...")
                                else:
                                    skipped_count += 1
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Ошибка при проверке эмбеддинга для {record_id}: {e}")
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.warning(f"Ошибка при генерации эмбеддинга для {record_id}: {e}")
                        error_count += 1
                else:
                    skipped_count += 1
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке записи {record_id}: {e}")
            error_count += 1
    
    logger.info(f"✅ Завершено: исправлено {fixed_count}, пропущено {skipped_count}, ошибок {error_count}")
    
    graph.conn.close()
    if vector_store:
        vector_store.close()
    if embedding_service:
        embedding_service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Исправление эмбеддингов в графе памяти")
    parser.add_argument("--chat", required=True, help="Название чата")
    parser.add_argument("--db-path", default="data/memory_graph.db", help="Путь к БД графа")
    parser.add_argument("--chroma-path", default="chroma_db", help="Путь к ChromaDB")
    args = parser.parse_args()
    
    fix_embeddings_for_chat(args.chat, args.db_path, args.chroma_path)

