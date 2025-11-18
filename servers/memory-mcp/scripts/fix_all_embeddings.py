#!/usr/bin/env python3
"""Скрипт для генерации эмбеддингов для всех узлов без эмбеддингов."""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from memory_mcp.memory.typed_graph import TypedGraphMemory
from memory_mcp.memory.embeddings import build_embedding_service_from_env
from memory_mcp.memory.vector_store import build_vector_store_from_env
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_all_missing_embeddings(db_path: str = "data/memory_graph.db", chroma_path: str = "chroma_db"):
    """Генерирует эмбеддинги для всех узлов без эмбеддингов."""
    
    # Инициализация
    graph = TypedGraphMemory(db_path=db_path)
    embedding_service = build_embedding_service_from_env()
    vector_store = build_vector_store_from_env()
    
    if not embedding_service:
        logger.error("Сервис эмбеддингов недоступен!")
        return
    
    if vector_store and embedding_service.dimension:
        vector_store.ensure_collection(embedding_service.dimension)
        logger.info("✅ Векторное хранилище инициализировано")
    
    # ChromaDB
    chroma_client = None
    messages_collection = None
    sessions_collection = None
    tasks_collection = None
    
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        messages_collection = chroma_client.get_collection('chat_messages')
        sessions_collection = chroma_client.get_collection('chat_sessions')
        tasks_collection = chroma_client.get_collection('chat_tasks')
        logger.info("✅ ChromaDB подключен")
    except Exception as e:
        logger.warning(f"ChromaDB недоступен: {e}")
    
    # Находим узлы без эмбеддингов
    cursor = graph.conn.cursor()
    cursor.execute("SELECT id, type, properties FROM nodes WHERE embedding IS NULL OR embedding = ''")
    rows = cursor.fetchall()
    
    logger.info(f"Найдено {len(rows)} узлов без эмбеддингов")
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for row in rows:
        record_id = row['id']
        node_type = row['type']
        props = json.loads(row['properties']) if row['properties'] else {}
        
        try:
            # Пытаемся найти эмбеддинг в ChromaDB
            embedding = None
            content = None
            
            if chroma_client:
                # Ищем в коллекциях
                for collection in [messages_collection, sessions_collection, tasks_collection]:
                    if collection is None:
                        continue
                    try:
                        # Пробуем найти по точному совпадению ID
                        result = collection.get(ids=[record_id], include=["documents", "embeddings"])
                        if result.get("ids") and len(result["ids"]) > 0:
                            idx = result["ids"].index(record_id)
                            if result.get("embeddings") and idx < len(result["embeddings"]):
                                embedding = result["embeddings"][idx]
                            if result.get("documents") and idx < len(result["documents"]):
                                content = result["documents"][idx]
                            break
                        
                        # Если не найдено по точному совпадению, пробуем найти по частичному совпадению
                        # Для record_id типа "telegram:Семья:257859" ищем в метаданных
                        if ":" in record_id:
                            parts = record_id.split(":")
                            if len(parts) >= 3:
                                try:
                                    msg_id = int(parts[-1])
                                    chat_name = parts[1]
                                    # Ищем по метаданным
                                    where_filter = {"chat": chat_name}
                                    result = collection.get(where=where_filter, include=["documents", "embeddings", "metadatas"])
                                    if result.get("ids"):
                                        # Ищем запись с похожим ID в метаданных
                                        for idx, meta in enumerate(result.get("metadatas", [])):
                                            if meta and meta.get("msg_id") == str(msg_id):
                                                if result.get("embeddings") and idx < len(result["embeddings"]):
                                                    embedding = result["embeddings"][idx]
                                                if result.get("documents") and idx < len(result["documents"]):
                                                    content = result["documents"][idx]
                                                break
                                except (ValueError, KeyError):
                                    pass
                    except (ValueError, IndexError):
                        continue
                    except Exception as e:
                        logger.debug(f"Ошибка при поиске в ChromaDB для {record_id}: {e}")
                        continue
            
            # Если эмбеддинг не найден в ChromaDB, генерируем его
            if embedding is None:
                # Получаем контент для генерации эмбеддинга
                if not content:
                    # Пытаемся получить контент из свойств узла
                    if record_id in graph.graph:
                        node_data = graph.graph.nodes[record_id]
                        content = node_data.get("content") or props.get("content", "")
                    else:
                        content = props.get("content", "")
                
                # Для TradingSignal и Entity генерируем эмбеддинг на основе метаданных
                if not content or len(content.strip()) == 0:
                    if node_type == "TradingSignal":
                        # Формируем текст из метаданных торгового сигнала
                        symbol = props.get("symbol", "")
                        signal_type = props.get("signal_type", "")
                        direction = props.get("direction", "")
                        content = f"Trading signal: {symbol} {signal_type} {direction}"
                    elif node_type == "Entity":
                        # Формируем текст из метаданных сущности
                        entity_type = props.get("entity_type", "")
                        entity_value = props.get("value", record_id)
                        content = f"Entity: {entity_type} {entity_value}"
                    else:
                        # Для других типов используем ID как fallback
                        content = f"Node {record_id} of type {node_type}"
                
                if content and len(content.strip()) > 0:
                    try:
                        embedding = embedding_service.embed(content)
                        if embedding:
                            # Преобразуем numpy массив в список
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            elif not isinstance(embedding, list):
                                embedding = list(embedding)
                    except Exception as e:
                        logger.warning(f"Ошибка при генерации эмбеддинга для {record_id}: {e}")
                        error_count += 1
                        continue
                else:
                    logger.debug(f"Нет контента для генерации эмбеддинга для {record_id}")
                    skipped_count += 1
                    continue
            
            # Сохраняем эмбеддинг в граф
            if embedding is not None and len(embedding) > 0:
                try:
                    # Преобразуем numpy массив в список, если нужно
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)
                    
                    # Сохраняем в граф
                    graph.update_node(record_id, embedding=embedding)
                    
                    # Сохраняем в Qdrant, если доступен
                    if vector_store:
                        payload_data = {
                            "record_id": record_id,
                            "source": props.get("source") or props.get("chat", "unknown"),
                            "tags": props.get("tags", []),
                            "timestamp": props.get("timestamp") or props.get("date_utc", ""),
                            "timestamp_iso": props.get("timestamp") or props.get("date_utc", ""),
                            "content_preview": (content or "")[:200],
                        }
                        chat_name = props.get("chat")
                        if chat_name:
                            payload_data["chat"] = chat_name
                        
                        try:
                            vector_store.upsert(record_id, embedding, payload_data)
                        except Exception as e:
                            logger.debug(f"Ошибка при сохранении в Qdrant для {record_id}: {e}")
                    
                    fixed_count += 1
                    if fixed_count % 50 == 0:
                        logger.info(f"Обработано {fixed_count} узлов...")
                        
                except Exception as e:
                    logger.warning(f"Ошибка при сохранении эмбеддинга для {record_id}: {e}")
                    error_count += 1
            else:
                skipped_count += 1
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке узла {record_id}: {e}")
            error_count += 1
    
    logger.info(f"✅ Завершено: исправлено {fixed_count}, пропущено {skipped_count}, ошибок {error_count}")
    
    # Проверяем финальную статистику
    cursor.execute("SELECT COUNT(*) as total FROM nodes")
    total = cursor.fetchone()['total']
    cursor.execute("SELECT COUNT(*) as with_emb FROM nodes WHERE embedding IS NOT NULL AND embedding != ''")
    with_emb = cursor.fetchone()['with_emb']
    
    logger.info(f"📊 Финальная статистика: {with_emb}/{total} узлов с эмбеддингами ({with_emb/total*100:.1f}%)")
    
    graph.conn.close()
    if vector_store:
        vector_store.close()
    if embedding_service:
        embedding_service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов для всех узлов без эмбеддингов")
    parser.add_argument("--db-path", default="data/memory_graph.db", help="Путь к БД графа")
    parser.add_argument("--chroma-path", default="chroma_db", help="Путь к ChromaDB")
    args = parser.parse_args()
    
    fix_all_missing_embeddings(args.db_path, args.chroma_path)

