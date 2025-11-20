#!/usr/bin/env python3
"""Скрипт для создания связей между уже проиндексированными сессиями."""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_mcp.memory.typed_graph import TypedGraphMemory
from memory_mcp.memory.graph_types import GraphEdge, EdgeType
from memory_mcp.config import get_settings
from memory_mcp.utils.datetime_utils import parse_datetime_utc
from memory_mcp.utils.naming import slugify
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_session_links_for_chat(graph: TypedGraphMemory, chat_name: str) -> int:
    """Создает связи между сессиями одного чата."""
    cursor = graph.conn.cursor()
    
    # Нормализуем имя чата
    chat_slug = slugify(chat_name) if chat_name else ""
    
    # Получаем все сессии чата
    query = """
        SELECT id, properties 
        FROM nodes 
        WHERE type = 'DocChunk' 
        AND properties IS NOT NULL
        AND json_extract(properties, '$.session_type') = 'session_summary'
        AND (
            json_extract(properties, '$.chat') = ?
            OR json_extract(properties, '$.source') = ?
        )
        ORDER BY json_extract(properties, '$.timestamp') ASC
    """
    
    cursor.execute(query, (chat_name, chat_name))
    sessions = cursor.fetchall()
    
    if len(sessions) < 2:
        logger.info(f"Для чата '{chat_name}' найдено {len(sessions)} сессий, связи не нужны")
        return 0
    
    logger.info(f"Найдено {len(sessions)} сессий для чата '{chat_name}'")
    
    created_links = 0
    
    # Создаем связи между соседними сессиями
    for i in range(len(sessions) - 1):
        current_session = sessions[i]
        next_session = sessions[i + 1]
        
        try:
            # Парсим properties
            current_props = json.loads(current_session["properties"]) if isinstance(current_session["properties"], str) else current_session["properties"]
            next_props = json.loads(next_session["properties"]) if isinstance(next_session["properties"], str) else next_session["properties"]
            
            if not current_props or not next_props:
                continue
            
            # Получаем timestamps
            current_timestamp_str = current_props.get("timestamp") or current_props.get("start_time_utc")
            next_timestamp_str = next_props.get("timestamp") or next_props.get("start_time_utc")
            
            if not current_timestamp_str or not next_timestamp_str:
                continue
            
            current_timestamp = parse_datetime_utc(current_timestamp_str, default=None)
            next_timestamp = parse_datetime_utc(next_timestamp_str, default=None)
            
            if not current_timestamp or not next_timestamp:
                continue
            
            # Создаем связь только если сессии близки по времени (в пределах 7 дней)
            time_diff = abs((next_timestamp - current_timestamp).total_seconds())
            if time_diff > 7 * 24 * 3600:  # 7 дней
                continue
            
            source_id = current_session["id"]
            target_id = next_session["id"]
            
            # Проверяем, существует ли уже связь
            cursor.execute("SELECT id FROM edges WHERE id = ?", (f"{source_id}-next-session-{target_id}",))
            if cursor.fetchone():
                logger.debug(f"Связь {source_id} -> {target_id} уже существует")
                continue
            
            edge = GraphEdge(
                id=f"{source_id}-next-session-{target_id}",
                source_id=source_id,
                target_id=target_id,
                type=EdgeType.RELATES_TO,
                weight=0.7,
                properties={
                    "time_diff_seconds": time_diff,
                    "relation_type": "session_sequence"
                },
            )
            
            if graph.add_edge(edge):
                created_links += 1
                logger.info(f"Создана связь: {source_id} -> {target_id} (разница: {time_diff / 3600:.1f} часов)")
        
        except Exception as e:
            logger.warning(f"Ошибка при создании связи между {current_session['id']} и {next_session['id']}: {e}")
            continue
    
    return created_links


def main():
    """Основная функция."""
    settings = get_settings()
    graph = TypedGraphMemory(db_path=settings.db_path)
    
    # Получаем список всех чатов
    cursor = graph.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT json_extract(properties, '$.chat') as chat
        FROM nodes 
        WHERE type = 'DocChunk' 
        AND properties IS NOT NULL
        AND json_extract(properties, '$.session_type') = 'session_summary'
        AND json_extract(properties, '$.chat') IS NOT NULL
    """)
    
    chats = [row["chat"] for row in cursor.fetchall() if row["chat"]]
    
    logger.info(f"Найдено чатов с сессиями: {len(chats)}")
    
    total_links = 0
    for chat_name in chats:
        logger.info(f"\nОбработка чата: {chat_name}")
        links = create_session_links_for_chat(graph, chat_name)
        total_links += links
        logger.info(f"Создано связей для '{chat_name}': {links}")
    
    logger.info(f"\n=== Итого создано связей: {total_links} ===")


if __name__ == "__main__":
    main()

