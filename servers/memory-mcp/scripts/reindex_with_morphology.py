#!/usr/bin/env python3
"""
Скрипт для переиндексации данных с нормализованными формами в FTS5

Обновляет FTS5 индекс для всех узлов, добавляя нормализованные формы слов
для улучшения поиска по русскому языку.
"""

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_mcp.memory.typed_graph import TypedGraphMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def reindex_with_morphology(db_path: str = "data/memory_graph.db", chat_filter: str = None):
    """
    Переиндексация всех узлов с нормализованными формами
    
    Args:
        db_path: Путь к базе данных
        chat_filter: Опциональный фильтр по чату (например, "Семья")
    """
    logger.info(f"Начинаем переиндексацию с морфологией для БД: {db_path}")
    
    # Инициализируем граф
    graph = TypedGraphMemory(db_path=db_path)
    
    # Получаем все узлы
    nodes = list(graph.graph.nodes(data=True))
    total_nodes = len(nodes)
    
    logger.info(f"Найдено узлов для переиндексации: {total_nodes}")
    
    if chat_filter:
        logger.info(f"Применяем фильтр по чату: {chat_filter}")
    
    reindexed_count = 0
    skipped_count = 0
    error_count = 0
    
    for node_id, node_data in nodes:
        try:
            # Фильтруем по чату, если указан
            if chat_filter:
                properties = node_data.get("properties", {})
                chat = properties.get("chat", "")
                source = properties.get("source", "")
                if chat_filter.lower() not in str(chat).lower() and chat_filter.lower() not in str(source).lower():
                    skipped_count += 1
                    continue
            
            # Получаем данные узла
            content = node_data.get("content", "")
            source = node_data.get("source", "")
            tags = node_data.get("tags", [])
            entities = node_data.get("entities", [])
            properties = node_data.get("properties", {})
            
            # Если нет контента, пропускаем
            if not content:
                skipped_count += 1
                continue
            
            # Обновляем FTS5 индекс (это автоматически добавит нормализованные формы)
            graph._fts_refresh_doc(
                node_id=node_id,
                content=content,
                source=source or "",
                tags=tags or [],
                entities=entities or [],
                properties=properties,
            )
            
            reindexed_count += 1
            
            if reindexed_count % 100 == 0:
                logger.info(f"Переиндексировано: {reindexed_count}/{total_nodes}")
        
        except Exception as e:
            logger.error(f"Ошибка при переиндексации узла {node_id}: {e}")
            error_count += 1
    
    logger.info("=" * 60)
    logger.info("Переиндексация завершена")
    logger.info(f"Всего узлов: {total_nodes}")
    logger.info(f"Переиндексировано: {reindexed_count}")
    logger.info(f"Пропущено: {skipped_count}")
    logger.info(f"Ошибок: {error_count}")
    logger.info("=" * 60)
    
    # Закрываем соединение
    graph.conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Переиндексация данных с нормализованными формами в FTS5"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/memory_graph.db",
        help="Путь к базе данных (по умолчанию: data/memory_graph.db)",
    )
    parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Фильтр по чату (например, 'Семья')",
    )
    
    args = parser.parse_args()
    
    reindex_with_morphology(db_path=args.db_path, chat_filter=args.chat)

