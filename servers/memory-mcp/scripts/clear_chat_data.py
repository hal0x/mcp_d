#!/usr/bin/env python3
"""Скрипт для очистки данных чата из базы данных и векторного хранилища."""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_mcp.memory.typed_graph import TypedGraphMemory
from memory_mcp.mcp.adapters import MemoryServiceAdapter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_chat_data(chat_name: str, db_path: str = "data/memory_graph.db"):
    """Очищает все данные чата из базы данных и векторного хранилища."""
    
    logger.info(f"Начинаем очистку данных для чата: {chat_name}")
    
    # 1. Очистка из SQLite базы данных
    try:
        graph = TypedGraphMemory(db_path=db_path)
        deleted_nodes = graph.delete_nodes_by_chat(chat_name)
        logger.info(f"✅ Удалено {deleted_nodes} узлов из базы данных")
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке базы данных: {e}")
        return False
    
    # 2. Очистка из ChromaDB
    try:
        adapter = MemoryServiceAdapter(db_path=db_path)
        deleted_chroma = adapter._clear_chromadb_chat(chat_name)
        logger.info(f"✅ Удалено {deleted_chroma} записей из ChromaDB")
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке ChromaDB: {e}")
        return False
    
    logger.info(f"✅ Очистка данных для чата '{chat_name}' завершена")
    logger.info(f"   - Узлов удалено: {deleted_nodes}")
    logger.info(f"   - Записей из ChromaDB удалено: {deleted_chroma}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python scripts/clear_chat_data.py <chat_name> [db_path]")
        print("Пример: python scripts/clear_chat_data.py 'Семья'")
        sys.exit(1)
    
    chat_name = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "data/memory_graph.db"
    
    success = clear_chat_data(chat_name, db_path)
    sys.exit(0 if success else 1)

