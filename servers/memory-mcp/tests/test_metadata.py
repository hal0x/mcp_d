#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности включения метаданных
"""

import asyncio
import json
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.mcp_server import TelegramDumpMCP


async def test_metadata_functionality():
    """Тестируем функциональность включения метаданных"""

    print("🚀 Тестируем функциональность включения метаданных...")

    # Создаем экземпляр MCP сервера
    mcp = TelegramDumpMCP()

    query = "андрюшка травма рука"

    print(f"\n🔍 Тестируем поиск по запросу: '{query}'")

    # Поиск без метаданных
    print("\n📋 Поиск БЕЗ метаданных:")
    try:
        result = await mcp._search_collection(
            collection_name="chat_sessions",
            query=query,
            chat_filter="Семья",
            limit=1,
            depth="shallow",
            include_metadata=False,
        )
        data = json.loads(result)
        print(f"✅ Найдено результатов: {data.get('total', 0)}")
        print(f"Метаданные: {'есть' if 'metadata' in data else 'нет'}")

    except Exception as e:
        print(f"❌ Ошибка в поиске без метаданных: {e}")

    # Поиск с метаданными
    print("\n📚 Поиск С метаданными:")
    try:
        result = await mcp._search_collection(
            collection_name="chat_sessions",
            query=query,
            chat_filter="Семья",
            limit=1,
            depth="shallow",
            include_metadata=True,
        )
        data = json.loads(result)
        print(f"✅ Найдено результатов: {data.get('total', 0)}")
        print(f"Метаданные: {'есть' if 'metadata' in data else 'нет'}")

        if "metadata" in data:
            metadata = data["metadata"]
            print(f"Количество сессий с метаданными: {len(metadata)}")

            for _key, meta in metadata.items():
                print(f"\n🔸 Сессия: {meta.get('session_id', 'unknown')}")
                print(f"   Чат: {meta.get('chat', 'unknown')}")
                print(f"   Файл: {meta.get('file_path', 'unknown')}")

                for section, content in meta.items():
                    if section not in ["session_id", "chat", "file_path"]:
                        print(f"   {section}: {len(content)} элементов")
                        if (
                            content and len(content) <= 3
                        ):  # Показываем первые несколько элементов
                            for item in content[:3]:
                                preview = item[:50] + "..." if len(item) > 50 else item
                                print(f"     - {preview}")

    except Exception as e:
        print(f"❌ Ошибка в поиске с метаданными: {e}")


if __name__ == "__main__":
    asyncio.run(test_metadata_functionality())
