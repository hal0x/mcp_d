#!/usr/bin/env python3
"""
Пример использования нового инструмента index_chat через MCP сервер
"""

import asyncio
import json

from scripts.mcp_server import TelegramDumpMCP


async def example_index_chat():
    """Пример использования инструмента index_chat"""

    print("🎯 Пример использования инструмента index_chat")
    print("=" * 60)

    # Создаем экземпляр MCP сервера
    mcp = TelegramDumpMCP()

    # Пример 1: Инкрементальная индексация чата "Believers Community"
    print("\n📊 Пример 1: Инкрементальная индексация")
    print("-" * 40)

    result1 = await mcp._index_chat(
        chat_name="Believers Community",
        force_full=False,
        recent_days=7,  # Только последние 7 дней
        enable_smart_aggregation=True,
    )

    data1 = json.loads(result1)
    if data1.get("success"):
        stats = data1.get("statistics", {})
        print(f"✅ Успешно проиндексирован чат: {data1['chat_name']}")
        print("   📈 Статистика:")
        print(f"      - Сессий: {stats.get('sessions_indexed', 0)}")
        print(f"      - Сообщений: {stats.get('messages_indexed', 0)}")
        print(f"      - Задач: {stats.get('tasks_indexed', 0)}")
    else:
        print(f"❌ Ошибка: {data1.get('error', 'Unknown error')}")

    # Пример 2: Полная переиндексация с настройками
    print("\n🔄 Пример 2: Полная переиндексация с настройками")
    print("-" * 40)

    result2 = await mcp._index_chat(
        chat_name="Семья",
        force_full=True,
        recent_days=0,  # Все сообщения
        enable_clustering=True,
        enable_smart_aggregation=True,
        max_messages_per_group=150,
        max_session_hours=8,
        gap_minutes=90,
    )

    data2 = json.loads(result2)
    if data2.get("success"):
        stats = data2.get("statistics", {})
        artifacts = data2.get("artifacts_created", {})
        print(f"✅ Успешно проиндексирован чат: {data2['chat_name']}")
        print("   📈 Статистика:")
        print(f"      - Сессий: {stats.get('sessions_indexed', 0)}")
        print(f"      - Сообщений: {stats.get('messages_indexed', 0)}")
        print(f"      - Задач: {stats.get('tasks_indexed', 0)}")
        print("   📁 Созданные артефакты:")
        print(f"      - Отчёты: {artifacts.get('reports_path', 'N/A')}")
        print(
            f"      - Коллекции: {', '.join(artifacts.get('chroma_collections', []))}"
        )
    else:
        print(f"❌ Ошибка: {data2.get('error', 'Unknown error')}")

    # Пример 3: Быстрая индексация последних дней
    print("\n⚡ Пример 3: Быстрая индексация последних дней")
    print("-" * 40)

    result3 = await mcp._index_chat(
        chat_name="TON Status",
        force_full=False,
        recent_days=3,  # Только последние 3 дня
        enable_smart_aggregation=True,
    )

    data3 = json.loads(result3)
    if data3.get("success"):
        stats = data3.get("statistics", {})
        print(f"✅ Успешно проиндексирован чат: {data3['chat_name']}")
        print("   📈 Статистика:")
        print(f"      - Сессий: {stats.get('sessions_indexed', 0)}")
        print(f"      - Сообщений: {stats.get('messages_indexed', 0)}")
        print(f"      - Задач: {stats.get('tasks_indexed', 0)}")
    else:
        print(f"❌ Ошибка: {data3.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("🎉 Примеры использования завершены!")
    print("\n💡 Для использования через MCP клиент:")
    print("   1. Запустите MCP сервер: python -m memory_mcp.mcp.server")
    print("   2. Подключитесь через MCP клиент")
    print("   3. Вызовите инструмент index_chat с нужными параметрами")


if __name__ == "__main__":
    try:
        asyncio.run(example_index_chat())
    except KeyboardInterrupt:
        print("\n👋 Пример прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
