#!/usr/bin/env python3
"""
Тестовый скрипт для проверки нового инструмента index_chat в MCP сервере
"""

import asyncio
import json
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импорты после изменения PYTHONPATH
from scripts.mcp_server import TelegramDumpMCP  # noqa: E402


async def test_index_chat():
    """Тестирование инструмента index_chat"""

    print("🧪 Тестирование инструмента index_chat")
    print("=" * 50)

    # Создаем экземпляр MCP сервера
    mcp = TelegramDumpMCP()

    # Тест 1: Проверяем список доступных чатов
    print("\n📋 Тест 1: Получение списка чатов")
    chats_result = await mcp._get_chats_list()
    chats_data = json.loads(chats_result)

    if "chats" in chats_data and chats_data["chats"]:
        available_chats = [chat["name"] for chat in chats_data["chats"]]
        print(f"✅ Найдено чатов: {len(available_chats)}")
        print(
            f"   Доступные чаты: {', '.join(available_chats[:5])}{'...' if len(available_chats) > 5 else ''}"
        )

        # Берем первый чат для тестирования
        test_chat = available_chats[0]
        print(f"🎯 Тестируем чат: {test_chat}")

        # Тест 2: Индексация чата (инкрементальная)
        print(f"\n📊 Тест 2: Инкрементальная индексация чата '{test_chat}'")
        index_result = await mcp._index_chat(
            chat_name=test_chat,
            force_full=False,
            recent_days=7,  # Только последние 7 дней
            enable_clustering=False,
            enable_smart_aggregation=True,
        )

        index_data = json.loads(index_result)
        if index_data.get("success"):
            print("✅ Инкрементальная индексация успешна")
            stats = index_data.get("statistics", {})
            print(f"   - Сессий: {stats.get('sessions_indexed', 0)}")
            print(f"   - Сообщений: {stats.get('messages_indexed', 0)}")
            print(f"   - Задач: {stats.get('tasks_indexed', 0)}")
        else:
            print(
                f"❌ Ошибка инкрементальной индексации: {index_data.get('error', 'Unknown error')}"
            )

        # Тест 3: Полная переиндексация чата
        print(f"\n🔄 Тест 3: Полная переиндексация чата '{test_chat}'")
        full_index_result = await mcp._index_chat(
            chat_name=test_chat,
            force_full=True,
            recent_days=0,  # Все сообщения
            enable_clustering=True,
            enable_smart_aggregation=True,
            max_messages_per_group=150,
            max_session_hours=8,
            gap_minutes=90,
        )

        full_index_data = json.loads(full_index_result)
        if full_index_data.get("success"):
            print("✅ Полная переиндексация успешна")
            stats = full_index_data.get("statistics", {})
            print(f"   - Сессий: {stats.get('sessions_indexed', 0)}")
            print(f"   - Сообщений: {stats.get('messages_indexed', 0)}")
            print(f"   - Задач: {stats.get('tasks_indexed', 0)}")

            # Показываем созданные артефакты
            artifacts = full_index_data.get("artifacts_created", {})
            print(f"   - Отчёты: {artifacts.get('reports_path', 'N/A')}")
            print(
                f"   - Коллекции: {', '.join(artifacts.get('chroma_collections', []))}"
            )
        else:
            print(
                f"❌ Ошибка полной переиндексации: {full_index_data.get('error', 'Unknown error')}"
            )

    else:
        print("❌ Чаты не найдены")
        return

    # Тест 4: Тестирование с несуществующим чатом
    print("\n🚫 Тест 4: Индексация несуществующего чата")
    invalid_result = await mcp._index_chat(chat_name="NonExistentChat")
    invalid_data = json.loads(invalid_result)

    if not invalid_data.get("success"):
        print("✅ Корректно обработана ошибка несуществующего чата")
        print(f"   Ошибка: {invalid_data.get('error', 'Unknown error')}")
    else:
        print("❌ Неожиданно успешная индексация несуществующего чата")

    print("\n" + "=" * 50)
    print("🎉 Тестирование завершено!")


async def test_mcp_tool_call():
    """Тестирование через MCP call_tool"""

    print("\n🔧 Тестирование через MCP call_tool")
    print("=" * 50)

    mcp = TelegramDumpMCP()

    # Регистрируем обработчики
    mcp._register_handlers()

    # Тест вызова через MCP
    try:
        result = await mcp.server.call_tool(
            "index_chat",
            {
                "chat_name": "Believers Community",
                "force_full": False,
                "recent_days": 3,
                "enable_smart_aggregation": True,
            },
        )

        print("✅ MCP call_tool успешно выполнен")
        print(f"Результат: {result[0].text[:200]}...")

    except Exception as e:
        print(f"❌ Ошибка MCP call_tool: {e}")


if __name__ == "__main__":
    print("🚀 Запуск тестов для инструмента index_chat")

    try:
        # Запускаем тесты
        asyncio.run(test_index_chat())
        asyncio.run(test_mcp_tool_call())

    except KeyboardInterrupt:
        print("\n👋 Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
