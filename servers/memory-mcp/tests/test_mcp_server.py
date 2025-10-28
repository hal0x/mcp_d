#!/usr/bin/env python3
"""
🧪 Тестирование MCP сервера для Telegram дампов

Тестирует все доступные инструменты MCP сервера с улучшенной русской токенизацией.
"""

import asyncio
import json
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.mcp_server import TelegramDumpMCP


async def test_mcp_server():
    """Тестирование MCP сервера"""
    print("🧪 ТЕСТИРОВАНИЕ MCP СЕРВЕРА")
    print("=" * 50)

    # Инициализируем MCP сервер
    mcp = TelegramDumpMCP()

    # Тестовые данные
    test_texts = [
        "Bitcoin достиг $120,000",
        "капитализация 1.5 млрд долларов",
        "рост на 15% за месяц",
        "объем торгов 500 млн USDT",
        "цена €4,500 за токен",
        "инвестиции ₽1,000,000 в проект",
    ]

    print("\n🔍 ТЕСТ 1: Токенизация текста")
    print("-" * 30)

    for i, text in enumerate(test_texts, 1):
        print(f'\n{i}. Текст: "{text}"')
        try:
            result = await mcp._tokenize_text(text)
            data = json.loads(result)

            print(f"   Токены: {data['tokens']}")
            print(f"   Статистика: {data['statistics']}")

            if data["analysis"]["money_tokens"]:
                print(f"   💰 Валюты: {data['analysis']['money_tokens']}")
            if data["analysis"]["amount_tokens"]:
                print(f"   💵 Суммы: {data['analysis']['amount_tokens']}")
            if data["analysis"]["value_tokens"]:
                print(f"   📊 Значения: {data['analysis']['value_tokens']}")
            if data["analysis"]["type_tokens"]:
                print(f"   🏷️  Типы: {data['analysis']['type_tokens']}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    print("\n🔍 ТЕСТ 2: Поиск числовых данных")
    print("-" * 35)

    numeric_queries = [
        "Bitcoin $120,000",
        "капитализация 1.5 млрд",
        "рост 15%",
        "объем 500 млн",
    ]

    for i, query in enumerate(numeric_queries, 1):
        print(f'\n{i}. Запрос: "{query}"')
        try:
            result = await mcp._search_numeric_data(query, limit=3)
            data = json.loads(result)

            print(f"   Найдено числовых токенов: {len(data['numeric_tokens'])}")
            print(f"   Числовые токены: {data['numeric_tokens']}")
            print(f"   Найдено результатов: {data['total']}")
            print(f"   Числовых совпадений: {data['numeric_matches']}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    print("\n🔍 ТЕСТ 3: Получение списка чатов")
    print("-" * 35)

    try:
        chats_result = await mcp._get_chats_list()
        chats_data = json.loads(chats_result)

        if chats_data.get("chats"):
            print(f"Найдено чатов: {chats_data['total']}")
            print("\nПримеры чатов:")
            for i, chat in enumerate(chats_data["chats"][:5], 1):
                print(f"   {i}. {chat['name']}: {chat['message_count']} сообщений")
                if chat.get("first_message"):
                    print(f"      Первое сообщение: {chat['first_message']}")
                if chat.get("last_message"):
                    print(f"      Последнее сообщение: {chat['last_message']}")
        else:
            print("Чаты не найдены. Запустите 'memory_mcp index' для создания индексов.")

    except Exception as e:
        print(f"❌ Ошибка получения списка чатов: {e}")

    print("\n🔍 ТЕСТ 4: Анализ чата")
    print("-" * 25)

    # Ищем доступные чаты
    try:
        stats_result = await mcp._get_stats()
        stats_data = json.loads(stats_result)

        if stats_data.get("total_chats", 0) > 0:
            print(f"Найдено чатов: {stats_data['total_chats']}")

            # Получаем список чатов
            chats_result = await mcp._get_chats_list()
            chats_data = json.loads(chats_result)

            if chats_data.get("chats"):
                # Берем первый чат для анализа
                first_chat = chats_data["chats"][0]
                chat_name = first_chat["name"]

                print(f'\nАнализируем чат: "{chat_name}"')
                try:
                    analysis_result = await mcp._analyze_chat_content(
                        chat_name, sample_size=50
                    )
                    analysis_data = json.loads(analysis_result)

                    print(f"   Всего сообщений: {analysis_data['total_messages']}")
                    print(f"   Проанализировано: {analysis_data['analyzed_messages']}")
                    print(f"   Всего токенов: {analysis_data['total_tokens']}")
                    print(f"   Уникальных токенов: {analysis_data['unique_tokens']}")

                    stats = analysis_data["token_statistics"]
                    print(f"   💰 Валютных токенов: {stats['money_tokens']}")
                    print(f"   💵 Суммовых токенов: {stats['amount_tokens']}")
                    print(f"   📊 Значений: {stats['value_tokens']}")
                    print(f"   🏷️  Типов: {stats['type_tokens']}")
                    print(f"   🇷🇺 Русских токенов: {stats['russian_tokens']}")
                    print(f"   🇺🇸 Английских токенов: {stats['english_tokens']}")

                except Exception as e:
                    print(f"   ❌ Ошибка анализа чата: {e}")
        else:
            print("Чаты не найдены. Запустите 'memory_mcp index' для создания индексов.")

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")

    print("\n🔍 ТЕСТ 5: Статистика системы")
    print("-" * 30)

    try:
        stats_result = await mcp._get_stats()
        stats_data = json.loads(stats_result)

        print("📊 Статистика коллекций:")
        for collection, count in stats_data["collections"].items():
            print(f"   {collection}: {count} записей")

        print("📈 Общая статистика:")
        print(f"   Всего записей: {stats_data['total_records']}")
        print(f"   Всего чатов: {stats_data['total_chats']}")
        print(f"   Путь к ChromaDB: {stats_data['chroma_path']}")
        print(f"   Путь к чатам: {stats_data['chats_path']}")

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")

    print("\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 50)


async def main():
    """Главная функция"""
    try:
        await test_mcp_server()
    except KeyboardInterrupt:
        print("\n👋 Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
