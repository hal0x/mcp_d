#!/usr/bin/env python3
"""
Скрипт для сравнения результатов поиска с разными запросами
Помогает оценить качество семантического поиска
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def simple_table(data, headers):
    """Простая реализация таблицы без внешних зависимостей"""
    # Вычисляем ширину столбцов
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Разделитель
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Формат строки
    def format_row(row):
        return (
            "| "
            + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    # Собираем таблицу
    result = [separator]
    result.append(format_row(headers))
    result.append(separator)
    for row in data:
        result.append(format_row(row))
    result.append(separator)

    return "\n".join(result)


import chromadb

from memory_mcp.core.lmstudio_client import LMStudioEmbeddingClient


async def search_single(
    query: str, collection_name: str, chroma_client, embedding_client, limit: int = 5
) -> Dict[str, Any]:
    """
    Выполняет один поиск и возвращает результаты с метриками
    """
    try:
        coll = chroma_client.get_collection(collection_name)
    except Exception:
        return {
            "query": query,
            "error": f"Коллекция {collection_name} не найдена",
            "results": [],
        }

    # Генерируем эмбеддинг
    query_embedding = await embedding_client._generate_single_embedding(query)

    if not query_embedding:
        return {
            "query": query,
            "error": "Не удалось сгенерировать эмбеддинг",
            "results": [],
        }

    # Выполняем поиск
    results = coll.query(query_embeddings=[query_embedding], n_results=limit)

    if not results["documents"] or not results["documents"][0]:
        return {
            "query": query,
            "results": [],
            "avg_distance": None,
            "min_distance": None,
            "max_distance": None,
        }

    # Собираем результаты
    parsed_results = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        parsed_results.append(
            {
                "chat": metadata.get("chat", "Unknown"),
                "text": doc[:100] + "..." if len(doc) > 100 else doc,
                "distance": distance,
            }
        )

    distances = results["distances"][0]

    return {
        "query": query,
        "results": parsed_results,
        "avg_distance": sum(distances) / len(distances),
        "min_distance": min(distances),
        "max_distance": max(distances),
        "count": len(parsed_results),
    }


async def compare_queries(
    queries: List[str], collection: str = "messages", limit: int = 5
):
    """
    Сравнивает результаты поиска для нескольких запросов
    """
    print(f"\n{'='*80}")
    print("🔍 СРАВНЕНИЕ ЗАПРОСОВ")
    print(f"{'='*80}\n")
    print(f"📊 Коллекция: chat_{collection}")
    print(f"📊 Лимит результатов: {limit}")
    print(f"📊 Запросов для сравнения: {len(queries)}\n")

    # Инициализируем клиентов
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_client = LMStudioEmbeddingClient()

    async with embedding_client:
        # Выполняем все запросы
        all_results = []
        for i, query in enumerate(queries, 1):
            print(f"⏳ Обработка запроса {i}/{len(queries)}: '{query}'")
            result = await search_single(
                query, f"chat_{collection}", chroma_client, embedding_client, limit
            )
            all_results.append(result)

        print()
        print(f"{'='*80}")
        print("📈 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print(f"{'='*80}\n")

        # Таблица метрик
        metrics_table = []
        for result in all_results:
            if "error" in result:
                metrics_table.append(
                    [
                        result["query"][:30] + "..."
                        if len(result["query"]) > 30
                        else result["query"],
                        "❌ ОШИБКА",
                        "-",
                        "-",
                        "-",
                    ]
                )
            else:
                # Определяем качество
                min_dist = result["min_distance"]
                if min_dist < 500:
                    quality = "⭐⭐⭐⭐⭐"
                elif min_dist < 600:
                    quality = "⭐⭐⭐⭐"
                elif min_dist < 700:
                    quality = "⭐⭐⭐"
                else:
                    quality = "⭐⭐"

                metrics_table.append(
                    [
                        result["query"][:30] + "..."
                        if len(result["query"]) > 30
                        else result["query"],
                        quality,
                        f"{result['min_distance']:.1f}",
                        f"{result['avg_distance']:.1f}",
                        f"{result['max_distance']:.1f}",
                    ]
                )

        print(
            simple_table(
                metrics_table, headers=["Запрос", "Качество", "Мин", "Средн", "Макс"]
            )
        )

        print("\n")
        print(f"{'='*80}")
        print("📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print(f"{'='*80}\n")

        # Детальные результаты для каждого запроса
        for i, result in enumerate(all_results, 1):
            print(f"\n{i}. Запрос: '{result['query']}'")
            print(f"   {'─'*76}")

            if "error" in result:
                print(f"   ❌ Ошибка: {result['error']}")
                continue

            if not result["results"]:
                print("   ⚠️  Результаты не найдены")
                continue

            for j, res in enumerate(result["results"], 1):
                print(f"   {j}. [{res['chat']}] (dist: {res['distance']:.1f})")
                print(f"      {res['text']}")
                print()

        # Рекомендации
        print(f"\n{'='*80}")
        print("💡 РЕКОМЕНДАЦИИ")
        print(f"{'='*80}\n")

        best_query = min(all_results, key=lambda x: x.get("min_distance", 9999))
        worst_query = max(all_results, key=lambda x: x.get("min_distance", 0))

        print(f"✅ Лучший запрос: '{best_query['query']}'")
        print(f"   Минимальное расстояние: {best_query.get('min_distance', 'N/A'):.1f}")
        print()
        print(f"⚠️  Худший запрос: '{worst_query['query']}'")
        print(
            f"   Минимальное расстояние: {worst_query.get('min_distance', 'N/A'):.1f}"
        )
        print()

        # Советы
        avg_min = sum(r.get("min_distance", 0) for r in all_results) / len(all_results)

        print("💭 Выводы:")
        if avg_min < 550:
            print(
                "   • Запросы хорошо подобраны — семантический поиск работает отлично"
            )
        elif avg_min < 650:
            print("   • Запросы работают удовлетворительно")
            print("   • Рекомендуется использовать более общие формулировки")
        else:
            print("   • Запросы слишком специфичные или нерелевантные базе")
            print("   • Попробуйте переформулировать в более общие термины")
            print("   • Рассмотрите гибридный поиск (векторный + текстовый)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Использование: python search_comparison.py 'запрос1' 'запрос2' ... [коллекция] [лимит]"
        )
        print()
        print("Примеры:")
        print(
            "  python search_comparison.py 'Falcon 9' 'космический запуск' 'ракета SpaceX'"
        )
        print(
            "  python search_comparison.py 'DeFi' 'децентрализованные финансы' sessions"
        )
        print("  python search_comparison.py 'блокчейн' 'blockchain' messages 10")
        print()
        print("Коллекции: messages (по умолчанию), sessions, tasks")
        sys.exit(1)

    # Парсим аргументы
    queries = []
    collection = "messages"
    limit = 5

    for arg in sys.argv[1:]:
        if arg in ["messages", "sessions", "tasks"]:
            collection = arg
        elif arg.isdigit():
            limit = int(arg)
        else:
            queries.append(arg)

    if not queries:
        print("❌ Ошибка: укажите хотя бы один запрос")
        sys.exit(1)

    asyncio.run(compare_queries(queries, collection, limit))
