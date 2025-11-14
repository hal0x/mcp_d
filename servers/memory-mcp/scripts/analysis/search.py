#!/usr/bin/env python3
"""
Скрипт для поиска по индексированным данным
Использует ту же логику, что и CLI команда memory_mcp search
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb

from memory_mcp.core.lmstudio_client import LMStudioEmbeddingClient


async def search(
    query: str, collection: str = "messages", chat_filter: str = None, limit: int = 10
):
    """
    Поиск по индексированным данным

    Args:
        query: Поисковый запрос
        collection: Коллекция (messages, sessions, tasks)
        chat_filter: Фильтр по чату (опционально)
        limit: Лимит результатов
    """
    print(f"🔍 Поиск в коллекции '{collection}': '{query}'")
    if chat_filter:
        print(f"📋 Фильтр чата: '{chat_filter}'")
    print(f"📊 Лимит: {limit}")
    print("=" * 50)
    print()

    try:
        # Инициализируем клиентов
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        embedding_client = LMStudioEmbeddingClient()

        # Получаем коллекцию
        collection_name = f"chat_{collection}"
        try:
            coll = chroma_client.get_collection(collection_name)
            print(f"✅ Найдено записей в коллекции: {coll.count()}")
            print()
        except Exception:
            print(f"❌ Коллекция {collection_name} не найдена")
            print("💡 Запустите 'memory_mcp index' для создания индексов")
            return

        # Генерируем эмбеддинг
        async with embedding_client:
            query_embedding = await embedding_client._generate_single_embedding(query)

            if not query_embedding:
                print("❌ Не удалось сгенерировать эмбеддинг для запроса")
                return

            # Выполняем поиск
            where_filter = {"chat": chat_filter} if chat_filter else None
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Запрашиваем больше для дедупликации
                where=where_filter,
            )

            if not results["documents"] or not results["documents"][0]:
                print("❌ Результаты не найдены")
                return

            # Дедупликация результатов
            seen_docs = set()
            unique_results = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                doc_key = doc[:100].strip()
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append((doc, metadata, distance))
                    if len(unique_results) >= limit:
                        break

            print(f"✅ Найдено уникальных результатов: {len(unique_results)}")
            if len(results["documents"][0]) > len(unique_results):
                removed = len(results["documents"][0]) - len(unique_results)
                print(f"   (удалено {removed} дубликатов)")
            print()

            # Выводим результаты
            for i, (doc, metadata, distance) in enumerate(unique_results, 1):
                chat_name = metadata.get("chat", metadata.get("chat_name", "Unknown"))
                print(f"{i}. {chat_name} (расстояние: {distance:.3f})")

                if collection == "messages":
                    text = doc[:200] + "..." if len(doc) > 200 else doc
                    print(f"   {text}")
                elif collection == "sessions":
                    session_id = metadata.get("session_id", "N/A")
                    time_range = metadata.get("time_span", "N/A")
                    print(f"   Session: {session_id}")
                    print(f"   Time: {time_range}")
                    summary = doc[:150] + "..." if len(doc) > 150 else doc
                    print(f"   Summary: {summary}")
                elif collection == "tasks":
                    task_text = doc[:200] + "..." if len(doc) > 200 else doc
                    owner = metadata.get("owner", "N/A")
                    due_date = metadata.get("due", "N/A")
                    priority = metadata.get("priority", "N/A")
                    print(f"   Task: {task_text}")
                    print(f"   Owner: {owner} | Due: {due_date} | Priority: {priority}")

                print()

    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python search.py 'запрос' [коллекция] [чат] [лимит]")
        print()
        print("Примеры:")
        print("  python search.py 'криптовалюта'")
        print("  python search.py 'TON блокчейн' sessions")
        print("  python search.py 'задача' tasks")
        print("  python search.py 'DeFi' messages 'LobsterDAO _' 5")
        print()
        print("Коллекции: messages (по умолчанию), sessions, tasks")
        sys.exit(1)

    query = sys.argv[1]
    collection = "messages"
    chat_filter = None
    limit = 10

    # Парсим аргументы
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ["messages", "sessions", "tasks"]:
            collection = arg
        elif arg.isdigit():
            limit = int(arg)
        else:
            chat_filter = arg
        i += 1

    asyncio.run(search(query, collection, chat_filter, limit))
