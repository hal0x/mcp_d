#!/usr/bin/env python3
"""
Скрипт для проверки доступности Ollama и модели
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_mcp.core.ollama_client import OllamaEmbeddingClient, OllamaEmbeddingClientSync


async def check_ollama_async():
    """Асинхронная проверка Ollama"""
    print("🔍 Проверка Ollama (асинхронная версия)...")

    client = OllamaEmbeddingClient()

    async with client:
        result = await client.test_connection()

        print(f"✅ Ollama доступен: {result['ollama_available']}")
        if result["ollama_available"]:
            print(f"📦 Версия Ollama: {result.get('ollama_version', 'unknown')}")

        print(f"🤖 Модель доступна: {result['model_available']}")
        print(f"📝 Модель: {result['model_name']}")
        print(f"🌐 URL: {result['base_url']}")

        if result.get("error"):
            print(f"❌ Ошибка: {result['error']}")

        # Тестируем генерацию эмбеддинга
        if result["ollama_available"] and result["model_available"]:
            print("\n🧪 Тестирование генерации эмбеддинга...")
            test_texts = [
                "Привет, это тестовое сообщение",
                "Hello, this is a test message",
            ]

            try:
                embeddings = await client.generate_embeddings(test_texts)
                print(f"✅ Сгенерировано {len(embeddings)} эмбеддингов")
                print(f"📏 Размер эмбеддинга: {len(embeddings[0])} измерений")

                # Проверяем, что эмбеддинги не нулевые
                non_zero_count = sum(
                    1 for emb in embeddings for val in emb if val != 0.0
                )
                print(f"🔢 Ненулевых значений: {non_zero_count}")

            except Exception as e:
                print(f"❌ Ошибка при генерации эмбеддинга: {e}")
        else:
            print("\n⚠️ Невозможно протестировать генерацию эмбеддингов")


def check_ollama_sync():
    """Синхронная проверка Ollama"""
    print("\n🔍 Проверка Ollama (синхронная версия)...")

    client = OllamaEmbeddingClientSync()
    result = client.test_connection()

    print(f"✅ Ollama доступен: {result['ollama_available']}")
    if result["ollama_available"]:
        print(f"📦 Версия Ollama: {result.get('ollama_version', 'unknown')}")

    print(f"🤖 Модель доступна: {result['model_available']}")
    print(f"📝 Модель: {result['model_name']}")
    print(f"🌐 URL: {result['base_url']}")

    if result.get("error"):
        print(f"❌ Ошибка: {result['error']}")

    # Тестируем генерацию эмбеддинга
    if result["ollama_available"] and result["model_available"]:
        print("\n🧪 Тестирование генерации эмбеддинга...")
        test_texts = ["Привет, это тестовое сообщение", "Hello, this is a test message"]

        try:
            embeddings = client.generate_embeddings(test_texts)
            print(f"✅ Сгенерировано {len(embeddings)} эмбеддингов")
            print(f"📏 Размер эмбеддинга: {len(embeddings[0])} измерений")

            # Проверяем, что эмбеддинги не нулевые
            non_zero_count = sum(1 for emb in embeddings for val in emb if val != 0.0)
            print(f"🔢 Ненулевых значений: {non_zero_count}")

        except Exception as e:
            print(f"❌ Ошибка при генерации эмбеддинга: {e}")
    else:
        print("\n⚠️ Невозможно протестировать генерацию эмбеддингов")


async def main():
    """Главная функция"""
    print("🚀 Проверка интеграции с Ollama")
    print("=" * 50)

    try:
        await check_ollama_async()
        check_ollama_sync()

        print("\n✅ Проверка завершена!")

    except Exception as e:
        print(f"\n❌ Ошибка при проверке: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
