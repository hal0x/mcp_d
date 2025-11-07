#!/usr/bin/env python3
"""Диагностический скрипт для проверки производительности embeddings."""

import asyncio
import time
import logging
from pathlib import Path
import sys

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from llm.embeddings_client import AsyncEmbeddingsClient
from index.vector_index import VectorIndex

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embeddings_performance():
    """Тестирует производительность embeddings."""
    
    print("🔍 Диагностика производительности embeddings")
    print("=" * 50)
    
    # Тестовые тексты разного размера
    test_texts = [
        "Короткий текст",
        "Средний текст для тестирования производительности embeddings модели",
        "Очень длинный текст " * 50,  # ~1000 символов
        "Очень очень длинный текст " * 200,  # ~5000 символов
    ]
    
    # Тестируем разные конфигурации
    configs = [
        {"host": "127.0.0.1", "port": 1234, "provider": "lmstudio"},
        {"host": "127.0.0.1", "port": 11434, "provider": "ollama"},
    ]
    
    for config in configs:
        print(f"\n📡 Тестирование {config['provider']} на {config['host']}:{config['port']}")
        print("-" * 40)
        
        try:
            client = AsyncEmbeddingsClient(
                model="text-embedding-qwen3-embedding-8b",
                host=config["host"],
                port=config["port"],
                provider=config["provider"],
                timeout_seconds=10  # Короткий таймаут для быстрой диагностики
            )
            
            for i, text in enumerate(test_texts):
                print(f"\nТест {i+1}: {len(text)} символов")
                print(f"Текст: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                start_time = time.perf_counter()
                try:
                    embedding = await client.embed(text)
                    elapsed = time.perf_counter() - start_time
                    
                    if embedding:
                        print(f"✅ Успех: {elapsed:.2f}с, размер: {len(embedding)}")
                    else:
                        print(f"❌ Пустой результат: {elapsed:.2f}с")
                        
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    print(f"❌ Ошибка: {elapsed:.2f}с - {e}")
                    
        except Exception as e:
            print(f"❌ Не удалось подключиться: {e}")

async def test_vector_index_performance():
    """Тестирует производительность VectorIndex."""
    
    print("\n\n🔍 Диагностика производительности VectorIndex")
    print("=" * 50)
    
    try:
        # Создаем временный индекс
        vector_index = VectorIndex(
            path="debug_index.json",
            model_name="text-embedding-qwen3-embedding-8b",
            host="127.0.0.1",
            port=1234
        )
        
        # Тестируем добавление
        test_docs = [
            ("doc1", "Python это язык программирования", {"topic": "programming"}),
            ("doc2", "Искусственный интеллект меняет мир", {"topic": "ai"}),
            ("doc3", "Машинное обучение использует данные", {"topic": "ml"}),
        ]
        
        print("📝 Тестирование добавления документов:")
        for doc_id, text, metadata in test_docs:
            start_time = time.perf_counter()
            await vector_index.add(doc_id, text, metadata)
            elapsed = time.perf_counter() - start_time
            print(f"  {doc_id}: {elapsed:.2f}с")
        
        # Тестируем поиск
        print("\n🔍 Тестирование поиска:")
        queries = ["Python программирование", "искусственный интеллект", "машинное обучение"]
        
        for query in queries:
            start_time = time.perf_counter()
            results = await vector_index.search(query, top_k=2)
            elapsed = time.perf_counter() - start_time
            print(f"  '{query}': {elapsed:.2f}с, найдено: {len(results)}")
        
        # Очистка
        Path("debug_index.json").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Ошибка VectorIndex: {e}")

async def main():
    """Главная функция диагностики."""
    await test_embeddings_performance()
    await test_vector_index_performance()
    
    print("\n\n📊 Рекомендации по оптимизации:")
    print("=" * 50)
    print("1. Проверьте, что сервер embeddings запущен и доступен")
    print("2. Убедитесь, что модель загружена и готова к работе")
    print("3. Рассмотрите использование более быстрой модели embeddings")
    print("4. Проверьте сетевые задержки и таймауты")
    print("5. Используйте batch-обработку для множественных запросов")

if __name__ == "__main__":
    asyncio.run(main())
