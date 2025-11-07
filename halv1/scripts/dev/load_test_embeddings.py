#!/usr/bin/env python3
"""Нагрузочное тестирование embeddings для проверки производительности."""

import asyncio
import time
import logging
from pathlib import Path
import sys
from typing import List, Dict, Any
import random
import string

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from llm.embeddings_client import AsyncEmbeddingsClient
from index.vector_index import VectorIndex
from utils.performance import profiler

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_texts(count: int, min_length: int = 10, max_length: int = 1000) -> List[str]:
    """Генерирует тестовые тексты разной длины."""
    texts = []
    for i in range(count):
        length = random.randint(min_length, max_length)
        text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
        texts.append(text)
    return texts

async def test_embeddings_client_performance():
    """Тестирует производительность AsyncEmbeddingsClient."""
    logger.info("🔍 Тестирование AsyncEmbeddingsClient")
    
    client = AsyncEmbeddingsClient(
        model="text-embedding-qwen3-embedding-8b",
        host="127.0.0.1",
        port=1234,
        timeout_seconds=5
    )
    
    # Тестируем разные сценарии
    test_scenarios = [
        {"name": "Короткие тексты", "count": 100, "min_length": 10, "max_length": 50},
        {"name": "Средние тексты", "count": 50, "min_length": 100, "max_length": 500},
        {"name": "Длинные тексты", "count": 20, "min_length": 500, "max_length": 2000},
        {"name": "Смешанные тексты", "count": 200, "min_length": 10, "max_length": 1000},
    ]
    
    for scenario in test_scenarios:
        logger.info(f"📊 {scenario['name']}: {scenario['count']} текстов")
        
        texts = generate_test_texts(
            scenario["count"], 
            scenario["min_length"], 
            scenario["max_length"]
        )
        
        # Тестируем одиночные запросы
        start_time = time.perf_counter()
        for text in texts[:10]:  # Тестируем только первые 10
            await client.embed(text)
        single_time = time.perf_counter() - start_time
        
        # Тестируем batch запросы
        start_time = time.perf_counter()
        await client.embed_many(texts)
        batch_time = time.perf_counter() - start_time
        
        logger.info(f"  Одиночные запросы (10): {single_time:.2f}с")
        logger.info(f"  Batch запросы ({len(texts)}): {batch_time:.2f}с")
        logger.info(f"  Скорость batch: {len(texts)/batch_time:.1f} текстов/с")

async def test_vector_index_performance():
    """Тестирует производительность VectorIndex."""
    logger.info("🔍 Тестирование VectorIndex")
    
    # Создаем временный индекс
    vector_index = VectorIndex(
        path="load_test_index.json",
        model_name="text-embedding-qwen3-embedding-8b",
        host="127.0.0.1",
        port=1234
    )
    
    # Генерируем тестовые данные
    test_docs = []
    for i in range(100):
        text = f"Тестовый документ номер {i} с содержимым для проверки производительности индексации"
        metadata = {"doc_id": str(i), "category": f"category_{i % 10}"}
        test_docs.append((f"doc_{i}", text, metadata))
    
    # Тестируем добавление документов
    logger.info("📝 Тестирование добавления документов")
    start_time = time.perf_counter()
    
    for doc_id, text, metadata in test_docs:
        await vector_index.add(doc_id, text, metadata)
    
    add_time = time.perf_counter() - start_time
    logger.info(f"  Добавлено {len(test_docs)} документов за {add_time:.2f}с")
    logger.info(f"  Скорость: {len(test_docs)/add_time:.1f} документов/с")
    
    # Тестируем поиск
    logger.info("🔍 Тестирование поиска")
    search_queries = [
        "тестовый документ",
        "производительность индексации",
        "категория",
        "номер",
        "содержимое"
    ]
    
    start_time = time.perf_counter()
    for query in search_queries:
        results = await vector_index.search(query, top_k=10)
    search_time = time.perf_counter() - start_time
    
    logger.info(f"  Выполнено {len(search_queries)} поисковых запросов за {search_time:.2f}с")
    logger.info(f"  Скорость: {len(search_queries)/search_time:.1f} запросов/с")
    
    # Очистка
    Path("load_test_index.json").unlink(missing_ok=True)

async def monitor_metrics_during_test():
    """Мониторит метрики во время тестирования."""
    logger.info("📊 Мониторинг метрик во время тестирования")
    
    initial_metrics = profiler.get_all_metrics()
    logger.info("Начальные метрики:")
    for metric, data in initial_metrics.items():
        if "avg_time_ms" in data:
            logger.info(f"  {metric}: {data['avg_time_ms']:.2f}мс")
    
    # Ждем немного для накопления метрик
    await asyncio.sleep(2)
    
    final_metrics = profiler.get_all_metrics()
    logger.info("Финальные метрики:")
    for metric, data in final_metrics.items():
        if "avg_time_ms" in data:
            initial_avg = initial_metrics.get(metric, {}).get("avg_time_ms", 0)
            current_avg = data["avg_time_ms"]
            change = current_avg - initial_avg
            logger.info(f"  {metric}: {current_avg:.2f}мс (изменение: {change:+.2f}мс)")

async def main():
    """Главная функция нагрузочного тестирования."""
    logger.info("🚀 Запуск нагрузочного тестирования embeddings")
    logger.info("=" * 60)
    
    try:
        # Мониторим метрики до тестирования
        await monitor_metrics_during_test()
        
        # Тестируем AsyncEmbeddingsClient
        await test_embeddings_client_performance()
        
        # Тестируем VectorIndex
        await test_vector_index_performance()
        
        # Мониторим метрики после тестирования
        await monitor_metrics_during_test()
        
        logger.info("✅ Нагрузочное тестирование завершено")
        
        # Показываем итоговые метрики
        final_metrics = profiler.get_all_metrics()
        logger.info("📊 Итоговые метрики производительности:")
        for metric, data in final_metrics.items():
            if "avg_time_ms" in data:
                avg_time = data["avg_time_ms"]
                status = "✅" if avg_time < 1000 else "⚠️" if avg_time < 5000 else "❌"
                logger.info(f"  {status} {metric}: {avg_time:.2f}мс")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
