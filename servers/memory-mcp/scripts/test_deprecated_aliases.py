#!/usr/bin/env python3
"""
Скрипт для проверки deprecated алиасов и их соответствия новым универсальным инструментам.

Проверяет, что старые инструменты работают и логируют предупреждения.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory_mcp.mcp.server import call_tool
import logging

# Настраиваем логирование для проверки предупреждений
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


async def test_deprecated_ingest_records():
    """Тест deprecated инструмента ingest_records"""
    print("\n📥 Тест deprecated: ingest_records")
    print("-" * 60)
    
    result = await call_tool("ingest_records", {
        "records": [{
            "record_id": "test_deprecated_1",
            "source": "test",
            "content": "Тест deprecated ingest_records",
            "timestamp": "2025-01-01T00:00:00Z",
            "tags": ["test"]
        }]
    })
    
    print(f"✅ Результат: загружено {result[1].get('records_ingested', 0)} записей")
    return result


async def test_deprecated_search_memory():
    """Тест deprecated инструмента search_memory"""
    print("\n🔍 Тест deprecated: search_memory")
    print("-" * 60)
    
    result = await call_tool("search_memory", {
        "query": "test",
        "top_k": 3
    })
    
    print(f"✅ Результат: {len(result[1].get('results', []))} результатов")
    return result


async def test_deprecated_batch_update_records():
    """Тест deprecated инструмента batch_update_records"""
    print("\n📦 Тест deprecated: batch_update_records")
    print("-" * 60)
    
    # Сначала найдем запись
    search_result = await call_tool("search_memory", {
        "query": "test",
        "top_k": 1
    })
    
    record_ids = [r.get('record_id') for r in search_result[1].get('results', [])[:1]]
    
    if not record_ids:
        print("⚠️  Нет записей для тестирования")
        return None
    
    result = await call_tool("batch_update_records", {
        "updates": [{
            "record_id": record_ids[0],
            "tags": ["deprecated_test"]
        }]
    })
    
    print(f"✅ Результат: обновлено {result[1].get('total_updated', 0)} записей")
    return result


async def test_deprecated_get_graph_neighbors():
    """Тест deprecated инструмента get_graph_neighbors"""
    print("\n🕸️  Тест deprecated: get_graph_neighbors")
    print("-" * 60)
    
    # Найдем node_id
    search_result = await call_tool("search_memory", {
        "query": "test",
        "top_k": 1
    })
    
    node_id = None
    if search_result[1].get('results'):
        node_id = search_result[1]['results'][0].get('record_id')
    
    if not node_id:
        print("⚠️  Нет узлов для тестирования")
        return None
    
    result = await call_tool("get_graph_neighbors", {
        "node_id": node_id,
        "direction": "both"
    })
    
    print(f"✅ Результат: найдено {len(result[1].get('neighbors', []))} соседей")
    return result


async def test_deprecated_get_tags_statistics():
    """Тест deprecated инструмента get_tags_statistics"""
    print("\n📊 Тест deprecated: get_tags_statistics")
    print("-" * 60)
    
    result = await call_tool("get_tags_statistics", {})
    
    print(f"✅ Результат: total_tags={result[1].get('total_tags', 0)}")
    return result


async def test_deprecated_update_summaries():
    """Тест deprecated инструмента update_summaries"""
    print("\n📝 Тест deprecated: update_summaries")
    print("-" * 60)
    
    result = await call_tool("update_summaries", {
        "chat": None,
        "force": False
    })
    
    print(f"✅ Результат: обновлено {result[1].get('chats_updated', 0)} чатов")
    return result


async def test_deprecated_start_background_indexing():
    """Тест deprecated инструмента start_background_indexing"""
    print("\n⚙️  Тест deprecated: start_background_indexing")
    print("-" * 60)
    
    result = await call_tool("start_background_indexing", {})
    
    print(f"✅ Результат: success={result[1].get('success', False)}")
    print(f"   Message: {result[1].get('message', 'N/A')[:50]}...")
    return result


async def main():
    """Запуск всех тестов deprecated алиасов"""
    print("=" * 60)
    print("🧪 Тестирование deprecated алиасов")
    print("=" * 60)
    print("\n⚠️  Ожидаются предупреждения в логах о deprecated инструментах")
    
    tests = [
        ("ingest_records", test_deprecated_ingest_records),
        ("search_memory", test_deprecated_search_memory),
        ("batch_update_records", test_deprecated_batch_update_records),
        ("get_graph_neighbors", test_deprecated_get_graph_neighbors),
        ("get_tags_statistics", test_deprecated_get_tags_statistics),
        ("update_summaries", test_deprecated_update_summaries),
        ("start_background_indexing", test_deprecated_start_background_indexing),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ Ошибка в тесте '{name}': {e}")
            results.append((name, False, str(e)))
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📈 Итоговая статистика")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if error:
            print(f"   Ошибка: {error[:100]}")
    
    print(f"\n✅ Успешно: {passed}/{len(results)}")
    print(f"❌ Ошибок: {failed}/{len(results)}")
    
    if failed == 0:
        print("\n🎉 Все deprecated алиасы работают корректно!")
        print("💡 Проверьте логи выше на наличие предупреждений о deprecated инструментах")
    else:
        print(f"\n⚠️  {failed} тест(ов) завершились с ошибками")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

