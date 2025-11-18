#!/usr/bin/env python3
"""
Скрипт для тестирования новых универсальных инструментов MCP сервера.

Тестирует:
- search (hybrid, smart, embedding, similar, trading)
- batch_operations (update, delete, fetch)
- graph_query (neighbors, path, related)
- background_indexing (start, stop, status)
- summaries (update, review)
- ingest (records, scraped)
- get_statistics (general, tags, indexing)
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory_mcp.mcp.server import call_tool
from memory_mcp.mcp.adapters import MemoryServiceAdapter
from memory_mcp.mcp.schema import (
    UnifiedSearchRequest,
    BatchOperationsRequest,
    GraphQueryRequest,
    BackgroundIndexingRequest,
    SummariesRequest,
    IngestRequest,
    GetStatisticsRequest,
    MemoryRecordPayload,
)


async def test_search_hybrid():
    """Тест универсального поиска: hybrid"""
    print("\n🔍 Тест 1: search (hybrid)")
    print("-" * 60)
    
    result = await call_tool("search", {
        "search_type": "hybrid",
        "query": "криптовалюты",
        "top_k": 5,
        "source": "telegram"
    })
    
    print(f"✅ Результат: {len(result[1].get('results', []))} результатов")
    if result[1].get('results'):
        print(f"   Первый результат: {result[1]['results'][0].get('record_id', 'N/A')[:50]}...")
    return result


async def test_search_smart():
    """Тест универсального поиска: smart"""
    print("\n🔍 Тест 2: search (smart)")
    print("-" * 60)
    
    try:
        result = await call_tool("search", {
            "search_type": "smart",
            "query": "обсуждение криптовалют",
            "top_k": 3
        })
        
        print(f"✅ Результат: session_id={result[1].get('session_id', 'N/A')}")
        print(f"   Confidence: {result[1].get('confidence_score', 0):.2f}")
        print(f"   Results: {len(result[1].get('results', []))}")
        return result
    except Exception as e:
        print(f"⚠️  Ошибка (возможно, нет LLM): {e}")
        return None


async def test_batch_operations_fetch():
    """Тест batch операций: fetch"""
    print("\n📦 Тест 3: batch_operations (fetch)")
    print("-" * 60)
    
    # Сначала найдем несколько record_id через поиск
    search_result = await call_tool("search", {
        "search_type": "hybrid",
        "query": "test",
        "top_k": 3
    })
    
    record_ids = [r.get('record_id') for r in search_result[1].get('results', [])[:2]]
    
    if not record_ids:
        print("⚠️  Нет записей для тестирования")
        return None
    
    result = await call_tool("batch_operations", {
        "operation": "fetch",
        "record_ids": record_ids
    })
    
    print(f"✅ Результат: найдено {result[1].get('total_found', 0)} записей")
    return result


async def test_batch_operations_update():
    """Тест batch операций: update"""
    print("\n📦 Тест 4: batch_operations (update)")
    print("-" * 60)
    
    # Найдем одну запись для обновления
    search_result = await call_tool("search", {
        "search_type": "hybrid",
        "query": "test",
        "top_k": 1
    })
    
    record_ids = [r.get('record_id') for r in search_result[1].get('results', [])[:1]]
    
    if not record_ids:
        print("⚠️  Нет записей для тестирования")
        return None
    
    result = await call_tool("batch_operations", {
        "operation": "update",
        "updates": [{
            "record_id": record_ids[0],
            "tags": ["test_tag", "unified_tools_test"]
        }]
    })
    
    print(f"✅ Результат: обновлено {result[1].get('total_updated', 0)} записей")
    return result


async def test_graph_query_neighbors():
    """Тест graph_query: neighbors"""
    print("\n🕸️  Тест 5: graph_query (neighbors)")
    print("-" * 60)
    
    # Найдем node_id через поиск
    search_result = await call_tool("search", {
        "search_type": "hybrid",
        "query": "test",
        "top_k": 1
    })
    
    node_id = None
    if search_result[1].get('results'):
        node_id = search_result[1]['results'][0].get('record_id')
    
    if not node_id:
        print("⚠️  Нет узлов для тестирования")
        return None
    
    result = await call_tool("graph_query", {
        "query_type": "neighbors",
        "node_id": node_id,
        "direction": "both"
    })
    
    print(f"✅ Результат: найдено {len(result[1].get('neighbors', []))} соседей")
    return result


async def test_background_indexing_status():
    """Тест background_indexing: status"""
    print("\n⚙️  Тест 6: background_indexing (status)")
    print("-" * 60)
    
    result = await call_tool("background_indexing", {
        "action": "status"
    })
    
    print(f"✅ Результат: running={result[1].get('running', False)}")
    print(f"   Message: {result[1].get('message', 'N/A')}")
    return result


async def test_summaries_review():
    """Тест summaries: review"""
    print("\n📝 Тест 7: summaries (review)")
    print("-" * 60)
    
    result = await call_tool("summaries", {
        "action": "review",
        "dry_run": True,
        "limit": 5
    })
    
    print(f"✅ Результат: обработано {result[1].get('files_processed', 0)} файлов")
    print(f"   Message: {result[1].get('message', 'N/A')}")
    return result


async def test_ingest_records():
    """Тест ingest: records"""
    print("\n📥 Тест 8: ingest (records)")
    print("-" * 60)
    
    test_record = MemoryRecordPayload(
        record_id=f"test_unified_{datetime.now(timezone.utc).timestamp()}",
        source="test",
        content="Тестовая запись для проверки универсального инструмента ingest",
        timestamp=datetime.now(timezone.utc),
        tags=["test", "unified_tools"],
        metadata={"test": True}
    )
    
    result = await call_tool("ingest", {
        "source_type": "records",
        "records": [test_record.model_dump()]
    })
    
    print(f"✅ Результат: загружено {result[1].get('records_ingested', 0)} записей")
    return result


async def test_get_statistics_general():
    """Тест get_statistics: general"""
    print("\n📊 Тест 9: get_statistics (general)")
    print("-" * 60)
    
    result = await call_tool("get_statistics", {
        "type": "general"
    })
    
    stats = result[1].get('graph_stats', {})
    print(f"✅ Результат: nodes={stats.get('nodes_count', 0)}, edges={stats.get('edges_count', 0)}")
    return result


async def test_get_statistics_tags():
    """Тест get_statistics: tags"""
    print("\n📊 Тест 10: get_statistics (tags)")
    print("-" * 60)
    
    result = await call_tool("get_statistics", {
        "type": "tags"
    })
    
    print(f"✅ Результат: total_tags={result[1].get('total_tags', 0)}")
    return result


async def test_get_statistics_indexing():
    """Тест get_statistics: indexing"""
    print("\n📊 Тест 11: get_statistics (indexing)")
    print("-" * 60)
    
    result = await call_tool("get_statistics", {
        "type": "indexing"
    })
    
    progress = result[1].get('indexing_progress', [])
    print(f"✅ Результат: чатов в прогрессе: {len(progress)}")
    return result


async def test_get_statistics_all():
    """Тест get_statistics: все типы"""
    print("\n📊 Тест 12: get_statistics (all)")
    print("-" * 60)
    
    result = await call_tool("get_statistics", {})
    
    print(f"✅ Результат: есть graph_stats={bool(result[1].get('graph_stats'))}")
    print(f"   есть tags_count={bool(result[1].get('tags_count'))}")
    print(f"   есть indexing_progress={bool(result[1].get('indexing_progress'))}")
    return result


async def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("🧪 Тестирование новых универсальных инструментов MCP")
    print("=" * 60)
    
    tests = [
        ("search (hybrid)", test_search_hybrid),
        ("search (smart)", test_search_smart),
        ("batch_operations (fetch)", test_batch_operations_fetch),
        ("batch_operations (update)", test_batch_operations_update),
        ("graph_query (neighbors)", test_graph_query_neighbors),
        ("background_indexing (status)", test_background_indexing_status),
        ("summaries (review)", test_summaries_review),
        ("ingest (records)", test_ingest_records),
        ("get_statistics (general)", test_get_statistics_general),
        ("get_statistics (tags)", test_get_statistics_tags),
        ("get_statistics (indexing)", test_get_statistics_indexing),
        ("get_statistics (all)", test_get_statistics_all),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ Ошибка в тесте '{name}': {e}")
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
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
        print("\n🎉 Все тесты пройдены успешно!")
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

