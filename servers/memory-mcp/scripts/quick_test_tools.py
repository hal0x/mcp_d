#!/usr/bin/env python3
"""
Быстрая проверка всех новых универсальных инструментов.

Запускает минимальные тесты для каждого инструмента.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory_mcp.mcp.server import call_tool


async def quick_test():
    """Быстрая проверка всех инструментов"""
    print("🚀 Быстрая проверка универсальных инструментов\n")
    
    tests = [
        ("search (hybrid)", "search", {"search_type": "hybrid", "query": "test", "top_k": 1}),
        ("batch_operations (fetch)", "batch_operations", {"operation": "fetch", "record_ids": ["test"]}),
        ("graph_query (neighbors)", "graph_query", {"query_type": "neighbors", "node_id": "test"}),
        ("background_indexing (status)", "background_indexing", {"action": "status"}),
        ("summaries (review)", "summaries", {"action": "review", "dry_run": True, "limit": 1}),
        ("get_statistics (general)", "get_statistics", {"type": "general"}),
    ]
    
    results = []
    for name, tool_name, args in tests:
        try:
            result = await call_tool(tool_name, args)
            results.append((name, True))
            print(f"✅ {name}")
        except Exception as e:
            results.append((name, False))
            print(f"❌ {name}: {str(e)[:50]}")
    
    print(f"\n✅ Успешно: {sum(1 for _, s in results if s)}/{len(results)}")
    print("🎉 Проверка завершена!")


if __name__ == "__main__":
    asyncio.run(quick_test())

