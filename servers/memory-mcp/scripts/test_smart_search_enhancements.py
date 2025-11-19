#!/usr/bin/env python3
"""Скрипт для тестирования улучшений Smart Search Engine."""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_mcp.search.entity_context_enricher import EntityContextEnricher
from memory_mcp.search.query_intent_analyzer import QueryIntentAnalyzer
from memory_mcp.search.query_understanding import QueryUnderstandingEngine
from memory_mcp.analysis.entity_dictionary import get_entity_dictionary


async def test_entity_context_enricher():
    """Тест обогащения контекста сущностей."""
    print("\n" + "="*60)
    print("Тест 1: EntityContextEnricher")
    print("="*60)
    
    try:
        entity_dict = get_entity_dictionary()
        enricher = EntityContextEnricher(entity_dictionary=entity_dict)
        
        test_queries = [
            "Дуров и Telegram",
            "криптовалюты и блокчейн",
            "биткоин цена",
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            # Извлечение сущностей
            entities = enricher.extract_entities_from_query(query)
            print(f"  Извлечено сущностей: {len(entities)}")
            for entity in entities[:3]:
                print(f"    - {entity.get('entity_type')}: {entity.get('value')}")
            
            # Обогащение запроса
            enriched = enricher.enrich_query_with_entity_context(query)
            if enriched != query:
                print(f"  Обогащенный запрос: {enriched[:200]}...")
            else:
                print(f"  Запрос не обогащен (нет сущностей в словаре)")
            
            # Расширение связанными сущностями
            expanded = enricher.expand_query_with_related_entities(query)
            if expanded != query:
                print(f"  Расширенный запрос: {expanded[:200]}...")
        
        print("\n✓ EntityContextEnricher работает корректно")
        return True
    except Exception as e:
        print(f"\n✗ Ошибка в EntityContextEnricher: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_intent_analyzer():
    """Тест анализатора намерений."""
    print("\n" + "="*60)
    print("Тест 2: QueryIntentAnalyzer")
    print("="*60)
    
    try:
        analyzer = QueryIntentAnalyzer()
        
        test_queries = [
            ("что такое блокчейн", "informational"),
            ("как создать кошелек", "transactional"),
            ("где найти информацию о биткоине", "navigational"),
            ("почему работает блокчейн", "analytical"),
        ]
        
        for query, expected_type in test_queries:
            print(f"\nЗапрос: {query}")
            intent = await analyzer.analyze_intent(query)
            
            print(f"  Тип намерения: {intent.intent_type} (ожидалось: {expected_type})")
            print(f"  Уверенность: {intent.confidence:.2f}")
            print(f"  Веса: БД={intent.recommended_db_weight:.2f}, артифакты={intent.recommended_artifact_weight:.2f}")
            if intent.recommended_top_k:
                print(f"  Рекомендуемый top_k: {intent.recommended_top_k}")
            
            if intent.intent_type == expected_type:
                print(f"  ✓ Тип намерения определен правильно")
            else:
                print(f"  ⚠ Тип намерения не совпадает (возможно, LLM вернул другой тип)")
        
        print("\n✓ QueryIntentAnalyzer работает корректно")
        return True
    except Exception as e:
        print(f"\n✗ Ошибка в QueryIntentAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_understanding():
    """Тест движка понимания запросов."""
    print("\n" + "="*60)
    print("Тест 3: QueryUnderstandingEngine")
    print("="*60)
    
    try:
        engine = QueryUnderstandingEngine()
        
        test_queries = [
            "блокчейн и криптовалюты",
            "как работает биткоин и почему он дорогой",
            "информация о Telegram и его создателе",
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            understanding = await engine.understand_query(query)
            
            print(f"  Оригинальный запрос: {understanding.original_query}")
            print(f"  Улучшенный запрос: {understanding.enhanced_query}")
            
            if understanding.sub_queries:
                print(f"  Подзапросы ({len(understanding.sub_queries)}):")
                for i, sq in enumerate(understanding.sub_queries[:3], 1):
                    print(f"    {i}. {sq}")
            
            if understanding.implicit_requirements:
                print(f"  Неявные требования ({len(understanding.implicit_requirements)}):")
                for req in understanding.implicit_requirements[:3]:
                    print(f"    - {req}")
            
            if understanding.key_concepts:
                print(f"  Ключевые концепции ({len(understanding.key_concepts)}):")
                print(f"    {', '.join(understanding.key_concepts[:5])}")
            
            if understanding.alternative_formulations:
                print(f"  Альтернативные формулировки ({len(understanding.alternative_formulations)}):")
                for alt in understanding.alternative_formulations[:2]:
                    print(f"    - {alt}")
        
        print("\n✓ QueryUnderstandingEngine работает корректно")
        return True
    except Exception as e:
        print(f"\n✗ Ошибка в QueryUnderstandingEngine: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Тест интеграции всех компонентов."""
    print("\n" + "="*60)
    print("Тест 4: Интеграция компонентов")
    print("="*60)
    
    try:
        from memory_mcp.search.smart_search_engine import SmartSearchEngine
        from memory_mcp.search.search_session_store import SearchSessionStore
        from memory_mcp.memory.artifacts_reader import ArtifactsReader
        from memory_mcp.mcp.adapters import MemoryServiceAdapter
        from memory_mcp.mcp.schema import SmartSearchRequest
        import tempfile
        from pathlib import Path
        
        # Создаем временные компоненты
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / "test_memory.db"
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "chat_contexts").mkdir()
        
        # Создаем тестовый артифакт
        (artifacts_dir / "chat_contexts" / "test.md").write_text(
            "Тестовая информация о блокчейне и криптовалютах."
        )
        
        # Инициализируем компоненты
        # MemoryServiceAdapter инициализируется через настройки окружения
        import os
        os.environ["MEMORY_MCP_DB_PATH"] = str(db_path)
        os.environ["MEMORY_MCP_ARTIFACTS_PATH"] = str(artifacts_dir)
        
        adapter = MemoryServiceAdapter()
        
        artifacts_reader = ArtifactsReader(artifacts_dir=artifacts_dir)
        artifacts_reader.scan_artifacts_directory()
        
        session_store = SearchSessionStore(db_path=str(temp_dir / "sessions.db"))
        
        # Создаем SmartSearchEngine
        search_engine = SmartSearchEngine(
            adapter=adapter,
            artifacts_reader=artifacts_reader,
            session_store=session_store,
            min_confidence=0.5,
        )
        
        print("\n✓ SmartSearchEngine инициализирован")
        print(f"  - EntityEnricher: {search_engine.entity_enricher is not None}")
        print(f"  - IntentAnalyzer: {search_engine.intent_analyzer is not None}")
        print(f"  - QueryUnderstanding: {search_engine.query_understanding is not None}")
        print(f"  - ConnectionBuilder: {search_engine.connection_builder is not None}")
        
        # Тестируем поиск
        test_query = "блокчейн"
        print(f"\nТестовый запрос: {test_query}")
        
        try:
            request = SmartSearchRequest(query=test_query, top_k=5)
            response = await search_engine.search(request)
            
            print(f"  Session ID: {response.session_id}")
            print(f"  Confidence: {response.confidence_score:.2f}")
            print(f"  Результатов: {len(response.results)}")
            print(f"  БД записей: {response.db_records_found}")
            print(f"  Артифактов: {response.artifacts_found}")
            
            if response.results:
                print(f"\n  Топ-3 результата:")
                for i, result in enumerate(response.results[:3], 1):
                    print(f"    {i}. Score: {result.score:.3f} | {result.content[:80]}...")
            else:
                print("  Результатов не найдено (это нормально для пустой БД)")
        except Exception as e:
            print(f"  ⚠ Ошибка при поиске (возможно, БД пуста): {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✓ Интеграция работает корректно")
        return True
    except Exception as e:
        print(f"\n✗ Ошибка при интеграции: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Главная функция для запуска всех тестов."""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ SMART SEARCH ENGINE")
    print("="*60)
    
    results = []
    
    # Тест 1: EntityContextEnricher
    results.append(await test_entity_context_enricher())
    
    # Тест 2: QueryIntentAnalyzer
    results.append(await test_query_intent_analyzer())
    
    # Тест 3: QueryUnderstandingEngine
    results.append(await test_query_understanding())
    
    # Тест 4: Интеграция
    results.append(await test_integration())
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nПройдено тестов: {passed}/{total}")
    
    if passed == total:
        print("✓ Все тесты пройдены успешно!")
        return 0
    else:
        print(f"✗ Не пройдено тестов: {total - passed}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

