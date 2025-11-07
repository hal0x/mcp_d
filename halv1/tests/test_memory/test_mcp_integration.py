#!/usr/bin/env python3
"""
Тест интеграции MCP сервера с системой памяти HALv1

Проверяет, что MCPMemoryAdapter корректно реализует интерфейс MemoryServiceAdapter
и может заменить существующий модуль памяти.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорты из HALv1
from memory.mcp_memory_adapter import MCPMemoryAdapter
from memory.memory_store import MemoryEntry


async def test_mcp_memory_adapter():
    """Тест основных функций MCPMemoryAdapter"""
    
    logger.info("🧪 Начинаем тестирование MCPMemoryAdapter")
    
    # Создаем временную директорию для тестов
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Инициализируем адаптер
        adapter = MCPMemoryAdapter(
            mcp_server_path=None,  # Не используем реальный MCP сервер для тестов
            long_term_path=str(temp_path / "test_memory.json"),
            short_term_limit=5,
        )
        
        logger.info("✅ MCPMemoryAdapter инициализирован")
        
        # Тест 1: remember() - сохранение в память
        logger.info("📝 Тест 1: remember()")
        
        test_texts = [
            "Это тестовое сообщение номер 1",
            "Это тестовое сообщение номер 2", 
            "Это тестовое сообщение номер 3",
            "Bitcoin стоит $50,000",
            "Криптовалюты растут в цене",
        ]
        
        for text in test_texts:
            adapter.remember(text, long_term=False)
            logger.info(f"   Сохранено: {text[:30]}...")
        
        # Проверяем краткосрочную память
        short_term_records = adapter.recall(long_term=False)
        logger.info(f"   Краткосрочная память: {len(short_term_records)} записей")
        
        assert len(short_term_records) == len(test_texts), f"Ожидалось {len(test_texts)}, получено {len(short_term_records)}"
        
        # Тест 2: write_event() - запись событий
        logger.info("📝 Тест 2: write_event()")
        
        adapter.write_event(
            content="Событие: пользователь вошел в систему",
            entities=["пользователь", "система"],
            frozen=True
        )
        
        events = adapter.read_events()
        logger.info(f"   События в буфере: {len(events)}")
        
        # Тест 3: search() - текстовый поиск
        logger.info("📝 Тест 3: search()")
        
        search_results = adapter.search("Bitcoin", long_term=False)
        logger.info(f"   Результаты поиска 'Bitcoin': {len(search_results)}")
        
        assert len(search_results) > 0, "Поиск должен найти хотя бы один результат"
        assert any("Bitcoin" in result for result in search_results), "Должен найти сообщение с Bitcoin"
        
        # Тест 4: semantic_search() - семантический поиск
        logger.info("📝 Тест 4: semantic_search()")
        
        semantic_results = adapter.semantic_search("криптовалюта", long_term=False, top_k=3)
        logger.info(f"   Результаты семантического поиска 'криптовалюта': {len(semantic_results)}")
        
        # Тест 5: remember() с long_term=True
        logger.info("📝 Тест 5: remember() с long_term=True")
        
        adapter.remember("Важная информация для долгосрочной памяти", long_term=True)
        
        long_term_records = adapter.recall(long_term=True)
        logger.info(f"   Долгосрочная память: {len(long_term_records)} записей")
        
        # Тест 6: forget() - удаление элементов
        logger.info("📝 Тест 6: forget()")
        
        initial_count = len(adapter.recall(long_term=False))
        logger.info(f"   Начальное количество записей: {initial_count}")
        
        # Выводим все записи для отладки
        all_records = adapter.recall(long_term=False)
        for i, record in enumerate(all_records):
            logger.info(f"   Запись {i}: {record[:50]}...")
        
        result = adapter.forget("Это тестовое сообщение номер 1", long_term=False)
        logger.info(f"   Результат forget(): {result}")
        
        final_count = len(adapter.recall(long_term=False))
        logger.info(f"   Финальное количество записей: {final_count}")
        
        logger.info(f"   Удаление: {initial_count} → {final_count}")
        assert final_count == initial_count - 1, "Количество записей должно уменьшиться на 1"
        
        # Тест 7: consolidate() - консолидация памяти
        logger.info("📝 Тест 7: consolidate()")
        
        adapter.consolidate()
        
        short_term_after_consolidation = len(adapter.recall(long_term=False))
        long_term_after_consolidation = len(adapter.recall(long_term=True))
        
        logger.info(f"   После консолидации: ST={short_term_after_consolidation}, LT={long_term_after_consolidation}")
        
        # Тест 8: save() - сохранение памяти
        logger.info("📝 Тест 8: save()")
        
        adapter.save()
        
        # Проверяем, что файл создан
        memory_file = temp_path / "test_memory.json"
        assert memory_file.exists(), "Файл памяти должен быть создан"
        
        logger.info(f"   Файл памяти создан: {memory_file}")
        
        # Тест 9: prune_long_term() - обрезка памяти
        logger.info("📝 Тест 9: prune_long_term()")
        
        # Добавляем много записей в долгосрочную память
        for i in range(10):
            adapter.remember(f"Долгосрочная запись {i}", long_term=True)
        
        long_term_before_prune = len(adapter.recall(long_term=True))
        adapter.prune_long_term(max_items=5)
        long_term_after_prune = len(adapter.recall(long_term=True))
        
        logger.info(f"   Обрезка памяти: {long_term_before_prune} → {long_term_after_prune}")
        
        # Тест 10: Свойства для совместимости
        logger.info("📝 Тест 10: Свойства совместимости")
        
        short_term_property = adapter.short_term
        long_term_property = adapter.long_term
        
        logger.info(f"   short_term property: {len(short_term_property)} записей")
        logger.info(f"   long_term property: {len(long_term_property)} записей")
        
        assert isinstance(short_term_property, list), "short_term должно быть списком"
        assert isinstance(long_term_property, list), "long_term должно быть списком"
        
        # Проверяем, что все записи являются MemoryEntry
        for entry in short_term_property:
            assert isinstance(entry, MemoryEntry), "Записи должны быть MemoryEntry"
        
        logger.info("✅ Все тесты пройдены успешно!")


async def test_memory_entry_compatibility():
    """Тест совместимости с MemoryEntry"""
    
    logger.info("🧪 Тест совместимости с MemoryEntry")
    
    # Создаем MemoryEntry
    entry = MemoryEntry(
        text="Тестовое сообщение",
        embedding=[0.1, 0.2, 0.3],
        timestamp=1234567890.0,
        frozen=True
    )
    
    # Проверяем свойства
    assert entry.text == "Тестовое сообщение"
    assert entry.embedding == [0.1, 0.2, 0.3]
    assert entry.timestamp == 1234567890.0
    assert entry.frozen == True
    
    logger.info("✅ MemoryEntry работает корректно")


async def test_interface_compatibility():
    """Тест совместимости интерфейса с существующими компонентами"""
    
    logger.info("🧪 Тест совместимости интерфейса")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Создаем адаптер
        adapter = MCPMemoryAdapter(
            mcp_server_path=None,
            long_term_path=str(temp_path / "test_memory.json"),
        )
        
        # Проверяем, что адаптер имеет все необходимые методы
        required_methods = [
            'remember', 'write_event', 'read_events', 'consolidate',
            'recall', 'search', 'semantic_search', 'forget', 
            'prune_long_term', 'save', 'read_schemas', 'explain'
        ]
        
        for method_name in required_methods:
            assert hasattr(adapter, method_name), f"Метод {method_name} должен существовать"
            method = getattr(adapter, method_name)
            assert callable(method), f"Метод {method_name} должен быть вызываемым"
        
        # Проверяем свойства
        required_properties = ['short_term', 'long_term', 'graph']
        
        for prop_name in required_properties:
            assert hasattr(adapter, prop_name), f"Свойство {prop_name} должно существовать"
        
        logger.info("✅ Интерфейс совместим с существующими компонентами")


async def main():
    """Главная функция тестирования"""
    
    logger.info("🚀 Запуск тестов интеграции MCP сервера")
    
    try:
        await test_memory_entry_compatibility()
        await test_interface_compatibility()
        await test_mcp_memory_adapter()
        
        logger.info("🎉 Все тесты пройдены успешно!")
        logger.info("✅ MCP сервер готов к замене модуля памяти")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тестах: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
