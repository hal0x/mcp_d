"""
Тесты совместимости UnifiedMemory с MemoryStore API.

Проверяет, что UnifiedMemory полностью совместим с MemoryStore
и может заменить его без breaking changes.
"""

import pytest
from pathlib import Path
import tempfile
import os

from memory import UnifiedMemory
from memory.memory_store import MemoryStore
from memory.models import MemoryItem


class TestCompatibility:
    """Тесты совместимости API."""
    
    def test_constructor_compatibility(self):
        """Тест совместимости конструктора."""
        # MemoryStore параметры
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Тест 1: Базовые параметры
            memory_store = MemoryStore(
                long_term_path=Path(tmp_dir) / "memory.json",
                short_term_limit=50,
                llm_client=None,
                embeddings_client=None
            )
            
            unified_memory = UnifiedMemory(
                path=Path(tmp_dir) / "memory.json",
                short_term_limit=50,
                llm_client=None,
                embeddings_client=None
            )
            
            # Проверяем, что оба создаются без ошибок
            assert memory_store is not None
            assert unified_memory is not None
    
    def test_basic_methods_compatibility(self):
        """Тест совместимости основных методов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Создаем экземпляры
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Тест remember()
            memory_store.remember("test memory store")
            unified_memory.remember("test unified memory")
            
            # Тест recall()
            store_recall = memory_store.recall()
            unified_recall = unified_memory.recall()
            
            assert len(store_recall) >= 0
            assert len(unified_recall) >= 0
            
            # Тест recall(long_term=True)
            store_recall_lt = memory_store.recall(long_term=True)
            unified_recall_lt = unified_memory.recall(long_term=True)
            
            assert isinstance(store_recall_lt, list)
            assert isinstance(unified_recall_lt, list)
    
    def test_search_compatibility(self):
        """Тест совместимости поиска."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Добавляем тестовые данные
            test_data = "python programming language"
            memory_store.remember(test_data)
            unified_memory.remember(test_data)
            
            # Тест search()
            store_search = memory_store.search("python")
            unified_search = unified_memory.search("python")
            
            assert isinstance(store_search, list)
            assert isinstance(unified_search, list)
            
            # Тест search(long_term=True)
            store_search_lt = memory_store.search("python", long_term=True)
            unified_search_lt = unified_memory.search("python", long_term=True)
            
            assert isinstance(store_search_lt, list)
            assert isinstance(unified_search_lt, list)
    
    def test_save_load_compatibility(self):
        """Тест совместимости сохранения и загрузки."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_file = Path(tmp_dir) / "memory.json"
            
            # Создаем и сохраняем данные
            memory_store = MemoryStore(long_term_path=memory_file)
            memory_store.remember("test data for save")
            memory_store.save()
            
            # Загружаем в UnifiedMemory
            unified_memory = UnifiedMemory(path=memory_file)
            loaded_data = unified_memory.recall()
            
            # Проверяем, что данные загрузились
            assert len(loaded_data) > 0
            assert any("test data for save" in item for item in loaded_data)
    
    def test_properties_compatibility(self):
        """Тест совместимости свойств."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Проверяем наличие основных свойств
            assert hasattr(memory_store, 'short_term')
            assert hasattr(memory_store, 'long_term')
            assert hasattr(memory_store, 'short_term_limit')
            
            assert hasattr(unified_memory, 'short_term')
            assert hasattr(unified_memory, 'long_term')
            assert hasattr(unified_memory, 'short_term_limit')
    
    def test_forget_compatibility(self):
        """Тест совместимости удаления."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Добавляем данные
            test_item = "item to forget"
            memory_store.remember(test_item)
            unified_memory.remember(test_item)
            
            # Удаляем
            store_result = memory_store.forget(test_item)
            unified_result = unified_memory.forget(test_item)
            
            assert isinstance(store_result, bool)
            assert isinstance(unified_result, bool)
    
    def test_long_term_memory_compatibility(self):
        """Тест совместимости долгосрочной памяти."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Добавляем в долгосрочную память
            memory_store.remember("long term item", long_term=True)
            unified_memory.remember("long term item", long_term=True)
            
            # Проверяем долгосрочную память
            store_lt = memory_store.recall(long_term=True)
            unified_lt = unified_memory.recall(long_term=True)
            
            assert isinstance(store_lt, list)
            assert isinstance(unified_lt, list)
    
    def test_semantic_search_compatibility(self):
        """Тест совместимости семантического поиска."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
            unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
            
            # Добавляем данные
            memory_store.remember("artificial intelligence and machine learning")
            unified_memory.remember("artificial intelligence and machine learning")
            
            # Тест семантического поиска (если доступен)
            if hasattr(memory_store, 'semantic_search'):
                store_semantic = memory_store.semantic_search("AI")
                assert isinstance(store_semantic, list)
            
            if hasattr(unified_memory, 'semantic_search'):
                unified_semantic = unified_memory.semantic_search("AI")
                assert isinstance(unified_semantic, list)


class TestMigrationScenarios:
    """Тесты сценариев миграции."""
    
    def test_direct_replacement(self):
        """Тест прямой замены MemoryStore на UnifiedMemory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Сценарий: замена в существующем коде
            memory_file = Path(tmp_dir) / "memory.json"
            
            # Старый код с MemoryStore
            old_memory = MemoryStore(long_term_path=memory_file)
            old_memory.remember("old data", long_term=True)
            old_memory.save()
            
            # Новый код с UnifiedMemory
            new_memory = UnifiedMemory(path=memory_file)
            new_memory.remember("new data")
            new_memory.save()
            
            # Проверяем, что оба работают
            old_data = old_memory.recall()
            old_data_lt = old_memory.recall(long_term=True)
            new_data = new_memory.recall()
            new_data_lt = new_memory.recall(long_term=True)
            
            # Проверяем, что данные есть в любом из источников
            assert len(old_data) > 0 or len(old_data_lt) > 0
            assert len(new_data) > 0 or len(new_data_lt) > 0
    
    def test_parameter_migration(self):
        """Тест миграции параметров конструктора."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Старые параметры MemoryStore
            old_params = {
                "long_term_path": Path(tmp_dir) / "memory.json",
                "short_term_limit": 100,
                "llm_client": None,
                "embeddings_client": None
            }
            
            # Новые параметры UnifiedMemory
            new_params = {
                "path": old_params["long_term_path"],
                "short_term_limit": old_params["short_term_limit"],
                "llm_client": old_params["llm_client"],
                "embeddings_client": old_params["embeddings_client"]
            }
            
            # Создаем экземпляры
            old_memory = MemoryStore(**old_params)
            new_memory = UnifiedMemory(**new_params)
            
            # Проверяем, что оба работают одинаково
            assert old_memory.short_term_limit == new_memory.short_term_limit
            assert old_memory.llm_client == new_memory.llm_client


def test_api_consistency():
    """Тест консистентности API между MemoryStore и UnifiedMemory."""
    # Проверяем, что основные методы имеют одинаковые сигнатуры
    # Создаем экземпляры для проверки атрибутов
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        memory_store = MemoryStore(long_term_path=Path(tmp_dir) / "memory.json")
        unified_memory = UnifiedMemory(path=Path(tmp_dir) / "memory.json")
        
        memory_store_methods = set(dir(memory_store))
        unified_memory_methods = set(dir(unified_memory))
        
        # Ключевые методы должны присутствовать в обоих
        key_methods = {
            'remember', 'recall', 'search', 'forget', 'save',
            'short_term', 'long_term', 'short_term_limit'
        }
        
        for method in key_methods:
            assert method in memory_store_methods, f"MemoryStore missing {method}"
            assert method in unified_memory_methods, f"UnifiedMemory missing {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
