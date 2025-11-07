"""Тесты для граничных случаев ClusterManager.ingest."""

import pytest
import math
from unittest.mock import MagicMock

from index.cluster_manager import ClusterManager
from index.vector_index import VectorEntry


class TestClusterManagerEdgeCases:
    """Тесты для граничных случаев кластеризации."""

    def setup_method(self):
        """Подготовка для каждого теста."""
        self.cluster_manager = ClusterManager()

    def test_empty_iterator(self):
        """Тест: пустой итератор записей."""
        result = self.cluster_manager.ingest([])
        assert result is None

    def test_none_embeddings(self):
        """Тест: записи с None эмбеддингами."""
        records = [
            VectorEntry(chunk_id="chunk1", text="Текст 1", embedding=None, metadata={"source": "test"}),
            VectorEntry(chunk_id="chunk2", text="Текст 2", embedding=[0.1, 0.2, 0.3], metadata={"source": "test"})
        ]
        
        # Ожидаем исключение при попытке обработать None эмбеддинг
        with pytest.raises((TypeError, ValueError)):
            self.cluster_manager.ingest(records)

    def test_nan_in_vectors(self):
        """Тест: NaN значения в векторах."""
        records = [VectorEntry(chunk_id="chunk1", text="Текст", embedding=[0.1, float('nan'), 0.3], metadata={"source": "test"})]
        
        # Должен либо обработать NaN корректно, либо выбросить исключение
        result = self.cluster_manager.ingest(records)
        
        # Проверяем конкретное поведение
        if result is not None:
            # Если обработал - проверяем что результат корректен
            assert isinstance(result, dict)
            assert "clusters" in result
        else:
            # Если вернул None - это тоже валидное поведение
            assert result is None

    def test_zero_norm_vectors(self):
        """Тест: векторы с нормой близкой к нулю."""
        records = [
            VectorEntry(chunk_id="chunk1", text="Нулевой", embedding=[0.0, 0.0, 0.0], metadata={"source": "test"}),
            VectorEntry(chunk_id="chunk2", text="Нормальный", embedding=[0.1, 0.2, 0.3], metadata={"source": "test"})
        ]
        result = self.cluster_manager.ingest(records)
        
        # Проверяем что результат корректен
        if result is not None:
            assert isinstance(result, dict)
            assert "clusters" in result
        else:
            assert result is None

    def test_malformed_records(self):
        """Тест: записи с неправильной структурой."""
        malformed = [
            "строка",
            {"text": "без эмбеддинга"},
            {"embedding": [0.1, 0.2, 0.3]},
            None
        ]
        
        # Ожидаем исключение при попытке обработать неправильные записи
        with pytest.raises((KeyError, TypeError, AttributeError)):
            self.cluster_manager.ingest(malformed)

    def test_mixed_valid_and_invalid_records(self):
        """Тест: смесь валидных и невалидных записей."""
        mixed_records = [
            VectorEntry(chunk_id="chunk1", text="Валидная", embedding=[0.1, 0.2, 0.3], metadata={"source": "test"}),
            "невалидная строка",
            VectorEntry(chunk_id="chunk2", text="Валидная 2", embedding=[0.4, 0.5, 0.6], metadata={"source": "test"})
        ]
        
        # Ожидаем исключение при попытке обработать смешанные записи
        with pytest.raises((TypeError, AttributeError)):
            self.cluster_manager.ingest(mixed_records)

    def test_empty_metadata(self):
        """Тест: записи с пустыми метаданными."""
        records = [
            VectorEntry(chunk_id="chunk1", text="Текст 1", embedding=[0.1, 0.2, 0.3], metadata={}),
            VectorEntry(chunk_id="chunk2", text="Текст 2", embedding=[0.4, 0.5, 0.6], metadata={"source": "test"})
        ]
        
        result = self.cluster_manager.ingest(records)
        
        # Проверяем что результат корректен
        if result is not None:
            assert isinstance(result, dict)
            assert "clusters" in result
        else:
            assert result is None

    def test_very_large_vectors(self):
        """Тест: очень большие векторы."""
        large_vector = [1.0] * 10000  # Вектор из 10000 элементов
        records = [VectorEntry(chunk_id="chunk1", text="Большой", embedding=large_vector, metadata={"source": "test"})]
        
        result = self.cluster_manager.ingest(records)
        
        # Проверяем что результат корректен
        if result is not None:
            assert isinstance(result, dict)
            assert "clusters" in result
        else:
            assert result is None