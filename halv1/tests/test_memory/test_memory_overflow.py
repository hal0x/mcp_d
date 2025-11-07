"""
Тесты для автоматической суммаризации при переполнении short_term памяти.
Покрывает проблему F-002 из отчёта аудита.
"""

import pytest
from unittest.mock import MagicMock, patch

from memory import UnifiedMemory
from memory.memory_store import MemoryEntry


class TestMemoryOverflow:
    """Тесты для автоматической суммаризации памяти."""

    def setup_method(self):
        """Подготовка для каждого теста."""
        # Создаём мок для LLM клиента
        self.llm_client = MagicMock()
        self.llm_client.generate.return_value = "Суммаризированный текст"
        
        # Создаём UnifiedMemory с моком LLM клиента
        self.memory = UnifiedMemory(llm_client=self.llm_client)
        self.memory.short_term_limit = 3  # Устанавливаем лимит = 3

    def test_no_overflow_within_limit(self):
        """Тест: нет переполнения при добавлении в пределах лимита."""
        # Добавляем записи в пределах лимита
        self.memory.remember("Первая запись")
        self.memory.remember("Вторая запись")
        self.memory.remember("Третья запись")
        
        assert len(self.memory.short_term) == 3
        assert len(self.memory.long_term) == 0
        self.llm_client.generate.assert_not_called()

    def test_overflow_triggers_summarization(self):
        """Тест: переполнение запускает суммаризацию."""
        # Заполняем до лимита
        self.memory.remember("Первая запись")
        self.memory.remember("Вторая запись")
        self.memory.remember("Третья запись")
        
        # Добавляем ещё одну запись - должно сработать переполнение
        self.memory.remember("Четвёртая запись")
        
        # Проверяем что произошла суммаризация
        # После переполнения short_term очищается полностью
        assert len(self.memory.short_term) == 0
        assert len(self.memory.long_term) == 1   # Суммаризированный текст
        self.llm_client.generate.assert_called_once()

    def test_overflow_exactly_at_limit(self):
        """Тест: переполнение ровно на границе лимита."""
        # Устанавливаем лимит = 2
        self.memory.short_term_limit = 2
        
        self.memory.remember("Первая")
        self.memory.remember("Вторая")
        assert len(self.memory.short_term) == 2
        
        # Добавляем третью - должно сработать переполнение
        self.memory.remember("Третья")
        # После переполнения short_term очищается полностью
        assert len(self.memory.short_term) == 0
        assert len(self.memory.long_term) == 1  # Суммаризированный текст

    def test_overflow_exceeds_by_one(self):
        """Тест: превышение лимита на единицу."""
        self.memory.short_term_limit = 1
        
        self.memory.remember("Первая")
        self.memory.remember("Вторая")  # Переполнение
        
        # После переполнения short_term очищается полностью
        assert len(self.memory.short_term) == 0
        assert len(self.memory.long_term) == 1  # Суммаризированный текст

    def test_massive_overflow(self):
        """Тест: массовое добавление записей."""
        self.memory.short_term_limit = 2
        
        # Добавляем много записей подряд
        for i in range(10):
            self.memory.remember(f"Запись {i}")
        
        # Должна остаться только последняя запись в short_term
        assert len(self.memory.short_term) == 1
        assert self.memory.short_term[0].text == "Запись 9"
        
        # И множественные суммаризации в long_term
        assert len(self.memory.long_term) >= 1

    def test_overflow_preserves_long_term_entries(self):
        """Тест: переполнение сохраняет существующие long_term записи."""
        # Добавляем записи в long_term вручную
        existing_entry = MemoryEntry(
            text="Существующая запись", 
            embedding=[0.4, 0.5, 0.6]
        )
        self.memory.long_term = [existing_entry]
        
        # Вызываем переполнение
        for i in range(5):
            self.memory.remember(f"Новая запись {i}")
        
        # Проверяем что старые записи сохранились
        assert len(self.memory.long_term) >= 1
        long_term_texts = [entry.text for entry in self.memory.long_term]
        assert "Существующая запись" in long_term_texts

    @patch('memory.unified_memory.UnifiedMemory.save')
    def test_overflow_calls_save(self, mock_save):
        """Тест: переполнение вызывает сохранение на диск."""
        # Заполняем до переполнения
        for i in range(5):
            self.memory.remember(f"Запись {i}")
        
        # Проверяем что save был вызван
        mock_save.assert_called()

    def test_zero_limit_edge_case(self):
        """Тест: граничный случай с лимитом = 0."""
        self.memory.short_term_limit = 0
        
        self.memory.remember("Единственная запись")
        
        # При лимите 0 всё должно идти в long_term
        assert len(self.memory.short_term) == 0
        assert len(self.memory.long_term) >= 1

    def test_negative_limit_edge_case(self):
        """Тест: граничный случай с отрицательным лимитом."""
        self.memory.short_term_limit = -1
        
        self.memory.remember("Запись")
        
        # При отрицательном лимите поведение должно быть безопасным
        assert len(self.memory.short_term) >= 0
        assert len(self.memory.long_term) >= 0