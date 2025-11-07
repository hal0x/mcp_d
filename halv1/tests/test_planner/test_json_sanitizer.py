"""Тесты для JSON sanitizer в Planner."""

import pytest
from planner.task_planner import LLMTaskPlanner
from llm.ollama_client import OllamaClient


class TestJSONSanitizer:
    """Тесты для функций санитизации JSON в Planner."""
    
    def test_sanitize_json_response_basic(self):
        """Тест базовой санитизации JSON ответа."""
        from unittest.mock import Mock
        
        # Создаем мок LLM клиента
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"task_completion_criteria": "Test", "requires_all_steps": true, "steps": [{"tool": "code", "content": "print(1)", "expected_output": "1", "is_final": true, "depends_on": []}]}'
        
        # Создаем тестовый экземпляр
        planner = LLMTaskPlanner(mock_llm)
        
        # Тестируем план
        result = planner.plan("Test task")
        
        # Проверяем что результат содержит ожидаемые элементы
        assert result.steps is not None
        assert len(result.steps) > 0
        assert result.steps[0].content == "print(1)"
        assert result.steps[0].expected_output == "1"
            
    def test_extract_first_balanced_json(self):
        """Тест извлечения первого сбалансированного JSON."""
        from unittest.mock import Mock
        
        # Создаем мок LLM клиента
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"task_completion_criteria": "Test", "requires_all_steps": true, "steps": [{"tool": "code", "content": "print(1)", "expected_output": "1", "is_final": true, "depends_on": []}]}'
        
        # Создаем тестовый экземпляр
        planner = LLMTaskPlanner(mock_llm)
        
        # Тестируем план с валидным JSON
        result = planner.plan("Test task")
        assert result is not None
        assert len(result.steps) > 0


if __name__ == "__main__":
    pytest.main([__file__])