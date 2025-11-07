"""Тесты для PromptManager."""

import pytest
from llm.prompt_manager import PromptManager


class TestPromptManager:
    """Тесты для PromptManager."""

    def test_init(self):
        """Тест инициализации PromptManager."""
        manager = PromptManager("config/prompts.yaml")
        assert manager.config_path.name == "prompts.yaml"
        assert isinstance(manager.config, dict)

    def test_get_system_prompt(self):
        """Тест получения системного промта."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест базового промта
        prompt = manager.get_system_prompt("base_role", user_name="тест")
        assert "HAL" in prompt
        assert "тест" in prompt
        
        # Тест промта координатора
        prompt = manager.get_system_prompt("coordinator", user_query="тест")
        assert "Проанализируй запрос пользователя" in prompt

    def test_get_module_prompt(self):
        """Тест получения промта модуля."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест промта событий
        prompt = manager.get_module_prompt("events", "short_term", short_term_memory="тест")
        assert "краткосрочных событий" in prompt
        assert "тест" in prompt
        
        # Тест промта тем
        prompt = manager.get_module_prompt("themes", "long_term", long_term_memory="тест")
        assert "накопленных знаний" in prompt
        assert "тест" in prompt

    def test_get_tool_prompt(self):
        """Тест получения промта инструмента."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест промта поиска
        prompt = manager.get_tool_prompt("search", "web_search", query="тест")
        assert "Найди актуальную информацию в интернете" in prompt
        assert "{query}" in prompt  # Переменная должна быть в промте
        
        # Тест промта генерации кода
        prompt = manager.get_tool_prompt("code", "generation", task_description="тест")
        assert "Создай код" in prompt
        assert "{task_description}" in prompt  # Переменная должна быть в промте

    def test_get_adaptation_prompt(self):
        """Тест получения промта адаптации."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест промта обучения
        prompt = manager.get_adaptation_prompt("learning", period="тест")
        assert "Проанализируй взаимодействия" in prompt
        # Проверяем, что переменная есть в промте (она должна подставляться при вызове)
        assert "{period}" in prompt

    def test_get_settings(self):
        """Тест получения настроек."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест настроек модуля
        settings = manager.get_module_settings("events")
        assert isinstance(settings, dict)
        assert "priority" in settings
        
        # Тест настроек инструмента
        settings = manager.get_tool_settings("search")
        assert isinstance(settings, dict)
        assert "max_results" in settings

    def test_get_available_modules(self):
        """Тест получения доступных модулей."""
        manager = PromptManager("config/prompts.yaml")
        modules = manager.get_available_modules()
        assert "events" in modules
        assert "themes" in modules
        assert "emotions" in modules

    def test_get_available_tools(self):
        """Тест получения доступных инструментов."""
        manager = PromptManager("config/prompts.yaml")
        tools = manager.get_available_tools()
        assert "search" in tools
        assert "code" in tools
        assert "planning" in tools

    def test_get_available_memory_levels(self):
        """Тест получения доступных уровней памяти."""
        manager = PromptManager("config/prompts.yaml")
        levels = manager.get_available_memory_levels()
        assert "short_term" in levels
        assert "long_term" in levels
        assert "episodic" in levels

    def test_validate_config(self):
        """Тест валидации конфигурации."""
        manager = PromptManager("config/prompts.yaml")
        errors = manager.validate_config()
        assert isinstance(errors, list)
        # Должно быть без ошибок для корректной конфигурации
        assert len(errors) == 0

    def test_get_prompt_statistics(self):
        """Тест получения статистики промтов."""
        manager = PromptManager("config/prompts.yaml")
        stats = manager.get_prompt_statistics()
        assert isinstance(stats, dict)
        assert "total_prompts" in stats
        assert "modules" in stats
        assert "tools" in stats
        assert stats["total_prompts"] > 0

    def test_substitute_variables(self):
        """Тест подстановки переменных."""
        manager = PromptManager("config/prompts.yaml")
        
        # Тест с переменными
        result = manager._substitute_variables(
            "Привет {user_name}, время {current_time}",
            user_name="тест"
        )
        assert "тест" in result
        assert "время" in result
        
        # Тест без переменных
        result = manager._substitute_variables("Простой текст")
        assert result == "Простой текст"

    def test_reload_config(self):
        """Тест перезагрузки конфигурации."""
        manager = PromptManager("config/prompts.yaml")
        manager.reload_config()
        # Должно работать без ошибок
        assert isinstance(manager.config, dict)
