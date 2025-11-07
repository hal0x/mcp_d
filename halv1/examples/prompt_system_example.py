"""Пример использования системы промтов HAL AI-агента."""

import asyncio
import logging
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.prompt_manager import PromptManager
from memory import UnifiedMemory

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Демонстрация работы системы промтов."""
    
    # Инициализация компонентов
    logger.info("Инициализация системы промтов...")
    
    # Создаем менеджер промтов
    prompt_manager = PromptManager("config/prompts.yaml")
    
    # Создаем LLM клиент (заглушка для демонстрации)
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            return f"Ответ на промт: {prompt[:100]}..."
    
    llm_client = MockLLMClient()
    
    # Создаем хранилище памяти
    memory_store = UnifiedMemory()
    
    # Добавляем тестовые данные в память
    memory_store.remember("Встреча с клиентом завтра в 14:00")
    memory_store.remember("Нужно подготовить презентацию по проекту X")
    memory_store.remember("Проблема с сервером - нужно срочно исправить")
    
    logger.info("✅ Система инициализирована")
    
    # Демонстрация системных промтов
    logger.info("\n=== СИСТЕМНЫЕ ПРОМТЫ ===")
    
    base_role = prompt_manager.get_system_prompt(
        "base_role",
        user_name="Алексей",
        timezone="Asia/Bangkok"
    )
    logger.info(f"Базовый промт роли:\n{base_role[:200]}...")
    
    coordinator = prompt_manager.get_system_prompt(
        "coordinator",
        user_query="Что у меня запланировано на завтра?",
        user_context="Рабочий контекст",
        available_tools="search, code, planning"
    )
    logger.info(f"Промт координатора:\n{coordinator[:200]}...")
    
    # Демонстрация модульных промтов
    logger.info("\n=== МОДУЛЬНЫЕ ПРОМТЫ ===")
    
    # События - краткосрочная память
    events_short = prompt_manager.get_module_prompt(
        "events",
        "short_term",
        short_term_memory="Встреча с клиентом завтра в 14:00\nПроблема с сервером",
        query_context="планы на завтра"
    )
    logger.info(f"Промт событий (краткосрочная память):\n{events_short[:200]}...")
    
    # Темы - долгосрочная память
    themes_long = prompt_manager.get_module_prompt(
        "themes",
        "long_term",
        long_term_memory="Проект X - разработка веб-приложения\nИзучение Python и машинного обучения",
        query_context="текущие проекты"
    )
    logger.info(f"Промт тем (долгосрочная память):\n{themes_long[:200]}...")
    
    # Эмоции - эпизодическая память
    emotions_episodic = prompt_manager.get_module_prompt(
        "emotions",
        "episodic",
        episodic_memory="Стресс из-за дедлайна проекта\nРадость от успешного решения проблемы",
        query_context="эмоциональное состояние"
    )
    logger.info(f"Промт эмоций (эпизодическая память):\n{emotions_episodic[:200]}...")
    
    # Демонстрация промтов инструментов
    logger.info("\n=== ПРОМТЫ ИНСТРУМЕНТОВ ===")
    
    # Поиск в интернете
    search_prompt = prompt_manager.get_tool_prompt(
        "search",
        "web_search",
        query="новости по Python 3.12",
        user_context="разработчик Python",
        priority_themes="программирование, технологии"
    )
    logger.info(f"Промт поиска:\n{search_prompt[:200]}...")
    
    # Генерация кода
    code_prompt = prompt_manager.get_tool_prompt(
        "code",
        "generation",
        task_description="создать функцию для сортировки списка чисел",
        user_context="разработчик Python",
        available_data="список чисел",
        code_requirements="использовать встроенную функцию sorted"
    )
    logger.info(f"Промт генерации кода:\n{code_prompt[:200]}...")
    
    # Планирование задач
    planning_prompt = prompt_manager.get_tool_prompt(
        "planning",
        "task_plan",
        task="подготовить презентацию для клиента",
        available_tools="search, code, file_io",
        user_context="рабочий контекст",
        priorities="срочность, важность"
    )
    logger.info(f"Промт планирования:\n{planning_prompt[:200]}...")
    
    # Демонстрация промтов адаптации
    logger.info("\n=== ПРОМТЫ АДАПТАЦИИ ===")
    
    learning_prompt = prompt_manager.get_adaptation_prompt(
        "learning",
        period="последняя неделя",
        interaction_data="Запросы пользователя и ответы агента",
        user_reactions="положительные отзывы",
        performance_metrics="время ответа, точность"
    )
    logger.info(f"Промт обучения:\n{learning_prompt[:200]}...")
    
    # Демонстрация настроек
    logger.info("\n=== НАСТРОЙКИ ===")
    
    # Настройки модулей
    events_settings = prompt_manager.get_module_settings("events")
    logger.info(f"Настройки модуля событий: {events_settings}")
    
    # Настройки инструментов
    search_settings = prompt_manager.get_tool_settings("search")
    logger.info(f"Настройки поиска: {search_settings}")
    
    # Системные настройки
    system_settings = prompt_manager.get_system_settings()
    logger.info(f"Системные настройки: {system_settings}")
    
    # Статистика промтов
    stats = prompt_manager.get_prompt_statistics()
    logger.info(f"Статистика промтов: {stats}")
    
    # Валидация конфигурации
    errors = prompt_manager.validate_config()
    if errors:
        logger.warning(f"Ошибки валидации: {errors}")
    else:
        logger.info("✅ Конфигурация промтов валидна")
    
    logger.info("\n🎉 Демонстрация системы промтов завершена!")


if __name__ == "__main__":
    asyncio.run(main())
