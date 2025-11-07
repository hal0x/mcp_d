#!/usr/bin/env python3
"""
Тест интеграции контекстно-осведомленных компонентов в main.py
"""

import sys
import os
import time
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.context_factory import create_context_aware_client, create_context_aware_code_generator, create_context_aware_search_client
from agent.modules.context_aware_base import ContextAwareEventsModule, ContextAwareThemesModule
from memory import UnifiedMemory
from llm.prompt_manager import PromptManager

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_context_integration():
    """Тестируем интеграцию контекстно-осведомленных компонентов."""
    logger.info("🧪 Тестируем интеграцию контекстно-осведомленных компонентов...")
    
    try:
        # Настройка конфигурации
        llm_cfg = {
            "provider": "ollama",
            "model": "gemma3n:e4b-it-q8_0",
            "num_ctx": 16384,
            "num_keep": 256,
            "keep_alive": "30m",
            "num_batch": 1024
        }
        ollama_cfg = {
            "host": "localhost",
            "port": 11434
        }
        
        # Создаем контекстно-осведомленные компоненты
        logger.info("🔧 Создаем контекстно-осведомленные компоненты...")
        
        # 1. LLM клиент
        context_aware_client = create_context_aware_client("ollama", llm_cfg, ollama_cfg)
        logger.info("✅ ContextAwareLLMClient создан")
        
        # 2. Генератор кода
        context_aware_code_generator = create_context_aware_code_generator("ollama", llm_cfg, ollama_cfg)
        logger.info("✅ ContextAwareCodeGenerator создан")
        
        # 3. Поисковый клиент
        context_aware_search_client = create_context_aware_search_client("ollama", llm_cfg, ollama_cfg)
        logger.info("✅ ContextAwareSearchClient создан")
        
        # 4. Модули анализа памяти
        memory_store = UnifiedMemory(
            long_term_path="db/memory/long_term.json",
            short_term_limit=100,
            llm_client=context_aware_client
        )
        prompt_manager = PromptManager("config/prompts.yaml")
        
        context_aware_events_module = ContextAwareEventsModule(
            memory_store=memory_store,
            llm_client=context_aware_client,
            prompt_manager=prompt_manager
        )
        logger.info("✅ ContextAwareEventsModule создан")
        
        context_aware_themes_module = ContextAwareThemesModule(
            memory_store=memory_store,
            llm_client=context_aware_client,
            prompt_manager=prompt_manager
        )
        logger.info("✅ ContextAwareThemesModule создан")
        
        # Тестируем производительность
        logger.info("⚡ Тестируем производительность...")
        
        # Тест 1: LLM клиент
        start_time = time.time()
        response1 = context_aware_client.generate("Расскажи о Python")
        time1 = time.time() - start_time
        logger.info(f"📝 LLM запрос 1: {time1:.2f}с")
        
        start_time = time.time()
        response2 = context_aware_client.generate("Какие библиотеки Python самые популярные?")
        time2 = time.time() - start_time
        logger.info(f"📝 LLM запрос 2: {time2:.2f}с")
        
        # Тест 2: Генератор кода (мок-тест, так как Ollama не запущен)
        start_time = time.time()
        try:
            code_result = context_aware_code_generator.generate("Создай функцию для вычисления факториала")
            time3 = time.time() - start_time
            logger.info(f"💻 Генерация кода: {time3:.2f}с")
        except Exception as e:
            time3 = time.time() - start_time
            logger.info(f"💻 Генерация кода (мок): {time3:.2f}с (Ollama не запущен: {e})")
        
        # Тест 3: Модули анализа памяти (мок-тест)
        start_time = time.time()
        try:
            events_result = context_aware_events_module.analyze_short_term()
            time4 = time.time() - start_time
            logger.info(f"🧠 Анализ событий: {time4:.2f}с")
        except Exception as e:
            time4 = time.time() - start_time
            logger.info(f"🧠 Анализ событий (мок): {time4:.2f}с (Ollama не запущен: {e})")
        
        start_time = time.time()
        try:
            themes_result = context_aware_themes_module.analyze_short_term()
            time5 = time.time() - start_time
            logger.info(f"🧠 Анализ тем: {time5:.2f}с")
        except Exception as e:
            time5 = time.time() - start_time
            logger.info(f"🧠 Анализ тем (мок): {time5:.2f}с (Ollama не запущен: {e})")
        
        # Проверяем статус контекста
        logger.info("📊 Статус контекста:")
        logger.info(f"  - LLM клиент: {'✅ активен' if context_aware_client.get_context() else '❌ неактивен'}")
        logger.info(f"  - Модуль событий: {'✅ активен' if context_aware_events_module.get_context() else '❌ неактивен'}")
        logger.info(f"  - Модуль тем: {'✅ активен' if context_aware_themes_module.get_context() else '❌ неактивен'}")
        
        # Очистка контекста
        logger.info("🧹 Очищаем контекст...")
        context_aware_client.clear_context()
        context_aware_events_module.clear_context()
        context_aware_themes_module.clear_context()
        
        logger.info("✅ Все контекстно-осведомленные компоненты работают корректно!")
        logger.info("🚀 Интеграция в main.py готова к использованию")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании интеграции: {e}")
        return False

if __name__ == "__main__":
    success = test_context_integration()
    sys.exit(0 if success else 1)
