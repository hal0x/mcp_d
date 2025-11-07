#!/usr/bin/env python3
"""
Быстрый тест улучшенных промптов для executor и critic.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.factory import create_llm_client
from llm.prompts import make_executor_prompt, make_critic_prompt
import yaml

async def test_improved_prompts():
    """Тестирование улучшенных промптов."""
    print("🧪 Тестирование улучшенных промптов...")
    
    # Загружаем конфигурацию
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_config = config.get("llm", {})
    
    # Создаем LLM клиент
    llm_client = create_llm_client(
        provider=llm_config.get("provider", "ollama"),
        llm_cfg=llm_config,
        ollama_cfg=llm_config
    )
    
    # Тестируем executor
    print("\n📝 Тестирование EXECUTOR...")
    executor_prompt = make_executor_prompt()
    print(f"Промпт: {executor_prompt[:100]}...")
    
    executor_response = llm_client.generate(executor_prompt)
    print(f"Ответ: {executor_response}")
    
    # Проверяем качество ответа
    if "Выполнено:" in executor_response or "Ошибка:" in executor_response or "Требуется подтверждение:" in executor_response:
        print("✅ EXECUTOR: Качество улучшено!")
    else:
        print("❌ EXECUTOR: Качество не улучшилось")
    
    # Тестируем critic
    print("\n📝 Тестирование CRITIC...")
    critic_prompt = make_critic_prompt()
    print(f"Промпт: {critic_prompt[:100]}...")
    
    critic_response = llm_client.generate(critic_prompt)
    print(f"Ответ: {critic_response}")
    
    # Проверяем качество ответа
    if "OK" in critic_response or "ПРОБЛЕМЫ:" in critic_response:
        print("✅ CRITIC: Качество улучшено!")
    else:
        print("❌ CRITIC: Качество не улучшилось")

if __name__ == "__main__":
    asyncio.run(test_improved_prompts())
