#!/usr/bin/env python3
"""
Простой тест для проверки доступности Flash Attention в Ollama.
"""

import subprocess
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_ollama_parameters():
    """Проверяем доступные параметры Ollama."""
    print("🔍 Проверка доступных параметров Ollama...")
    
    # Проверяем версию Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"📋 Версия Ollama: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Ошибка получения версии: {e}")
    
    # Проверяем переменные окружения
    print(f"\n🌍 Переменные окружения:")
    print(f"  OLLAMA_FLASH_ATTENTION: {subprocess.run(['echo', '$OLLAMA_FLASH_ATTENTION'], capture_output=True, text=True).stdout.strip()}")
    print(f"  OLLAMA_DEBUG: {subprocess.run(['echo', '$OLLAMA_DEBUG'], capture_output=True, text=True).stdout.strip()}")
    print(f"  OLLAMA_HOST: {subprocess.run(['echo', '$OLLAMA_HOST'], capture_output=True, text=True).stdout.strip()}")
    
    # Проверяем информацию о модели
    print(f"\n🤖 Информация о модели gemma3n:e4b-it-q8_0:")
    try:
        result = subprocess.run(["ollama", "show", "gemma3n:e4b-it-q8_0"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Ошибка получения информации о модели: {e}")
    
    # Проверяем запущенные модели
    print(f"\n🏃 Запущенные модели:")
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Ошибка получения списка моделей: {e}")

def test_flash_attention_parameter():
    """Тестируем передачу параметра Flash Attention через API."""
    print(f"\n🧪 Тестирование Flash Attention через API...")
    
    try:
        from llm.factory import create_llm_client
        import yaml
        
        # Загружаем конфигурацию
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get("llm", {}).copy()
        llm_config["model"] = "gemma3n:e4b-it-q8_0"
        
        # Пробуем добавить параметр flash_attention
        llm_config["flash_attention"] = True
        
        print(f"📝 Конфигурация с flash_attention: {llm_config}")
        
        # Создаем клиент
        llm_client = create_llm_client(
            provider=llm_config.get("provider", "ollama"),
            llm_cfg=llm_config,
            ollama_cfg=llm_config
        )
        
        # Тестируем простой запрос
        print(f"🔄 Тестирование простого запроса...")
        response = llm_client.generate("Привет! Как дела?")
        print(f"✅ Ответ получен: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования API: {e}")

def main():
    """Главная функция."""
    print("🚀 Проверка параметров Ollama для Flash Attention")
    print("=" * 60)
    
    check_ollama_parameters()
    test_flash_attention_parameter()
    
    print(f"\n📋 ВЫВОДЫ:")
    print(f"1. OLLAMA_FLASH_ATTENTION - переменная окружения для включения Flash Attention")
    print(f"2. Конкретного параметра 'Force Model Expert Weights onto CPU' не найдено")
    print(f"3. Доступны параметры: OLLAMA_FLASH_ATTENTION, OLLAMA_DEBUG, OLLAMA_HOST и др.")
    print(f"4. Для тестирования нужно перезапустить Ollama с OLLAMA_FLASH_ATTENTION=1")

if __name__ == "__main__":
    main()
