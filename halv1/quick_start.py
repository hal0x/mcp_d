#!/usr/bin/env python3
"""
Быстрый старт для создания поискового индекса канала "Вселенная Плюс"
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Проверка установленных зависимостей"""
    required_packages = [
        'numpy',
        'sentence_transformers', 
        'faiss',
        'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Установите их командой:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ Все необходимые пакеты установлены!")
    return True

def check_json_files():
    """Проверка наличия JSON файлов"""
    current_dir = Path(__file__).parent
    json_files = list(current_dir.glob("*.json"))
    
    if not json_files:
        print("❌ JSON файлы не найдены в текущем каталоге!")
        print("Убедитесь, что вы находитесь в каталоге с JSON файлами канала.")
        return False
    
    print(f"✅ Найдено {len(json_files)} JSON файлов")
    return True

def create_index():
    """Создание поискового индекса"""
    print("\n🚀 Создаем поисковый индекс...")
    
    try:
        from create_search_index import TelegramChannelIndexer
        
        indexer = TelegramChannelIndexer('.')
        indexer.build_index()
        
        print("✅ Индекс создан успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания индекса: {e}")
        return False

def start_web_interface():
    """Запуск веб-интерфейса"""
    print("\n🌐 Запускаем веб-интерфейс...")
    print("Откройте браузер и перейдите по адресу: http://localhost:5000")
    
    try:
        from search_interface import app, init_searcher
        
        if init_searcher():
            app.run(debug=False, host='0.0.0.0', port=5000)
        else:
            print("❌ Ошибка инициализации поисковой системы!")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка запуска веб-интерфейса: {e}")
        return False

def main():
    """Основная функция"""
    print("🔍 Быстрый старт поискового индекса канала 'Вселенная Плюс'")
    print("=" * 60)
    
    # Проверяем зависимости
    if not check_requirements():
        return
    
    # Проверяем JSON файлы
    if not check_json_files():
        return
    
    # Создаем индекс
    if not create_index():
        return
    
    # Запускаем веб-интерфейс
    start_web_interface()

if __name__ == "__main__":
    main()
