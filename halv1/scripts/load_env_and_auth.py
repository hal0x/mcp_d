#!/usr/bin/env python3
"""
Скрипт для загрузки переменных окружения и авторизации Telethon
"""
import os
import sys
from pathlib import Path
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

def load_env_file():
    """Загружает переменные из .env файла"""
    env_path = Path('.env')
    if not env_path.exists():
        print("Файл .env не найден")
        return False
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Убираем кавычки если есть
                value = value.strip('"').strip("'")
                os.environ[key] = value
    
    return True

def main():
    print("Загрузка переменных окружения...")
    if not load_env_file():
        sys.exit(1)
    
    # Получаем API ID и API Hash
    api_id = os.getenv('TELETHON_API_ID')
    api_hash = os.getenv('TELETHON_API_HASH')
    
    if not api_id or not api_hash:
        print("Ошибка: TELETHON_API_ID и TELETHON_API_HASH не найдены в .env файле")
        sys.exit(1)
    
    print(f"API ID: {api_id}")
    print(f"API Hash: {api_hash[:8]}...")
    
    # Путь к файлу сессии
    session_path = 'db/session/user.session'
    
    # Создаем клиент
    client = TelegramClient(session_path, int(api_id), api_hash)
    
    try:
        print("Проверка существующей сессии...")
        client.start()
        
        # Проверяем, что мы авторизованы
        me = client.get_me()
        print(f"✅ Сессия уже авторизована!")
        print(f"Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        
        # Получаем список диалогов
        dialogs = client.get_dialogs(limit=5)
        print(f"Найдено диалогов: {len(dialogs)}")
        
        for dialog in dialogs[:3]:  # Показываем первые 3
            print(f"  - {dialog.name} (ID: {dialog.id})")
        
    except Exception as e:
        print(f"❌ Сессия не авторизована: {e}")
        print("\nДля авторизации выполните:")
        print("python scripts/telethon_auth.py")
        print("или")
        print("TELETHON_PHONE='+ваш_номер' python scripts/telethon_auth_env.py")
        sys.exit(1)
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
