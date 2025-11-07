#!/usr/bin/env python3
"""
Скрипт для проверки авторизации Telethon
"""
import os
import sys
from telethon import TelegramClient

def main():
    # Получаем API ID и API Hash из переменных окружения
    api_id = os.getenv('TELETHON_API_ID')
    api_hash = os.getenv('TELETHON_API_HASH')
    
    if not api_id or not api_hash:
        print("Ошибка: TELETHON_API_ID и TELETHON_API_HASH должны быть установлены в переменных окружения")
        sys.exit(1)
    
    # Путь к файлу сессии
    session_path = 'db/session/user.session'
    
    # Создаем клиент
    client = TelegramClient(session_path, int(api_id), api_hash)
    
    try:
        print("Проверка авторизации Telethon...")
        
        # Запускаем клиент
        client.start()
        
        # Проверяем, что мы авторизованы
        me = client.get_me()
        print(f"✅ Авторизация работает!")
        print(f"Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        
        # Получаем список диалогов
        dialogs = client.get_dialogs(limit=5)
        print(f"Найдено диалогов: {len(dialogs)}")
        
        for dialog in dialogs[:3]:  # Показываем первые 3
            print(f"  - {dialog.name} (ID: {dialog.id})")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Возможно, требуется повторная авторизация")
        sys.exit(1)
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
