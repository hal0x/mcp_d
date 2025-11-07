#!/usr/bin/env python3
"""
Скрипт для однократной авторизации Telethon
"""
import os
import sys
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

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
        print("Запуск авторизации Telethon...")
        print("Введите номер телефона в международном формате (например, +1234567890):")
        phone = input("Номер телефона: ")
        
        # Запускаем клиент
        client.start(phone=phone)
        
        print("✅ Авторизация успешна!")
        print(f"Сессия сохранена в: {session_path}")
        
        # Проверяем, что мы авторизованы
        me = client.get_me()
        print(f"Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        
    except SessionPasswordNeededError:
        print("Требуется двухфакторная аутентификация.")
        password = input("Введите пароль 2FA: ")
        client.sign_in(password=password)
        print("✅ Авторизация с 2FA успешна!")
        
    except Exception as e:
        print(f"❌ Ошибка авторизации: {e}")
        sys.exit(1)
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
