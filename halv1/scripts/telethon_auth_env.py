#!/usr/bin/env python3
"""
Скрипт для авторизации Telethon с использованием переменных окружения
Использование: TELETHON_PHONE="+1234567890" python scripts/telethon_auth_env.py
"""
import os
import sys
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

def main():
    # Получаем данные из переменных окружения
    api_id = os.getenv('TELETHON_API_ID')
    api_hash = os.getenv('TELETHON_API_HASH')
    phone = os.getenv('TELETHON_PHONE')
    password = os.getenv('TELETHON_PASSWORD')  # Для 2FA
    
    if not api_id or not api_hash:
        print("Ошибка: TELETHON_API_ID и TELETHON_API_HASH должны быть установлены в переменных окружения")
        sys.exit(1)
    
    if not phone:
        print("Ошибка: TELETHON_PHONE должен быть установлен в переменных окружения")
        print("Пример: TELETHON_PHONE='+1234567890' python scripts/telethon_auth_env.py")
        sys.exit(1)
    
    # Путь к файлу сессии
    session_path = 'db/session/user.session'
    
    # Создаем клиент
    client = TelegramClient(session_path, int(api_id), api_hash)
    
    try:
        print(f"Запуск авторизации Telethon для номера: {phone}")
        
        # Запускаем клиент
        client.start(phone=phone)
        
        print("✅ Авторизация успешна!")
        print(f"Сессия сохранена в: {session_path}")
        
        # Проверяем, что мы авторизованы
        me = client.get_me()
        print(f"Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        
    except SessionPasswordNeededError:
        if password:
            print("Используется пароль 2FA из переменной окружения...")
            client.sign_in(password=password)
            print("✅ Авторизация с 2FA успешна!")
        else:
            print("Требуется двухфакторная аутентификация.")
            password_input = input("Введите пароль 2FA: ")
            client.sign_in(password=password_input)
            print("✅ Авторизация с 2FA успешна!")
        
    except Exception as e:
        print(f"❌ Ошибка авторизации: {e}")
        sys.exit(1)
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
