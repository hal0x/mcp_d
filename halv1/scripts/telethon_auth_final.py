#!/usr/bin/env python3
"""
Финальный скрипт для авторизации Telethon с поддержкой переменных окружения
"""
import os
import sys
import asyncio
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

async def main():
    print("Загрузка переменных окружения...")
    if not load_env_file():
        sys.exit(1)
    
    # Получаем данные
    api_id = os.getenv('TELETHON_API_ID')
    api_hash = os.getenv('TELETHON_API_HASH')
    phone = os.getenv('TELETHON_PHONE')
    password = os.getenv('TELETHON_PASSWORD')
    
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
        if phone:
            print(f"Попытка авторизации с номером: {phone}")
            await client.start(phone=phone)
        else:
            print("Проверка существующей сессии...")
            await client.start()
        
        # Проверяем, что мы авторизованы
        me = await client.get_me()
        print(f"✅ Авторизация успешна!")
        print(f"Авторизован как: {me.first_name} {me.last_name or ''} (@{me.username or 'без username'})")
        
        # Получаем список диалогов
        dialogs = await client.get_dialogs(limit=5)
        print(f"Найдено диалогов: {len(dialogs)}")
        
        for dialog in dialogs[:3]:  # Показываем первые 3
            print(f"  - {dialog.name} (ID: {dialog.id})")
        
    except SessionPasswordNeededError:
        if password:
            print("Используется пароль 2FA из переменной окружения...")
            await client.sign_in(password=password)
            print("✅ Авторизация с 2FA успешна!")
        else:
            print("Требуется двухфакторная аутентификация.")
            password_input = input("Введите пароль 2FA: ")
            await client.sign_in(password=password_input)
            print("✅ Авторизация с 2FA успешна!")
        
    except Exception as e:
        print(f"❌ Ошибка авторизации: {e}")
        if not phone:
            print("\nДля авторизации установите переменную TELETHON_PHONE:")
            print("export TELETHON_PHONE='+ваш_номер'")
            print("python scripts/telethon_auth_final.py")
        sys.exit(1)
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
