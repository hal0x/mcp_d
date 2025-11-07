#!/usr/bin/env python3
"""Интерактивная авторизация Telethon."""

import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient
import os

load_dotenv()

api_id = int(os.getenv('TELETHON_API_ID'))
api_hash = os.getenv('TELETHON_API_HASH')

async def main():
    client = TelegramClient('db/session/user', api_id, api_hash)
    
    try:
        await client.start()
        print("✅ Авторизация успешна!")
        print("Теперь можно запускать индексацию с INDEX_NOW=1")
    except Exception as e:
        print(f"❌ Ошибка авторизации: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())


