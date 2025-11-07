#!/usr/bin/env python3
"""Скрипт для авторизации Telethon."""

from dotenv import load_dotenv
from telethon import TelegramClient
import os

load_dotenv()

api_id = int(os.getenv('TELETHON_API_ID'))
api_hash = os.getenv('TELETHON_API_HASH')

print(f"API ID: {api_id}")
print(f"API Hash: {api_hash[:10]}...")

client = TelegramClient('db/session/user', api_id, api_hash)

try:
    client.start()
    print("✅ Авторизация успешна!")
    client.disconnect()
except Exception as e:
    print(f"❌ Ошибка авторизации: {e}")


