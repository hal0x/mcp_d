#!/usr/bin/env python3
"""
Скрипт для проверки доступности LM Studio Server и моделей
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_mcp.core.lmstudio_client import LMStudioEmbeddingClient
from memory_mcp.config import get_settings


async def check_lmstudio():
    """Проверка доступности LM Studio Server"""
    print("🔍 Проверка LM Studio Server...")
    print()

    settings = get_settings()
    client = LMStudioEmbeddingClient(
        model_name=settings.lmstudio_model,
        base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
    )

    async with client:
        result = await client.test_connection()

        if result.get("lmstudio_available"):
            print("✅ LM Studio Server доступен")
            print(f"   URL: {result['base_url']}")
            print(f"   Доступные модели: {', '.join(result.get('available_models', []))}")
            print()

            if result.get("model_available"):
                print(f"✅ Модель '{result['model_name']}' найдена")
                return 0
            else:
                print(f"❌ Модель '{result['model_name']}' не найдена")
                print(f"   Доступные модели: {', '.join(result.get('available_models', []))}")
                return 1
        else:
            print("❌ LM Studio Server недоступен")
            if "error" in result:
                print(f"   Ошибка: {result['error']}")
            print(f"   Убедитесь, что LM Studio запущен и сервер работает на {settings.lmstudio_host}:{settings.lmstudio_port}")
            return 1


if __name__ == "__main__":
    exit_code = asyncio.run(check_lmstudio())
    sys.exit(exit_code)

