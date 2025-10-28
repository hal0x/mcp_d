#!/usr/bin/env python3
"""
Примеры отправки запросов к Ollama с полным контекстом 32,768 токенов
"""

import asyncio
from typing import Any, Dict, Optional

import aiohttp
import requests


class OllamaFullContextClient:
    """Клиент для работы с Ollama с максимальным контекстом"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "gemma3n:e4b-it-q8_0"
        self.max_context = 32768  # Максимальный контекст модели

    async def generate_with_full_context_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        top_p: float = 0.93,
        presence_penalty: float = 0.05,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Асинхронная генерация с полным контекстом

        Args:
            prompt: Основной промпт
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов для генерации
            top_p: Top-p параметр
            presence_penalty: Presence penalty
            system_prompt: Системный промпт (опционально)

        Returns:
            Сгенерированный текст
        """

        # Формируем полный промпт
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Проверяем длину промпта
        estimated_tokens = len(full_prompt) // 4
        if estimated_tokens > self.max_context - max_tokens:
            print(f"⚠️  Промпт слишком длинный: {estimated_tokens} токенов")
            print(f"   Максимум для промпта: {self.max_context - max_tokens} токенов")
            # Обрезаем промпт
            max_prompt_chars = (self.max_context - max_tokens) * 4
            full_prompt = full_prompt[:max_prompt_chars] + "..."

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "num_ctx": self.max_context,  # Используем полный контекст
                "num_thread": 8,  # Количество потоков
                "repeat_penalty": 1.1,  # Штраф за повторения
                "stop": ["</s>", "Human:", "Assistant:"],  # Стоп-слова
            },
        }

        try:
            timeout = aiohttp.ClientTimeout(total=600)  # 10 минут
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "").strip()
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return f"Ошибка: {str(e)}"

    def generate_with_full_context_sync(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        top_p: float = 0.93,
        presence_penalty: float = 0.05,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Синхронная генерация с полным контекстом
        """

        # Формируем полный промпт
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Проверяем длину промпта
        estimated_tokens = len(full_prompt) // 4
        if estimated_tokens > self.max_context - max_tokens:
            print(f"⚠️  Промпт слишком длинный: {estimated_tokens} токенов")
            max_prompt_chars = (self.max_context - max_tokens) * 4
            full_prompt = full_prompt[:max_prompt_chars] + "..."

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "num_ctx": self.max_context,
                "num_thread": 8,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "Human:", "Assistant:"],
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=600
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return f"Ошибка: {str(e)}"

    async def chat_with_context_async(
        self,
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Чат с контекстом (поддержка истории сообщений)

        Args:
            messages: Список сообщений [{"role": "user", "content": "..."}, ...]
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            system_prompt: Системный промпт

        Returns:
            Ответ модели
        """

        # Формируем промпт из истории сообщений
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.title()}: {content}")

        full_prompt = "\n\n".join(prompt_parts)

        return await self.generate_with_full_context_async(
            prompt=full_prompt, temperature=temperature, max_tokens=max_tokens
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                for model in models:
                    if model["name"] == self.model_name:
                        return model
            return {}
        except Exception as e:
            print(f"❌ Ошибка получения информации о модели: {e}")
            return {}


# Примеры использования
async def example_usage():
    """Примеры использования клиента"""

    client = OllamaFullContextClient()

    print("🤖 Тестирование Ollama с полным контекстом 32,768 токенов\n")

    # 1. Простая генерация
    print("1️⃣ Простая генерация:")
    simple_prompt = (
        "Расскажи подробно о квантовых вычислениях и их применении в криптографии."
    )
    result = await client.generate_with_full_context_async(
        prompt=simple_prompt, max_tokens=2000, temperature=0.7
    )
    print(f"Результат: {result[:200]}...\n")

    # 2. Генерация с системным промптом
    print("2️⃣ Генерация с системным промптом:")
    system_prompt = (
        "Ты эксперт по блокчейну и криптовалютам. Отвечай подробно и технически."
    )
    user_prompt = "Объясни разницу между Proof of Work и Proof of Stake консенсусом."
    result = await client.generate_with_full_context_async(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=3000,
        temperature=0.5,
    )
    print(f"Результат: {result[:200]}...\n")

    # 3. Чат с контекстом
    print("3️⃣ Чат с контекстом:")
    messages = [
        {"role": "user", "content": "Привет! Меня зовут Алексей."},
        {
            "role": "assistant",
            "content": "Привет, Алексей! Рад познакомиться. Как дела?",
        },
        {"role": "user", "content": "Отлично! Расскажи мне о машинном обучении."},
    ]
    result = await client.chat_with_context_async(
        messages=messages, max_tokens=1500, temperature=0.6
    )
    print(f"Результат: {result[:200]}...\n")

    # 4. Длинный промпт (тест контекста)
    print("4️⃣ Тест длинного промпта:")
    long_text = (
        "Анализируй следующий текст: "
        + "Это очень длинный текст для тестирования контекста. " * 1000
    )
    result = await client.generate_with_full_context_async(
        prompt=long_text, max_tokens=1000, temperature=0.3
    )
    print(f"Результат: {result[:200]}...\n")

    # 5. Информация о модели
    print("5️⃣ Информация о модели:")
    model_info = client.get_model_info()
    if model_info:
        print(f"Модель: {model_info.get('name', 'Unknown')}")
        print(f"Размер: {model_info.get('size', 'Unknown')} байт")
        print(f"Модифицирован: {model_info.get('modified_at', 'Unknown')}")
    else:
        print("Информация о модели недоступна")


def curl_examples():
    """Примеры запросов через curl"""

    print("🌐 Примеры запросов через curl:\n")

    # 1. Простой запрос
    print("1️⃣ Простой запрос:")
    curl_simple = """
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma3n:e4b-it-q8_0",
    "prompt": "Расскажи о блокчейне",
    "stream": false,
    "options": {
      "num_ctx": 32768,
      "num_predict": 1000,
      "temperature": 0.7
    }
  }'
"""
    print(curl_simple)

    # 2. Запрос с системным промптом
    print("2️⃣ Запрос с системным промптом:")
    curl_system = """
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma3n:e4b-it-q8_0",
    "prompt": "System: Ты эксперт по криптовалютам.\\n\\nUser: Объясни DeFi",
    "stream": false,
    "options": {
      "num_ctx": 32768,
      "num_predict": 2000,
      "temperature": 0.5,
      "top_p": 0.9,
      "repeat_penalty": 1.1
    }
  }'
"""
    print(curl_system)

    # 3. Стриминг запрос
    print("3️⃣ Стриминг запрос:")
    curl_stream = """
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma3n:e4b-it-q8_0",
    "prompt": "Напиши эссе о будущем ИИ",
    "stream": true,
    "options": {
      "num_ctx": 32768,
      "num_predict": 3000,
      "temperature": 0.8
    }
  }'
"""
    print(curl_stream)


if __name__ == "__main__":
    print("🚀 Ollama Full Context Client")
    print("=" * 50)

    # Показываем примеры curl
    curl_examples()

    print("\n" + "=" * 50)
    print("🐍 Python примеры:")

    # Запускаем Python примеры
    asyncio.run(example_usage())
