#!/usr/bin/env python3
"""
Скрипт для отправки запроса с картинкой в Ollama через Python
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path


async def send_image_request(image_path: str, prompt: str = "Опиши что изображено на этой картинке"):
    """Отправка запроса с изображением в Ollama"""
    
    # Проверяем существование файла
    if not Path(image_path).exists():
        print(f"❌ Файл {image_path} не найден")
        return
    
    # Читаем и кодируем изображение в base64
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Подготавливаем запрос
    request_data = {
        "model": "hf.co/lmstudio-community/Magistral-Small-2509-GGUF:Q4_K_M",
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    
    print(f"🖼️ Отправляем запрос с изображением: {image_path}")
    print(f"📝 Промпт: {prompt}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Ответ получен:")
                    print("-" * 50)
                    print(result.get("response", "Нет ответа"))
                    print("-" * 50)
                    
                    # Дополнительная информация
                    if "eval_duration" in result:
                        print(f"⏱️ Время обработки: {result['eval_duration']/1e9:.2f} секунд")
                    
                    if "eval_count" in result:
                        print(f"🔢 Количество токенов: {result['eval_count']}")
                        
                else:
                    print(f"❌ Ошибка: {response.status}")
                    error_text = await response.text()
                    print(f"Детали ошибки: {error_text}")
                    
    except Exception as e:
        print(f"❌ Ошибка при отправке запроса: {e}")


async def main():
    """Основная функция"""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python ollama_image_request.py <путь_к_изображению> [промпт]")
        print("Пример: python ollama_image_request.py image.jpg 'Что изображено на картинке?'")
        return
    
    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Опиши что изображено на этой картинке"
    
    await send_image_request(image_path, prompt)


if __name__ == "__main__":
    asyncio.run(main())
