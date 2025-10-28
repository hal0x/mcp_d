#!/usr/bin/env python3
"""
Расширение OllamaEmbeddingClient для работы с изображениями
"""

import asyncio
import base64
import json
from pathlib import Path
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.memory_mcp.core.ollama_client import OllamaEmbeddingClient


class OllamaVisionClient(OllamaEmbeddingClient):
    """Клиент для работы с изображениями через Ollama"""
    
    async def analyze_image(self, image_path: str, prompt: str = "Опиши что изображено на этой картинке") -> str:
        """Анализ изображения с помощью модели"""
        
        # Проверяем существование файла
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Файл {image_path} не найден")
        
        # Читаем и кодируем изображение в base64
        with open(image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Подготавливаем запрос
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        print(f"🖼️ Анализируем изображение: {image_path}")
        print(f"📝 Промпт: {prompt}")
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "Нет ответа")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ошибка API: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"❌ Ошибка при анализе изображения: {e}")
            raise

    async def compare_images(self, image1_path: str, image2_path: str, prompt: str = "Сравни эти два изображения") -> str:
        """Сравнение двух изображений"""
        
        # Проверяем существование файлов
        for path in [image1_path, image2_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Файл {path} не найден")
        
        # Читаем и кодируем изображения в base64
        images_base64 = []
        for path in [image1_path, image2_path]:
            with open(path, 'rb') as image_file:
                images_base64.append(base64.b64encode(image_file.read()).decode('utf-8'))
        
        # Подготавливаем запрос
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_base64,
            "stream": False
        }
        
        print(f"🖼️ Сравниваем изображения: {image1_path} и {image2_path}")
        print(f"📝 Промпт: {prompt}")
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "Нет ответа")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ошибка API: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"❌ Ошибка при сравнении изображений: {e}")
            raise


async def demo_vision_client():
    """Демонстрация работы с изображениями"""
    
    print("🚀 Демонстрация работы с изображениями через Ollama")
    print("=" * 60)
    
    # Создаем клиент
    async with OllamaVisionClient() as client:
        
        # Проверяем доступность модели
        if not await client.check_model_availability():
            print("❌ Модель недоступна")
            return
        
        print("✅ Модель доступна")
        
        # Примеры использования (замените на реальные пути к изображениям)
        example_images = [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg"
        ]
        
        for image_path in example_images:
            if Path(image_path).exists():
                try:
                    print(f"\n📸 Анализируем: {image_path}")
                    result = await client.analyze_image(image_path)
                    print("✅ Результат:")
                    print("-" * 40)
                    print(result)
                    print("-" * 40)
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print(f"⚠️ Файл {image_path} не найден, пропускаем")
        
        # Если есть два изображения, можно их сравнить
        if len(example_images) >= 2 and all(Path(p).exists() for p in example_images[:2]):
            try:
                print(f"\n🔍 Сравниваем изображения")
                result = await client.compare_images(example_images[0], example_images[1])
                print("✅ Результат сравнения:")
                print("-" * 40)
                print(result)
                print("-" * 40)
            except Exception as e:
                print(f"❌ Ошибка при сравнении: {e}")


if __name__ == "__main__":
    asyncio.run(demo_vision_client())
