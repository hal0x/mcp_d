#!/usr/bin/env python3
"""Тест переиспользования контекста."""

import asyncio
import logging
import time
from typing import Dict, Any

from base_tester import BaseTester

logger = logging.getLogger(__name__)


class ContextReuseTester(BaseTester):
    """Тестер переиспользования контекста."""
    
    def __init__(self):
        super().__init__("context_reuse")
    
    async def run_test(self) -> Dict[str, Any]:
        """Тест переиспользования контекста."""
        logger.info("🧪 Тестирование переиспользования контекста...")
        
        # Тестовые промпты для последовательной обработки
        prompts = [
            "Расскажи о Python программировании",
            "Какие основные библиотеки используются в Python?",
            "Как создать веб-приложение на Python?",
            "Какие фреймворки популярны для Python веб-разработки?",
            "Как развернуть Python приложение в продакшене?"
        ]
        
        times_with_context = []
        times_without_context = []
        
        # Тест с переиспользованием контекста
        if hasattr(self.client, 'generate') and hasattr(self.client.generate, '__code__'):
            try:
                context = None
                for i, prompt in enumerate(prompts):
                    start_time = time.perf_counter()
                    response, context = self.client.generate(prompt, context)
                    end_time = time.perf_counter()
                    times_with_context.append(end_time - start_time)
                    logger.info(f"Запрос {i+1} с контекстом: {end_time - start_time:.2f}с")
            except TypeError:
                logger.warning("Клиент не поддерживает новый API, используем старый")
                for i, prompt in enumerate(prompts):
                    start_time = time.perf_counter()
                    response = self.client.generate_simple(prompt)
                    end_time = time.perf_counter()
                    times_without_context.append(end_time - start_time)
                    logger.info(f"Запрос {i+1} без контекста: {end_time - start_time:.2f}с")
        else:
            # Используем старый API
            for i, prompt in enumerate(prompts):
                start_time = time.perf_counter()
                response = self.client.generate_simple(prompt)
                end_time = time.perf_counter()
                times_without_context.append(end_time - start_time)
                logger.info(f"Запрос {i+1}: {end_time - start_time:.2f}с")
        
        return {
            "with_context": times_with_context,
            "without_context": times_without_context,
            "avg_with_context": sum(times_with_context) / len(times_with_context) if times_with_context else 0,
            "avg_without_context": sum(times_without_context) / len(times_without_context) if times_without_context else 0,
            "total_prompts": len(prompts),
            "success": True
        }


async def main():
    """Запуск теста."""
    tester = ContextReuseTester()
    results = await tester.run()
    
    print(f"\n{'='*60}")
    print(tester.generate_summary(results))
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
