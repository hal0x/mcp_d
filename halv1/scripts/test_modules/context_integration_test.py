#!/usr/bin/env python3
"""Тест интеграции контекстно-осведомленных компонентов."""

import asyncio
import logging
import time
from typing import Dict, Any

from base_tester import BaseTester
from llm.context_factory import create_context_aware_client, create_context_aware_code_generator

logger = logging.getLogger(__name__)


class ContextIntegrationTester(BaseTester):
    """Тестер интеграции контекстно-осведомленных компонентов."""
    
    def __init__(self):
        super().__init__("context_integration")
    
    async def run_test(self) -> Dict[str, Any]:
        """Тест интеграции контекстно-осведомленных компонентов."""
        logger.info("🧪 Тестирование интеграции контекстно-осведомленных компонентов...")
        
        results = {}
        
        # Тест 1: Контекстно-осведомленный клиент
        logger.info("Тестируем контекстно-осведомленный клиент...")
        context_client = create_context_aware_client(
            provider="ollama",
            llm_cfg={"model": "gemma3n:e4b-it-q8_0"},
            ollama_cfg={"keep_alive": "30m", "num_batch": 1024}
        )
        
        # Последовательные запросы с переиспользованием контекста
        prompts = [
            "Расскажи о Python программировании",
            "Какие основные библиотеки используются в Python?",
            "Как создать веб-приложение на Python?"
        ]
        
        context_times = []
        context_responses = []
        
        for i, prompt in enumerate(prompts):
            start_time = time.perf_counter()
            response, context = context_client.generate(prompt)
            end_time = time.perf_counter()
            
            context_times.append(end_time - start_time)
            context_responses.append(response)
            
            logger.info(f"Запрос {i+1} с контекстом: {end_time - start_time:.2f}с")
        
        results["context_client"] = {
            "execution_times": context_times,
            "avg_time": sum(context_times) / len(context_times),
            "responses": context_responses,
            "context_reused": context_client.get_context() is not None,
            "success": True
        }
        
        # Тест 2: Контекстно-осведомленный генератор кода
        logger.info("Тестируем контекстно-осведомленный генератор кода...")
        code_generator = create_context_aware_code_generator(
            provider="ollama",
            llm_cfg={"model": "gemma3n:e4b-it-q8_0"},
            ollama_cfg={"keep_alive": "30m", "num_batch": 1024}
        )
        
        code_tasks = [
            "Создай функцию для вычисления факториала",
            "Создай функцию для сортировки списка",
            "Создай класс для работы с файлами"
        ]
        
        code_times = []
        code_results = []
        
        for i, task in enumerate(code_tasks):
            start_time = time.perf_counter()
            try:
                code = code_generator.generate(task)
                end_time = time.perf_counter()
                
                code_times.append(end_time - start_time)
                code_results.append({
                    "task": task,
                    "code": code,
                    "success": True
                })
                
                logger.info(f"Генерация кода {i+1}: {end_time - start_time:.2f}с")
                
            except Exception as e:
                end_time = time.perf_counter()
                code_times.append(end_time - start_time)
                code_results.append({
                    "task": task,
                    "error": str(e),
                    "success": False
                })
                
                logger.error(f"Ошибка генерации кода {i+1}: {e}")
        
        results["code_generator"] = {
            "execution_times": code_times,
            "avg_time": sum(code_times) / len(code_times),
            "results": code_results,
            "success_rate": sum(1 for r in code_results if r["success"]) / len(code_results),
            "success": True
        }
        
        # Тест 3: Сравнение производительности
        logger.info("Сравниваем производительность с контекстом и без...")
        
        # Тест без контекста (сброс контекста)
        context_client.clear_context()
        
        no_context_times = []
        for i, prompt in enumerate(prompts):
            start_time = time.perf_counter()
            response, _ = context_client.generate(prompt)
            end_time = time.perf_counter()
            
            no_context_times.append(end_time - start_time)
            logger.info(f"Запрос {i+1} без контекста: {end_time - start_time:.2f}с")
        
        # Вычисляем улучшение производительности
        context_avg = sum(context_times) / len(context_times)
        no_context_avg = sum(no_context_times) / len(no_context_times)
        improvement = ((no_context_avg - context_avg) / no_context_avg * 100) if no_context_avg > 0 else 0
        
        results["performance_comparison"] = {
            "context_avg_time": context_avg,
            "no_context_avg_time": no_context_avg,
            "improvement_percent": improvement,
            "context_times": context_times,
            "no_context_times": no_context_times,
            "success": True
        }
        
        # Общая статистика
        results["summary"] = {
            "total_tests": 3,
            "successful_tests": sum(1 for r in results.values() if r.get("success", False)),
            "context_improvement": improvement,
            "code_generation_success_rate": results["code_generator"]["success_rate"]
        }
        
        return results


async def main():
    """Запуск теста."""
    tester = ContextIntegrationTester()
    results = await tester.run()
    
    print(f"\n{'='*60}")
    print(tester.generate_summary(results))
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
