#!/usr/bin/env python3
"""Тест качества промптов."""

import asyncio
import logging
import time
from typing import Dict, Any

from base_tester import BaseTester
from llm.prompts import (
    make_web_summary_prompt,
    make_agent_summary_prompt,
    make_code_prompt,
    make_math_calculation_prompt,
    make_planner_system_prompt,
    make_executor_prompt,
    make_critic_prompt
)

logger = logging.getLogger(__name__)


class PromptQualityTester(BaseTester):
    """Тестер качества промптов."""
    
    def __init__(self):
        super().__init__("prompt_quality")
    
    async def run_test(self) -> Dict[str, Any]:
        """Тест качества промптов."""
        logger.info("🧪 Тестирование качества промптов...")
        
        test_cases = {
            "web_summary": {
                "prompt_func": make_web_summary_prompt,
                "input": "Python — высокоуровневый язык программирования общего назначения с динамической типизацией и автоматическим управлением памятью.",
                "expected_keywords": ["Python", "программирование", "язык"]
            },
            "agent_summary": {
                "prompt_func": make_agent_summary_prompt,
                "input": {
                    "mode": "summary",
                    "user_name": "@hal0x",
                    "theme": "Разработка",
                    "timezone": "Asia/Bangkok",
                    "window_start": "2024-01-01T00:00:00",
                    "window_end": "2024-01-01T23:59:59",
                    "now_iso": "2024-01-01T12:00:00",
                    "messages_block": "1|chat|user|2024-01-01T10:00:00|Привет! Как дела с проектом?"
                },
                "expected_keywords": ["HAL", "сводка", "важные"]
            },
            "code_generation": {
                "prompt_func": make_code_prompt,
                "input": "Создай функцию для вычисления факториала числа",
                "expected_keywords": ["def", "factorial", "return"]
            },
            "math_calculation": {
                "prompt_func": make_math_calculation_prompt,
                "input": "15 * 27 + 42",
                "expected_keywords": ["447"]
            },
            "planner_system": {
                "prompt_func": make_planner_system_prompt,
                "input": ["search", "code", "file_io"],
                "expected_keywords": ["JSON", "steps", "tool"]
            },
            "executor": {
                "prompt_func": lambda x: make_executor_prompt(),
                "input": "",
                "expected_keywords": ["ИСПОЛНИТЕЛЬ", "HAL", "выполняю"]
            },
            "critic": {
                "prompt_func": lambda x: make_critic_prompt(),
                "input": "",
                "expected_keywords": ["КРИТИК", "план", "проверка"]
            }
        }
        
        results = {}
        
        for test_name, test_case in test_cases.items():
            logger.info(f"Тестируем {test_name}...")
            
            # Создаем промпт
            if test_name == "agent_summary":
                prompt = test_case["prompt_func"](**test_case["input"])
            elif test_name == "planner_system":
                prompt = test_case["prompt_func"](test_case["input"])
            else:
                prompt = test_case["prompt_func"](test_case["input"])
            
            # Измеряем время выполнения
            start_time = time.perf_counter()
            
            try:
                if hasattr(self.client, 'generate') and hasattr(self.client.generate, '__code__'):
                    response, _ = self.client.generate(prompt)
                else:
                    response = self.client.generate_simple(prompt)
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Проверяем качество ответа
                response_lower = response.lower()
                quality_score = sum(1 for keyword in test_case["expected_keywords"] 
                                  if keyword.lower() in response_lower) / len(test_case["expected_keywords"])
                
                results[test_name] = {
                    "execution_time": execution_time,
                    "quality_score": quality_score,
                    "response_length": len(response),
                    "prompt_length": len(prompt),
                    "success": True
                }
                
                logger.info(f"✅ {test_name}: {execution_time:.2f}с, качество: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"❌ {test_name}: Ошибка - {e}")
                results[test_name] = {
                    "execution_time": 0,
                    "quality_score": 0,
                    "response_length": 0,
                    "prompt_length": len(prompt),
                    "success": False,
                    "error": str(e)
                }
        
        # Вычисляем общую статистику
        successful_tests = [r for r in results.values() if r["success"]]
        avg_execution_time = sum(r["execution_time"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_quality = sum(r["quality_score"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        
        return {
            "individual_results": results,
            "summary": {
                "total_tests": len(test_cases),
                "successful_tests": len(successful_tests),
                "avg_execution_time": avg_execution_time,
                "avg_quality": avg_quality
            },
            "success": True
        }


async def main():
    """Запуск теста."""
    tester = PromptQualityTester()
    results = await tester.run()
    
    print(f"\n{'='*60}")
    print(tester.generate_summary(results))
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
