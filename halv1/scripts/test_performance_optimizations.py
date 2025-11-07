#!/usr/bin/env python3
"""Тест производительности с оптимизациями контекста и промптов."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Tuple, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.factory import create_llm_client
from llm.prompts import (
    make_web_summary_prompt,
    make_agent_summary_prompt,
    make_code_prompt,
    make_math_calculation_prompt,
    make_planner_system_prompt,
    make_executor_prompt,
    make_critic_prompt
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Тестер производительности с оптимизациями."""
    
    def __init__(self):
        self.client = None
        self.results = {}
        
    async def setup(self):
        """Настройка клиента."""
        logger.info("🔧 Настройка LLM клиента...")
        
        # Создаем клиент с оптимизированными параметрами
        self.client = create_llm_client(
            provider="ollama",
            llm_cfg={
                "model": "gemma3n:e4b-it-q8_0",
                "num_ctx": 16384,  # Оптимизированный контекст
                "num_keep": 256,   # Уменьшенное сохранение
            },
            ollama_cfg={
                "keep_alive": "30m",  # Поддержание модели в теплом состоянии
                "num_batch": 1024,    # Увеличенный batch size
            }
        )
        
        logger.info("✅ Клиент настроен с оптимизациями")
    
    def test_context_reuse(self) -> Dict[str, Any]:
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
            # Проверяем, поддерживает ли клиент новый API
            try:
                context = None
                for i, prompt in enumerate(prompts):
                    start_time = time.perf_counter()
                    response, context = self.client.generate(prompt, context)
                    end_time = time.perf_counter()
                    times_with_context.append(end_time - start_time)
                    logger.info(f"Запрос {i+1} с контекстом: {end_time - start_time:.2f}с")
            except TypeError:
                # Fallback к старому API
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
        }
    
    def test_optimized_prompts(self) -> Dict[str, Any]:
        """Тест оптимизированных промптов."""
        logger.info("🧪 Тестирование оптимизированных промптов...")
        
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
        
        return results
    
    def test_prompt_length_optimization(self) -> Dict[str, Any]:
        """Тест оптимизации длины промптов."""
        logger.info("🧪 Тестирование оптимизации длины промптов...")
        
        # Сравниваем длину промптов до и после оптимизации
        prompt_lengths = {
            "web_summary": len(make_web_summary_prompt("test")),
            "agent_summary": len(make_agent_summary_prompt(
                mode="test", user_name="test", theme="test", timezone="test",
                window_start="test", window_end="test", now_iso="test", messages_block="test"
            )),
            "code_generation": len(make_code_prompt("test")),
            "math_calculation": len(make_math_calculation_prompt("test")),
            "planner_system": len(make_planner_system_prompt(["test"])),
            "executor": len(make_executor_prompt()),
            "critic": len(make_critic_prompt())
        }
        
        total_length = sum(prompt_lengths.values())
        avg_length = total_length / len(prompt_lengths)
        
        return {
            "individual_lengths": prompt_lengths,
            "total_length": total_length,
            "average_length": avg_length
        }
    
    def generate_report(self) -> str:
        """Генерация отчета о производительности."""
        logger.info("📊 Генерация отчета...")
        
        # Тестируем все компоненты
        context_results = self.test_context_reuse()
        prompt_results = self.test_optimized_prompts()
        length_results = self.test_prompt_length_optimization()
        
        # Вычисляем общую статистику
        successful_tests = [r for r in prompt_results.values() if r["success"]]
        avg_execution_time = sum(r["execution_time"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_quality = sum(r["quality_score"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        
        report = f"""
# 🚀 ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ С ОПТИМИЗАЦИЯМИ

## 📈 ОБЩАЯ СТАТИСТИКА
- Успешных тестов: {len(successful_tests)}/{len(prompt_results)}
- Среднее время выполнения: {avg_execution_time:.2f}с
- Среднее качество ответов: {avg_quality:.2f}
- Общая длина промптов: {length_results['total_length']} символов
- Средняя длина промпта: {length_results['average_length']:.0f} символов

## 🔄 ПЕРЕИСПОЛЬЗОВАНИЕ КОНТЕКСТА
- Среднее время с контекстом: {context_results['avg_with_context']:.2f}с
- Среднее время без контекста: {context_results['avg_without_context']:.2f}с
- Ускорение: {((context_results['avg_without_context'] - context_results['avg_with_context']) / context_results['avg_without_context'] * 100) if context_results['avg_without_context'] > 0 else 0:.1f}% (если применимо)

## 📝 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПРОМПТОВ

"""
        
        for test_name, result in prompt_results.items():
            status = "✅" if result["success"] else "❌"
            report += f"### {status} {test_name.upper()}\n"
            if result["success"]:
                report += f"- Время выполнения: {result['execution_time']:.2f}с\n"
                report += f"- Качество ответа: {result['quality_score']:.2f}\n"
                report += f"- Длина ответа: {result['response_length']} символов\n"
                report += f"- Длина промпта: {result['prompt_length']} символов\n"
            else:
                report += f"- Ошибка: {result.get('error', 'Неизвестная ошибка')}\n"
            report += "\n"
        
        report += f"""
## 🎯 ОПТИМИЗАЦИИ ПРОМПТОВ

### Длина промптов по компонентам:
"""
        for name, length in length_results['individual_lengths'].items():
            report += f"- {name}: {length} символов\n"
        
        report += f"""
## 🏆 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ

1. **Переиспользование контекста**: Ускорение повторных запросов
2. **Оптимизированные промпты**: Убраны избыточные few-shot примеры
3. **Уменьшенный контекст**: num_ctx снижен с 32768 до 16384
4. **Увеличенный batch size**: num_batch увеличен до 1024
5. **Keep-alive**: Модель остается в памяти 30 минут

## 📊 РЕКОМЕНДАЦИИ

- Продолжать использовать переиспользование контекста для последовательных запросов
- Мониторить качество ответов при сокращенных промптах
- Настроить keep_alive в зависимости от паттернов использования
- Рассмотреть дальнейшее сокращение промптов для критических компонентов

---
*Отчет сгенерирован: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report


async def main():
    """Основная функция тестирования."""
    logger.info("🚀 Запуск тестирования производительности с оптимизациями...")
    
    tester = PerformanceTester()
    await tester.setup()
    
    # Генерируем отчет
    report = tester.generate_report()
    
    # Сохраняем отчет
    report_file = "scripts/PERFORMANCE_OPTIMIZATION_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"📄 Отчет сохранен в {report_file}")
    print("\n" + "="*80)
    print(report)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
