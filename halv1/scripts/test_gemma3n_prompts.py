#!/usr/bin/env python3
"""
Скрипт для тестирования модели gemma3n:e4b-it-q8_0 на промптах HAL AI-агента.

Этот скрипт:
1. Тестирует все основные промпты из llm/prompts.py
2. Тестирует промпты из config/prompts.yaml через PromptManager
3. Запускает интеграционные тесты с реальной моделью
4. Собирает метрики качества и выявляет слабые места
5. Предлагает улучшения промптов
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.factory import create_llm_client
from llm.prompts import (
    make_agent_summary_prompt,
    make_code_prompt,
    make_critic_prompt,
    make_executor_prompt,
    make_math_calculation_prompt,
    make_planner_system_prompt,
    make_web_summary_prompt,
)
from llm.prompt_manager import PromptManager
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptTester:
    """Тестер промптов для модели gemma3n:e4b-it-q8_0."""
    
    def __init__(self):
        """Инициализация тестера."""
        self.llm_client = None
        self.prompt_manager = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "unknown",  # будет установлено из конфигурации
            "tests": {},
            "summary": {}
        }
        
    async def setup(self):
        """Настройка клиентов и менеджеров."""
        logger.info("🔧 Настройка тестера...")
        
        # Загружаем конфигурацию
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        llm_config = config.get("llm", {})
        
        # Создаем LLM клиент
        self.llm_client = create_llm_client(
            provider=llm_config.get("provider", "ollama"),
            llm_cfg=llm_config,
            ollama_cfg=llm_config
        )
        
        # Создаем менеджер промптов
        self.prompt_manager = PromptManager("config/prompts.yaml")
        
        # Устанавливаем модель из конфигурации
        self.results["model"] = llm_config.get("model", "unknown")
        
        logger.info("✅ Настройка завершена")
    
    async def test_basic_prompts(self) -> Dict[str, Any]:
        """Тестирование основных промптов из llm/prompts.py."""
        logger.info("🧪 Тестирование основных промптов...")
        
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
            "code_generation_with_error": {
                "prompt_func": make_code_prompt,
                "input": ("Создай функцию для вычисления факториала числа", "SyntaxError: invalid syntax"),
                "expected_keywords": ["def", "factorial", "исправить"]
            },
            "planner_system": {
                "prompt_func": make_planner_system_prompt,
                "input": ["search", "code", "file_io"],
                "expected_keywords": ["планировщик", "JSON", "шаги"]
            },
            "executor": {
                "prompt_func": make_executor_prompt,
                "input": None,
                "expected_keywords": ["исполнитель", "задач", "JSON"]
            },
            "critic": {
                "prompt_func": make_critic_prompt,
                "input": None,
                "expected_keywords": ["критик", "планов", "корректность"]
            }
        }
        
        results = {}
        
        for test_name, test_case in test_cases.items():
            logger.info(f"  📝 Тестирование {test_name}...")
            
            try:
                # Генерируем промпт
                if test_case["input"] is None:
                    prompt = test_case["prompt_func"]()
                elif isinstance(test_case["input"], tuple):
                    prompt = test_case["prompt_func"](*test_case["input"])
                elif isinstance(test_case["input"], dict):
                    prompt = test_case["prompt_func"](**test_case["input"])
                else:
                    prompt = test_case["prompt_func"](test_case["input"])
                
                # Отправляем в модель
                start_time = time.time()
                response = await self._generate_response(prompt)
                response_time = time.time() - start_time
                
                # Анализируем качество ответа
                quality_score = self._analyze_response_quality(
                    response, 
                    test_case["expected_keywords"],
                    test_name
                )
                
                results[test_name] = {
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "response_time": response_time,
                    "quality_score": quality_score,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response,
                    "status": "success"
                }
                
                logger.info(f"    ✅ {test_name}: качество {quality_score:.2f}, время {response_time:.2f}с")
                
            except Exception as e:
                logger.error(f"    ❌ {test_name}: ошибка {str(e)}")
                results[test_name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return results
    
    async def test_prompt_manager_prompts(self) -> Dict[str, Any]:
        """Тестирование промптов из PromptManager."""
        logger.info("🧪 Тестирование промптов PromptManager...")
        
        test_cases = {
            "base_role": {
                "method": "get_system_prompt",
                "args": {"prompt_name": "base_role", "user_name": "@hal0x", "current_time": "2024-01-01T12:00:00", "timezone": "Asia/Bangkok"},
                "expected_keywords": ["HAL", "помощник", "память"]
            },
            "coordinator": {
                "method": "get_system_prompt", 
                "args": {"prompt_name": "coordinator", "user_query": "Помоги с проектом"},
                "expected_keywords": ["Проанализируй", "запрос", "стратегию"]
            },
            "events_short_term": {
                "method": "get_module_prompt",
                "args": {"module": "events", "memory_level": "short_term", "short_term_memory": "Недавние события"},
                "expected_keywords": ["события", "краткосрочные"]
            },
            "themes_long_term": {
                "method": "get_module_prompt",
                "args": {"module": "themes", "memory_level": "long_term", "long_term_memory": "Накопленные знания"},
                "expected_keywords": ["темы", "накопленные"]
            }
        }
        
        results = {}
        
        for test_name, test_case in test_cases.items():
            logger.info(f"  📝 Тестирование {test_name}...")
            
            try:
                # Получаем промпт
                method = getattr(self.prompt_manager, test_case["method"])
                prompt = method(**test_case["args"])
                
                # Отправляем в модель
                start_time = time.time()
                response = await self._generate_response(prompt)
                response_time = time.time() - start_time
                
                # Анализируем качество
                quality_score = self._analyze_response_quality(
                    response,
                    test_case["expected_keywords"],
                    test_name
                )
                
                results[test_name] = {
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "response_time": response_time,
                    "quality_score": quality_score,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response,
                    "status": "success"
                }
                
                logger.info(f"    ✅ {test_name}: качество {quality_score:.2f}, время {response_time:.2f}с")
                
            except Exception as e:
                logger.error(f"    ❌ {test_name}: ошибка {str(e)}")
                results[test_name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return results
    
    async def test_integration_scenarios(self) -> Dict[str, Any]:
        """Тестирование интеграционных сценариев."""
        logger.info("🧪 Тестирование интеграционных сценариев...")
        
        scenarios = {
            "math_calculation": {
                "prompt": make_math_calculation_prompt("15 + 27"),
                "expected_keywords": ["42", "15", "27"]
            },
            "code_generation_task": {
                "prompt": "Создай Python функцию для проверки простого числа",
                "expected_keywords": ["def", "is_prime", "return"]
            },
            "planning_task": {
                "prompt": "Создай план для поиска информации о машинном обучении и создания простого примера",
                "expected_keywords": ["план", "поиск", "пример", "машинное обучение"]
            },
            "web_content_analysis": {
                "prompt": make_web_summary_prompt("Искусственный интеллект (ИИ) — это область компьютерных наук, которая занимается созданием интеллектуальных машин, способных выполнять задачи, обычно требующие человеческого интеллекта."),
                "expected_keywords": ["ИИ", "интеллект", "машины", "задачи"]
            }
        }
        
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            logger.info(f"  📝 Тестирование сценария {scenario_name}...")
            
            try:
                start_time = time.time()
                response = await self._generate_response(scenario["prompt"])
                response_time = time.time() - start_time
                
                quality_score = self._analyze_response_quality(
                    response,
                    scenario["expected_keywords"],
                    scenario_name
                )
                
                results[scenario_name] = {
                    "prompt_length": len(scenario["prompt"]),
                    "response_length": len(response),
                    "response_time": response_time,
                    "quality_score": quality_score,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response,
                    "status": "success"
                }
                
                logger.info(f"    ✅ {scenario_name}: качество {quality_score:.2f}, время {response_time:.2f}с")
                
            except Exception as e:
                logger.error(f"    ❌ {scenario_name}: ошибка {str(e)}")
                results[scenario_name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return results
    
    async def _generate_response(self, prompt: str) -> str:
        """Генерация ответа от модели."""
        try:
            # Используем generate для синхронного вызова
            response = self.llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка: {str(e)}"
    
    def _analyze_response_quality(self, response: str, expected_keywords: List[str], test_name: str) -> float:
        """Анализ качества ответа."""
        if not response or response.startswith("Ошибка:"):
            return 0.0
        
        score = 0.0
        total_keywords = len(expected_keywords)
        
        # Проверяем наличие ключевых слов
        response_lower = response.lower()
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                score += 1.0
        
        # Базовый скор за наличие ответа
        if response:
            score += 0.5
        
        # Штраф за слишком короткие ответы
        if len(response) < 10:
            score *= 0.5
        
        # Бонус за структурированность (JSON, код, списки)
        if any(marker in response for marker in ["{", "}", "def ", "class ", "- ", "1.", "2."]):
            score += 0.2
        
        # Нормализуем скор
        max_possible = total_keywords + 0.7  # 0.5 за ответ + 0.2 за структуру
        return min(score / max_possible, 1.0)
    
    def generate_improvement_suggestions(self) -> List[str]:
        """Генерация предложений по улучшению промптов."""
        suggestions = []
        
        # Анализируем результаты
        all_tests = {}
        for category, tests in self.results["tests"].items():
            all_tests.update(tests)
        
        # Находим слабые места
        low_quality_tests = [
            (name, data) for name, data in all_tests.items()
            if data.get("status") == "success" and data.get("quality_score", 0) < 0.6
        ]
        
        if low_quality_tests:
            suggestions.append("🔧 Слабое качество ответов:")
            for name, data in low_quality_tests:
                suggestions.append(f"  - {name}: качество {data['quality_score']:.2f}")
        
        # Анализируем время ответа
        slow_tests = [
            (name, data) for name, data in all_tests.items()
            if data.get("status") == "success" and data.get("response_time", 0) > 10.0
        ]
        
        if slow_tests:
            suggestions.append("⏱️ Медленные ответы:")
            for name, data in slow_tests:
                suggestions.append(f"  - {name}: {data['response_time']:.2f}с")
        
        # Общие рекомендации
        suggestions.extend([
            "📝 Рекомендации по улучшению:",
            "  1. Добавить более четкие инструкции в промпты",
            "  2. Использовать примеры для лучшего понимания",
            "  3. Добавить валидацию формата ответов",
            "  4. Оптимизировать длину промптов",
            "  5. Добавить fallback-стратегии для ошибок"
        ])
        
        return suggestions
    
    async def run_all_tests(self):
        """Запуск всех тестов."""
        logger.info("🚀 Запуск тестирования gemma3n:e4b-it-q8_0...")
        
        await self.setup()
        
        # Тестируем основные промпты
        self.results["tests"]["basic_prompts"] = await self.test_basic_prompts()
        
        # Тестируем промпты PromptManager
        self.results["tests"]["prompt_manager"] = await self.test_prompt_manager_prompts()
        
        # Тестируем интеграционные сценарии
        self.results["tests"]["integration"] = await self.test_integration_scenarios()
        
        # Генерируем сводку
        self._generate_summary()
        
        # Сохраняем результаты
        self._save_results()
        
        logger.info("✅ Тестирование завершено!")
    
    def _generate_summary(self):
        """Генерация сводки результатов."""
        all_tests = {}
        for category, tests in self.results["tests"].items():
            all_tests.update(tests)
        
        successful_tests = [t for t in all_tests.values() if t.get("status") == "success"]
        failed_tests = [t for t in all_tests.values() if t.get("status") == "error"]
        
        if successful_tests:
            avg_quality = sum(t.get("quality_score", 0) for t in successful_tests) / len(successful_tests)
            avg_time = sum(t.get("response_time", 0) for t in successful_tests) / len(successful_tests)
        else:
            avg_quality = 0.0
            avg_time = 0.0
        
        self.results["summary"] = {
            "total_tests": len(all_tests),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(all_tests) if all_tests else 0,
            "average_quality": avg_quality,
            "average_response_time": avg_time,
            "improvement_suggestions": self.generate_improvement_suggestions()
        }
    
    def _save_results(self):
        """Сохранение результатов в файл."""
        model_name = self.results["model"].replace(":", "_").replace("/", "_")
        results_file = project_root / "scripts" / f"{model_name}_test_results.json"
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Результаты сохранены в {results_file}")
    
    def print_summary(self):
        """Вывод сводки результатов."""
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("📊 СВОДКА ТЕСТИРОВАНИЯ GEMMA3N:E4B-IT-Q8_0")
        print("="*60)
        print(f"📈 Всего тестов: {summary['total_tests']}")
        print(f"✅ Успешных: {summary['successful_tests']}")
        print(f"❌ Неудачных: {summary['failed_tests']}")
        print(f"📊 Успешность: {summary['success_rate']:.1%}")
        print(f"⭐ Среднее качество: {summary['average_quality']:.2f}")
        print(f"⏱️ Среднее время ответа: {summary['average_response_time']:.2f}с")
        
        print("\n🔧 ПРЕДЛОЖЕНИЯ ПО УЛУЧШЕНИЮ:")
        for suggestion in summary['improvement_suggestions']:
            print(suggestion)
        
        print("\n" + "="*60)


async def main():
    """Главная функция."""
    tester = PromptTester()
    
    try:
        await tester.run_all_tests()
        tester.print_summary()
    except KeyboardInterrupt:
        logger.info("⏹️ Тестирование прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
