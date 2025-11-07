#!/usr/bin/env python3
"""
Тестирование gemma3n:e4b-it-q8_0 с разными параметрами размышления.
Ищем оптимальные настройки temperature, top_p, и других параметров.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.factory import create_llm_client
from llm.prompts import (
    make_agent_summary_prompt,
    make_code_prompt,
    make_math_calculation_prompt,
    make_web_summary_prompt,
)
import yaml

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParameterTester:
    """Тестер параметров для gemma3n:e4b-it-q8_0."""
    
    def __init__(self):
        self.base_config = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "gemma3n:e4b-it-q8_0",
            "parameter_tests": {},
            "summary": {}
        }
    
    async def setup(self):
        """Настройка базовой конфигурации."""
        logger.info("🔧 Загрузка базовой конфигурации...")
        
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.base_config = config.get("llm", {}).copy()
        self.base_config["model"] = "gemma3n:e4b-it-q8_0"
        
        logger.info("✅ Конфигурация загружена")
    
    async def test_parameter_set(self, params: dict, test_name: str):
        """Тестирование набора параметров."""
        logger.info(f"🧪 Тестирование {test_name}...")
        
        # Создаем конфигурацию с тестируемыми параметрами
        test_config = self.base_config.copy()
        test_config.update(params)
        
        # Создаем LLM клиент
        llm_client = create_llm_client(
            provider=test_config.get("provider", "ollama"),
            llm_cfg=test_config,
            ollama_cfg=test_config
        )
        
        # Тестовые кейсы
        test_cases = {
            "web_summary": {
                "prompt": make_web_summary_prompt("Python — высокоуровневый язык программирования общего назначения с динамической типизацией и автоматическим управлением памятью."),
                "expected_keywords": ["Python", "программирование", "язык"]
            },
            "agent_summary": {
                "prompt": make_agent_summary_prompt(
                    mode="summary",
                    user_name="@hal0x",
                    theme="Разработка",
                    timezone="Asia/Bangkok",
                    window_start="2024-01-01T00:00:00",
                    window_end="2024-01-01T23:59:59",
                    now_iso="2024-01-01T12:00:00",
                    messages_block="1|chat|user|2024-01-01T10:00:00|Привет! Как дела с проектом?"
                ),
                "expected_keywords": ["HAL", "сводка", "важные"]
            },
            "code_generation": {
                "prompt": make_code_prompt("Создай функцию для вычисления факториала числа"),
                "expected_keywords": ["def", "factorial", "return"]
            },
            "math_calculation": {
                "prompt": make_math_calculation_prompt("15 + 27"),
                "expected_keywords": ["42", "15", "27"]
            }
        }
        
        results = {}
        total_time = 0
        
        for test_name_case, test_case in test_cases.items():
            logger.info(f"  📝 {test_name_case}...")
            
            try:
                start_time = time.time()
                response = await self._generate_response(llm_client, test_case["prompt"])
                response_time = time.time() - start_time
                total_time += response_time
                
                quality_score = self._analyze_response_quality(
                    response, 
                    test_case["expected_keywords"],
                    test_name_case
                )
                
                results[test_name_case] = {
                    "prompt_length": len(test_case["prompt"]),
                    "response_length": len(response),
                    "response_time": response_time,
                    "quality_score": quality_score,
                    "response_preview": response[:150] + "..." if len(response) > 150 else response,
                    "status": "success"
                }
                
                logger.info(f"    ✅ качество {quality_score:.2f}, время {response_time:.2f}с")
                
            except Exception as e:
                logger.error(f"    ❌ ошибка {str(e)}")
                results[test_name_case] = {
                    "error": str(e),
                    "status": "error"
                }
        
        # Вычисляем средние показатели
        successful_tests = [t for t in results.values() if t.get("status") == "success"]
        if successful_tests:
            avg_quality = sum(t.get("quality_score", 0) for t in successful_tests) / len(successful_tests)
            avg_time = sum(t.get("response_time", 0) for t in successful_tests) / len(successful_tests)
        else:
            avg_quality = 0.0
            avg_time = 0.0
        
        return {
            "parameters": params,
            "tests": results,
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(results) - len(successful_tests),
                "success_rate": len(successful_tests) / len(results) if results else 0,
                "average_quality": avg_quality,
                "average_response_time": avg_time,
                "total_time": total_time
            }
        }
    
    async def _generate_response(self, llm_client, prompt: str) -> str:
        """Генерация ответа от модели."""
        try:
            response = llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка: {str(e)}"
    
    def _analyze_response_quality(self, response: str, expected_keywords: list, test_name: str) -> float:
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
        
        # Бонус за структурированность
        if any(marker in response for marker in ["{", "}", "def ", "class ", "- ", "1.", "2."]):
            score += 0.2
        
        # Бонус за русский язык (для русских промптов)
        if test_name in ["web_summary", "agent_summary", "math_calculation"]:
            russian_indicators = ["это", "для", "что", "как", "в", "на", "с", "по", "от", "до"]
            if any(indicator in response_lower for indicator in russian_indicators):
                score += 0.3
        
        # Нормализуем скор
        max_possible = total_keywords + 1.0  # 0.5 за ответ + 0.2 за структуру + 0.3 за русский
        return min(score / max_possible, 1.0)
    
    async def run_all_tests(self):
        """Запуск всех тестов параметров."""
        logger.info("🚀 Запуск тестирования параметров gemma3n:e4b-it-q8_0...")
        
        await self.setup()
        
        # Наборы параметров для тестирования
        parameter_sets = {
            "conservative": {
                "temperature": 0.1,
                "top_p": 0.8,
                "seed": 42
            },
            "balanced": {
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": 42
            },
            "creative": {
                "temperature": 0.7,
                "top_p": 0.95,
                "seed": 42
            },
            "very_creative": {
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": 42
            },
            "focused": {
                "temperature": 0.2,
                "top_p": 0.85,
                "seed": 42
            },
            "deterministic": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42
            },
            "high_quality": {
                "temperature": 0.4,
                "top_p": 0.9,
                "seed": 42,
                "num_ctx": 32768,
                "num_keep": 1024
            },
            "fast": {
                "temperature": 0.2,
                "top_p": 0.8,
                "seed": 42,
                "num_ctx": 16384,
                "num_keep": 512
            }
        }
        
        for param_name, params in parameter_sets.items():
            try:
                logger.info(f"\n🔄 Тестирование набора: {param_name}")
                result = await self.test_parameter_set(params, param_name)
                self.results["parameter_tests"][param_name] = result
                
                # Небольшая пауза между тестами
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при тестировании {param_name}: {e}")
                continue
        
        self._generate_summary()
        self._save_results()
        
        logger.info("✅ Тестирование параметров завершено!")
    
    def _generate_summary(self):
        """Генерация сводки результатов."""
        tests = self.results["parameter_tests"]
        
        if not tests:
            self.results["summary"] = {"error": "Нет результатов тестирования"}
            return
        
        # Находим лучшие параметры по разным критериям
        best_quality = max(tests.items(), key=lambda x: x[1]["summary"]["average_quality"])
        best_speed = min(tests.items(), key=lambda x: x[1]["summary"]["average_response_time"])
        best_balanced = max(tests.items(), key=lambda x: x[1]["summary"]["average_quality"] / x[1]["summary"]["average_response_time"])
        
        self.results["summary"] = {
            "total_parameter_sets": len(tests),
            "best_quality": {
                "name": best_quality[0],
                "quality": best_quality[1]["summary"]["average_quality"],
                "time": best_quality[1]["summary"]["average_response_time"]
            },
            "best_speed": {
                "name": best_speed[0],
                "quality": best_speed[1]["summary"]["average_quality"],
                "time": best_speed[1]["summary"]["average_response_time"]
            },
            "best_balanced": {
                "name": best_balanced[0],
                "quality": best_balanced[1]["summary"]["average_quality"],
                "time": best_balanced[1]["summary"]["average_response_time"],
                "efficiency": best_balanced[1]["summary"]["average_quality"] / best_balanced[1]["summary"]["average_response_time"]
            }
        }
    
    def _save_results(self):
        """Сохранение результатов."""
        results_file = project_root / "scripts" / "gpt_oss_parameter_test_results.json"
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Результаты сохранены в {results_file}")
    
    def print_summary(self):
        """Вывод сводки результатов."""
        summary = self.results["summary"]
        
        print(f"\n{'='*80}")
        print("📊 СВОДКА ТЕСТИРОВАНИЯ ПАРАМЕТРОВ GPT-OSS-20B:LATEST")
        print(f"{'='*80}")
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"📈 Протестировано наборов параметров: {summary['total_parameter_sets']}")
        
        print(f"\n🏆 ЛУЧШЕЕ КАЧЕСТВО:")
        best_qual = summary["best_quality"]
        print(f"  Набор: {best_qual['name']}")
        print(f"  Качество: {best_qual['quality']:.3f}")
        print(f"  Время: {best_qual['time']:.2f}с")
        
        print(f"\n⚡ ЛУЧШАЯ СКОРОСТЬ:")
        best_speed = summary["best_speed"]
        print(f"  Набор: {best_speed['name']}")
        print(f"  Качество: {best_speed['quality']:.3f}")
        print(f"  Время: {best_speed['time']:.2f}с")
        
        print(f"\n⚖️ ЛУЧШИЙ БАЛАНС:")
        best_bal = summary["best_balanced"]
        print(f"  Набор: {best_bal['name']}")
        print(f"  Качество: {best_bal['quality']:.3f}")
        print(f"  Время: {best_bal['time']:.2f}с")
        print(f"  Эффективность: {best_bal['efficiency']:.4f}")
        
        print(f"\n{'='*80}")
        
        # Детальная таблица всех результатов
        print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"{'Набор параметров':<20} {'Качество':<10} {'Время (с)':<12} {'Эффективность':<12}")
        print("-" * 80)
        
        for param_name, data in self.results["parameter_tests"].items():
            summary_data = data["summary"]
            efficiency = summary_data["average_quality"] / summary_data["average_response_time"] if summary_data["average_response_time"] > 0 else 0
            print(f"{param_name:<20} {summary_data['average_quality']:<10.3f} {summary_data['average_response_time']:<12.2f} {efficiency:<12.4f}")


async def main():
    """Главная функция."""
    tester = ParameterTester()
    
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
