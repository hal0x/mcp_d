#!/usr/bin/env python3
"""
Тестирование параметра think через API для gemma3n:e4b-it-q8_0.
Проверяем влияние на качество и скорость ответов через HTTP API.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.prompts import (
    make_agent_summary_prompt,
    make_code_prompt,
    make_math_calculation_prompt,
    make_web_summary_prompt,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThinkAPITester:
    """Тестер параметра think через API для gemma3n:e4b-it-q8_0."""
    
    def __init__(self):
        self.api_url = "http://127.0.0.1:11434/api/generate"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "gemma3n:e4b-it-q8_0",
            "think_api_tests": {},
            "summary": {}
        }
    
    async def test_think_level(self, think_level: str, test_name: str):
        """Тестирование конкретного уровня think через API."""
        logger.info(f"🧪 Тестирование {test_name} (think: {think_level})...")
        
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
            },
            "complex_reasoning": {
                "prompt": "Проанализируй следующую задачу и предложи пошаговое решение: 'У нас есть массив чисел [1, 2, 3, 4, 5]. Нужно найти все возможные комбинации из 3 элементов, сумма которых равна 10.'",
                "expected_keywords": ["комбинации", "сумма", "10", "массив"]
            }
        }
        
        results = {}
        total_time = 0
        
        async with aiohttp.ClientSession() as session:
            for test_name_case, test_case in test_cases.items():
                logger.info(f"  📝 {test_name_case}...")
                
                try:
                    start_time = time.time()
                    response = await self._generate_response_with_think_api(session, test_case["prompt"], think_level)
                    response_time = time.time() - start_time
                    total_time += response_time
                    
                    quality_score = self._analyze_response_quality(
                        response, 
                        test_case["expected_keywords"],
                        test_name_case
                    )
                    
                    # Анализ качества рассуждений
                    reasoning_indicators = self._analyze_reasoning_quality(response)
                    
                    results[test_name_case] = {
                        "prompt_length": len(test_case["prompt"]),
                        "response_length": len(response),
                        "response_time": response_time,
                        "quality_score": quality_score,
                        "reasoning_indicators": reasoning_indicators,
                        "response_preview": response[:200] + "..." if len(response) > 200 else response,
                        "status": "success"
                    }
                    
                    logger.info(f"    ✅ качество {quality_score:.2f}, время {response_time:.2f}с, рассуждения {reasoning_indicators['score']:.2f}")
                    
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
            avg_reasoning = sum(t.get("reasoning_indicators", {}).get("score", 0) for t in successful_tests) / len(successful_tests)
        else:
            avg_quality = 0.0
            avg_time = 0.0
            avg_reasoning = 0.0
        
        return {
            "think_level": think_level,
            "tests": results,
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(results) - len(successful_tests),
                "success_rate": len(successful_tests) / len(results) if results else 0,
                "average_quality": avg_quality,
                "average_response_time": avg_time,
                "average_reasoning_score": avg_reasoning,
                "total_time": total_time
            }
        }
    
    async def _generate_response_with_think_api(self, session: aiohttp.ClientSession, prompt: str, think_level: str) -> str:
        """Генерация ответа через API с параметром think."""
        try:
            payload = {
                "model": "gemma3n:e4b-it-q8_0",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "think": think_level,
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "num_ctx": 32768
                }
            }
            
            async with session.post(self.api_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                data = await response.json()
                return data.get("response", "").strip()
                
        except Exception as e:
            logger.error(f"Ошибка API запроса: {e}")
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
        max_possible = total_keywords + 1.0
        return min(score / max_possible, 1.0)
    
    def _analyze_reasoning_quality(self, response: str) -> dict:
        """Анализ качества рассуждений в ответе."""
        if not response or response.startswith("Ошибка:"):
            return {"score": 0.0, "indicators": []}
        
        indicators = []
        score = 0.0
        
        # Индикаторы глубокого рассуждения
        reasoning_markers = [
            "сначала", "затем", "далее", "в результате", "поэтому", "следовательно",
            "рассмотрим", "проанализируем", "разберем", "пошагово", "поэтапно",
            "шаг 1", "шаг 2", "этап", "алгоритм", "логика", "рассуждение",
            "if", "then", "else", "while", "for", "def", "class", "function",
            "let me think", "let's think", "first", "then", "next", "therefore",
            "thinking:", "<thinking>", "</thinking>", "рассуждение:"
        ]
        
        response_lower = response.lower()
        for marker in reasoning_markers:
            if marker in response_lower:
                indicators.append(marker)
                score += 0.1
        
        # Бонус за структурированность
        if any(marker in response for marker in ["1.", "2.", "3.", "•", "-", "*"]):
            indicators.append("структурированность")
            score += 0.2
        
        # Бонус за длину (глубокие рассуждения обычно длиннее)
        if len(response) > 1000:
            indicators.append("очень_детально")
            score += 0.4
        elif len(response) > 500:
            indicators.append("детально")
            score += 0.3
        elif len(response) > 200:
            indicators.append("умеренно_детально")
            score += 0.1
        
        # Бонус за примеры
        if any(word in response_lower for word in ["например", "пример", "например:", "example", "for example"]):
            indicators.append("примеры")
            score += 0.2
        
        # Бонус за thinking mode (если видим промежуточные рассуждения)
        if any(word in response_lower for word in ["<thinking>", "</thinking>", "thinking:", "рассуждение:"]):
            indicators.append("thinking_mode")
            score += 0.3
        
        return {
            "score": min(score, 1.0),
            "indicators": indicators,
            "response_length": len(response)
        }
    
    async def run_all_tests(self):
        """Запуск всех тестов think через API."""
        logger.info("🚀 Запуск тестирования think через API для gemma3n:e4b-it-q8_0...")
        
        # Уровни think для тестирования
        think_levels = {
            "low": "low",
            "medium": "medium", 
            "high": "high"
        }
        
        for level_name, level_value in think_levels.items():
            try:
                logger.info(f"\n🔄 Тестирование уровня: {level_name}")
                result = await self.test_think_level(level_value, level_name)
                self.results["think_api_tests"][level_name] = result
                
                # Небольшая пауза между тестами
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при тестировании {level_name}: {e}")
                continue
        
        self._generate_summary()
        self._save_results()
        
        logger.info("✅ Тестирование think через API завершено!")
    
    def _generate_summary(self):
        """Генерация сводки результатов."""
        tests = self.results["think_api_tests"]
        
        if not tests:
            self.results["summary"] = {"error": "Нет результатов тестирования"}
            return
        
        # Находим лучшие параметры по разным критериям
        best_quality = max(tests.items(), key=lambda x: x[1]["summary"]["average_quality"])
        best_speed = min(tests.items(), key=lambda x: x[1]["summary"]["average_response_time"])
        best_reasoning = max(tests.items(), key=lambda x: x[1]["summary"]["average_reasoning_score"])
        best_balanced = max(tests.items(), key=lambda x: x[1]["summary"]["average_quality"] / x[1]["summary"]["average_response_time"])
        
        self.results["summary"] = {
            "total_think_levels": len(tests),
            "best_quality": {
                "name": best_quality[0],
                "quality": best_quality[1]["summary"]["average_quality"],
                "time": best_quality[1]["summary"]["average_response_time"],
                "reasoning": best_quality[1]["summary"]["average_reasoning_score"]
            },
            "best_speed": {
                "name": best_speed[0],
                "quality": best_speed[1]["summary"]["average_quality"],
                "time": best_speed[1]["summary"]["average_response_time"],
                "reasoning": best_speed[1]["summary"]["average_reasoning_score"]
            },
            "best_reasoning": {
                "name": best_reasoning[0],
                "quality": best_reasoning[1]["summary"]["average_quality"],
                "time": best_reasoning[1]["summary"]["average_response_time"],
                "reasoning": best_reasoning[1]["summary"]["average_reasoning_score"]
            },
            "best_balanced": {
                "name": best_balanced[0],
                "quality": best_balanced[1]["summary"]["average_quality"],
                "time": best_balanced[1]["summary"]["average_response_time"],
                "reasoning": best_balanced[1]["summary"]["average_reasoning_score"],
                "efficiency": best_balanced[1]["summary"]["average_quality"] / best_balanced[1]["summary"]["average_response_time"]
            }
        }
    
    def _save_results(self):
        """Сохранение результатов."""
        results_file = project_root / "scripts" / "gpt_oss_think_api_results.json"
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Результаты сохранены в {results_file}")
    
    def print_summary(self):
        """Вывод сводки результатов."""
        summary = self.results["summary"]
        
        print(f"\n{'='*80}")
        print("📊 СВОДКА ТЕСТИРОВАНИЯ THINK ЧЕРЕЗ API GPT-OSS-20B:LATEST")
        print(f"{'='*80}")
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"📈 Протестировано уровней think: {summary['total_think_levels']}")
        
        print(f"\n🏆 ЛУЧШЕЕ КАЧЕСТВО:")
        best_qual = summary["best_quality"]
        print(f"  Уровень: {best_qual['name']}")
        print(f"  Качество: {best_qual['quality']:.3f}")
        print(f"  Время: {best_qual['time']:.2f}с")
        print(f"  Рассуждения: {best_qual['reasoning']:.3f}")
        
        print(f"\n⚡ ЛУЧШАЯ СКОРОСТЬ:")
        best_speed = summary["best_speed"]
        print(f"  Уровень: {best_speed['name']}")
        print(f"  Качество: {best_speed['quality']:.3f}")
        print(f"  Время: {best_speed['time']:.2f}с")
        print(f"  Рассуждения: {best_speed['reasoning']:.3f}")
        
        print(f"\n🧠 ЛУЧШИЕ РАССУЖДЕНИЯ:")
        best_reason = summary["best_reasoning"]
        print(f"  Уровень: {best_reason['name']}")
        print(f"  Качество: {best_reason['quality']:.3f}")
        print(f"  Время: {best_reason['time']:.2f}с")
        print(f"  Рассуждения: {best_reason['reasoning']:.3f}")
        
        print(f"\n⚖️ ЛУЧШИЙ БАЛАНС:")
        best_bal = summary["best_balanced"]
        print(f"  Уровень: {best_bal['name']}")
        print(f"  Качество: {best_bal['quality']:.3f}")
        print(f"  Время: {best_bal['time']:.2f}с")
        print(f"  Рассуждения: {best_bal['reasoning']:.3f}")
        print(f"  Эффективность: {best_bal['efficiency']:.4f}")
        
        print(f"\n{'='*80}")
        
        # Детальная таблица всех результатов
        print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"{'Уровень':<10} {'Качество':<10} {'Время (с)':<12} {'Рассуждения':<12} {'Эффективность':<12}")
        print("-" * 80)
        
        for level_name, data in self.results["think_api_tests"].items():
            summary_data = data["summary"]
            efficiency = summary_data["average_quality"] / summary_data["average_response_time"] if summary_data["average_response_time"] > 0 else 0
            print(f"{level_name:<10} {summary_data['average_quality']:<10.3f} {summary_data['average_response_time']:<12.2f} {summary_data['average_reasoning_score']:<12.3f} {efficiency:<12.4f}")


async def main():
    """Главная функция."""
    tester = ThinkAPITester()
    
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
