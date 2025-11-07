#!/usr/bin/env python3
"""Специальный тест для executor и critic с детальным анализом."""

import asyncio
import logging
import time
from typing import Dict, Any

from base_tester import BaseTester
from llm.prompts import make_executor_prompt, make_critic_prompt

logger = logging.getLogger(__name__)


class ExecutorCriticTester(BaseTester):
    """Специальный тестер для executor и critic."""
    
    def __init__(self):
        super().__init__("executor_critic")
    
    async def run_test(self) -> Dict[str, Any]:
        """Детальный тест executor и critic."""
        logger.info("🧪 Детальное тестирование executor и critic...")
        
        results = {}
        
        # Тест executor
        logger.info("Тестируем executor...")
        executor_prompt = make_executor_prompt()
        
        start_time = time.perf_counter()
        try:
            if hasattr(self.client, 'generate') and hasattr(self.client.generate, '__code__'):
                response, _ = self.client.generate(executor_prompt)
            else:
                response = self.client.generate_simple(executor_prompt)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Детальный анализ ответа
            analysis = self._analyze_executor_response(response)
            
            results["executor"] = {
                "execution_time": execution_time,
                "response": response,
                "response_length": len(response),
                "prompt_length": len(executor_prompt),
                "analysis": analysis,
                "success": True
            }
            
            logger.info(f"✅ Executor: {execution_time:.2f}с, анализ: {analysis['score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Executor: Ошибка - {e}")
            results["executor"] = {
                "execution_time": 0,
                "response": "",
                "response_length": 0,
                "prompt_length": len(executor_prompt),
                "analysis": {"score": 0, "issues": [f"Ошибка: {e}"]},
                "success": False,
                "error": str(e)
            }
        
        # Тест critic
        logger.info("Тестируем critic...")
        critic_prompt = make_critic_prompt()
        
        start_time = time.perf_counter()
        try:
            if hasattr(self.client, 'generate') and hasattr(self.client.generate, '__code__'):
                response, _ = self.client.generate(critic_prompt)
            else:
                response = self.client.generate_simple(critic_prompt)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Детальный анализ ответа
            analysis = self._analyze_critic_response(response)
            
            results["critic"] = {
                "execution_time": execution_time,
                "response": response,
                "response_length": len(response),
                "prompt_length": len(critic_prompt),
                "analysis": analysis,
                "success": True
            }
            
            logger.info(f"✅ Critic: {execution_time:.2f}с, анализ: {analysis['score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Critic: Ошибка - {e}")
            results["critic"] = {
                "execution_time": 0,
                "response": "",
                "response_length": 0,
                "prompt_length": len(critic_prompt),
                "analysis": {"score": 0, "issues": [f"Ошибка: {e}"]},
                "success": False,
                "error": str(e)
            }
        
        return {
            "individual_results": results,
            "summary": {
                "executor_score": results["executor"]["analysis"]["score"],
                "critic_score": results["critic"]["analysis"]["score"],
                "total_issues": len(results["executor"]["analysis"]["issues"]) + len(results["critic"]["analysis"]["issues"])
            },
            "success": True
        }
    
    def _analyze_executor_response(self, response: str) -> Dict[str, Any]:
        """Анализ ответа executor."""
        response_lower = response.lower()
        
        # Ключевые слова для executor
        expected_keywords = [
            "исполнитель", "hal", "выполняю", "задача", "шаг", 
            "инструмент", "json", "ошибка", "успех", "подтверждение"
        ]
        
        found_keywords = [kw for kw in expected_keywords if kw in response_lower]
        score = len(found_keywords) / len(expected_keywords)
        
        issues = []
        if score < 0.5:
            issues.append("Мало ключевых слов исполнителя")
        if len(response) < 50:
            issues.append("Слишком короткий ответ")
        if "json" not in response_lower:
            issues.append("Не упоминает JSON формат")
        if "ошибка" not in response_lower and "успех" not in response_lower:
            issues.append("Не показывает форматы ответов")
        
        return {
            "score": score,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in expected_keywords if kw not in response_lower],
            "issues": issues
        }
    
    def _analyze_critic_response(self, response: str) -> Dict[str, Any]:
        """Анализ ответа critic."""
        response_lower = response.lower()
        
        # Ключевые слова для critic
        expected_keywords = [
            "критик", "план", "проверка", "json", "безопасность", 
            "выполнимость", "корректность", "ok", "проблемы"
        ]
        
        found_keywords = [kw for kw in expected_keywords if kw in response_lower]
        score = len(found_keywords) / len(expected_keywords)
        
        issues = []
        if score < 0.5:
            issues.append("Мало ключевых слов критика")
        if len(response) < 50:
            issues.append("Слишком короткий ответ")
        if "json" not in response_lower:
            issues.append("Не упоминает проверку JSON")
        if "безопасность" not in response_lower:
            issues.append("Не упоминает безопасность")
        if "ok" not in response_lower and "проблемы" not in response_lower:
            issues.append("Не показывает форматы ответов")
        
        return {
            "score": score,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in expected_keywords if kw not in response_lower],
            "issues": issues
        }


async def main():
    """Запуск теста."""
    tester = ExecutorCriticTester()
    results = await tester.run()
    
    print(f"\n{'='*60}")
    print(tester.generate_summary(results))
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
