#!/usr/bin/env python3
"""Тест длины промптов."""

import asyncio
import logging
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


class PromptLengthTester(BaseTester):
    """Тестер длины промптов."""
    
    def __init__(self):
        super().__init__("prompt_length")
    
    async def run_test(self) -> Dict[str, Any]:
        """Тест длины промптов."""
        logger.info("🧪 Тестирование длины промптов...")
        
        # Сравниваем длину промптов
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
        
        # Анализируем длинные промпты
        long_prompts = {k: v for k, v in prompt_lengths.items() if v > 500}
        
        return {
            "individual_lengths": prompt_lengths,
            "total_length": total_length,
            "average_length": avg_length,
            "long_prompts": long_prompts,
            "optimization_recommendations": self._get_optimization_recommendations(prompt_lengths),
            "success": True
        }
    
    def _get_optimization_recommendations(self, lengths: Dict[str, int]) -> Dict[str, str]:
        """Получение рекомендаций по оптимизации."""
        recommendations = {}
        
        for name, length in lengths.items():
            if length > 800:
                recommendations[name] = f"Критично длинный ({length} символов) - сократить до ~500"
            elif length > 500:
                recommendations[name] = f"Длинный ({length} символов) - можно сократить до ~400"
            else:
                recommendations[name] = f"Оптимальный ({length} символов)"
        
        return recommendations


async def main():
    """Запуск теста."""
    tester = PromptLengthTester()
    results = await tester.run()
    
    print(f"\n{'='*60}")
    print(tester.generate_summary(results))
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
