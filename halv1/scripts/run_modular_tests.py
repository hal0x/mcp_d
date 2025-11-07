#!/usr/bin/env python3
"""Запуск модульных тестов производительности."""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "test_modules"))

from test_modules.context_reuse_test import ContextReuseTester
from test_modules.prompt_quality_test import PromptQualityTester
from test_modules.prompt_length_test import PromptLengthTester
from test_modules.executor_critic_test import ExecutorCriticTester
from test_modules.context_integration_test import ContextIntegrationTester

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModularTestRunner:
    """Запускатор модульных тестов."""
    
    def __init__(self):
        self.testers = {
            "context": ContextReuseTester,
            "quality": PromptQualityTester,
            "length": PromptLengthTester,
            "executor-critic": ExecutorCriticTester,
            "context-integration": ContextIntegrationTester,
        }
    
    async def run_single_test(self, test_name: str):
        """Запуск одного теста."""
        if test_name not in self.testers:
            logger.error(f"❌ Неизвестный тест: {test_name}")
            logger.info(f"Доступные тесты: {', '.join(self.testers.keys())}")
            return
        
        logger.info(f"🚀 Запуск теста: {test_name}")
        
        try:
            tester = self.testers[test_name]()
            results = await tester.run()
            
            print(f"\n{'='*80}")
            print(f"# 📊 РЕЗУЛЬТАТЫ ТЕСТА: {test_name.upper()}")
            print(f"{'='*80}")
            print(tester.generate_summary(results))
            print(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в тесте {test_name}: {e}")
            raise
    
    async def run_all_tests(self):
        """Запуск всех тестов."""
        logger.info("🚀 Запуск всех модульных тестов...")
        
        results = {}
        
        for test_name, tester_class in self.testers.items():
            try:
                logger.info(f"🧪 Запуск теста: {test_name}")
                tester = tester_class()
                result = await tester.run()
                results[test_name] = result
                logger.info(f"✅ Тест {test_name} завершен")
                
            except Exception as e:
                logger.error(f"❌ Ошибка в тесте {test_name}: {e}")
                results[test_name] = {"error": str(e), "success": False}
        
        # Генерируем общий отчет
        self.generate_combined_report(results)
    
    def generate_combined_report(self, results: dict):
        """Генерация общего отчета."""
        report = f"""
# 🚀 ОБЩИЙ ОТЧЕТ МОДУЛЬНЫХ ТЕСТОВ

## 📈 СВОДКА
- Всего тестов: {len(results)}
- Успешных: {sum(1 for r in results.values() if r.get('success', False))}
- С ошибками: {sum(1 for r in results.values() if not r.get('success', True))}

## 📝 РЕЗУЛЬТАТЫ ПО ТЕСТАМ

"""
        
        for test_name, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            report += f"### {status} {test_name.upper()}\n"
            
            if result.get('success', False):
                if 'summary' in result:
                    summary = result['summary']
                    report += f"- Среднее время: {summary.get('avg_execution_time', 0):.2f}с\n"
                    report += f"- Среднее качество: {summary.get('avg_quality', 0):.2f}\n"
                elif 'execution_time' in result:
                    report += f"- Время выполнения: {result.get('execution_time', 0):.2f}с\n"
                    report += f"- Качество: {result.get('quality_score', 0):.2f}\n"
            else:
                report += f"- Ошибка: {result.get('error', 'Неизвестная ошибка')}\n"
            
            report += "\n"
        
        report += f"""
## 🎯 РЕКОМЕНДАЦИИ

1. **Для быстрого тестирования**: используйте отдельные тесты
2. **Для полного анализа**: запустите все тесты
3. **При ошибках**: проверьте логи и исправьте проблемы
4. **Мониторинг**: регулярно запускайте тесты для отслеживания производительности

---
*Отчет сгенерирован: {asyncio.get_event_loop().time()}*
"""
        
        # Сохраняем отчет
        report_file = Path("scripts/test_results/COMBINED_REPORT.md")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"📄 Общий отчет сохранен в {report_file}")
        print(report)


async def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Модульные тесты производительности")
    parser.add_argument(
        "test_name", 
        nargs="?", 
        choices=["context", "quality", "length", "executor-critic", "context-integration", "all"],
        default="all",
        help="Название теста для запуска (по умолчанию: all)"
    )
    
    args = parser.parse_args()
    
    runner = ModularTestRunner()
    
    if args.test_name == "all":
        await runner.run_all_tests()
    else:
        await runner.run_single_test(args.test_name)


if __name__ == "__main__":
    asyncio.run(main())
