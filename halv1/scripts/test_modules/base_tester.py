#!/usr/bin/env python3
"""Базовый класс для модульных тестов производительности."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm.factory import create_llm_client

logger = logging.getLogger(__name__)


class BaseTester(ABC):
    """Базовый класс для тестирования производительности."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.client = None
        self.results = {}
        self.results_dir = Path("scripts/test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def setup(self):
        """Настройка клиента."""
        logger.info(f"🔧 Настройка LLM клиента для {self.test_name}...")
        
        self.client = create_llm_client(
            provider="ollama",
            llm_cfg={
                "model": "gemma3n:e4b-it-q8_0",
                "num_ctx": 16384,
                "num_keep": 256,
            },
            ollama_cfg={
                "keep_alive": "30m",
                "num_batch": 1024,
            }
        )
        
        logger.info(f"✅ Клиент настроен для {self.test_name}")
    
    @abstractmethod
    async def run_test(self) -> Dict[str, Any]:
        """Запуск конкретного теста."""
        pass
    
    def save_results(self, results: Dict[str, Any]):
        """Сохранение результатов в файл."""
        results_file = self.results_dir / f"{self.test_name}_results.json"
        
        # Добавляем метаданные
        full_results = {
            "test_name": self.test_name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": results
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Результаты сохранены в {results_file}")
        return results_file
    
    def load_previous_results(self) -> Optional[Dict[str, Any]]:
        """Загрузка предыдущих результатов."""
        results_file = self.results_dir / f"{self.test_name}_results.json"
        
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    async def run(self):
        """Запуск полного цикла тестирования."""
        logger.info(f"🚀 Запуск теста {self.test_name}...")
        
        try:
            await self.setup()
            results = await self.run_test()
            self.save_results(results)
            
            logger.info(f"✅ Тест {self.test_name} завершен успешно")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка в тесте {self.test_name}: {e}")
            raise
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """Генерация краткого отчета."""
        return f"""
# 📊 {self.test_name.upper()} - КРАТКИЙ ОТЧЕТ

## 📈 СТАТИСТИКА
- Время выполнения: {results.get('execution_time', 0):.2f}с
- Качество: {results.get('quality_score', 0):.2f}
- Успешность: {results.get('success', False)}
- Длина ответа: {results.get('response_length', 0)} символов

## 📝 ДЕТАЛИ
{json.dumps(results, ensure_ascii=False, indent=2)}
"""
