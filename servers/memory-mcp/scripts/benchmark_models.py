#!/usr/bin/env python3
"""
Комплексный бенчмарк для тестирования скорости и качества всех моделей Ollama
в контексте проекта tg_dump
"""

import asyncio
import time
import json
import sys
import os
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import statistics
from tqdm import tqdm

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.memory_mcp.core.ollama_client import OllamaEmbeddingClient


@dataclass
class BenchmarkResult:
    """Результат бенчмарка для одной модели"""
    model_name: str
    test_name: str
    duration: float
    success: bool
    error_message: str = ""
    tokens_per_second: float = 0.0
    quality_score: float = 0.0
    memory_usage_mb: float = 0.0
    additional_metrics: Dict[str, Any] = None


@dataclass
class ModelBenchmark:
    """Полный бенчмарк для модели"""
    model_name: str
    model_size_gb: float
    total_tests: int
    successful_tests: int
    average_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    average_tokens_per_second: float
    average_quality_score: float
    results: List[BenchmarkResult]


class ModelBenchmarker:
    """Класс для проведения бенчмарков моделей"""
    
    def __init__(self):
        # Настройка логирования
        self._setup_logging()
        
        self.models = {
            "hf.co/lmstudio-community/Magistral-Small-2509-GGUF:Q4_K_M": 15.0,
            "dengcao/Qwen3-Embedding-4B:Q5_K_M": 2.9,
            "gemma3n:e4b-it-q8_0": 9.5,
            "gpt-oss-20b:latest": 12.0
        }
        self.ollama_base_url = "http://localhost:11434"
        self.max_context_tokens = 7000  # Ограничение контекста до 7к токенов
        
        # Тестовые данные для разных сценариев
        self.test_scenarios = {
            "short_text": "Привет! Это короткое сообщение для тестирования.",
            "medium_text": """
            В чате обсуждается проект по разработке мобильного приложения. 
            Участники обсуждают различные аспекты: дизайн интерфейса, функциональность, 
            технические требования, сроки разработки. Есть разногласия по поводу 
            выбора технологического стека. Некоторые предлагают React Native, 
            другие - Flutter. Также обсуждается вопрос с базой данных и серверной частью.
            Нужно принять решение по архитектуре приложения и распределить задачи между участниками.
            """,
            "long_text": """
            Это очень длинный документ для тестирования возможностей моделей.
            В нем содержится множество информации о различных аспектах разработки программного обеспечения.
            
            Обсуждение архитектуры системы:
            - Микросервисная архитектура vs монолитная
            - Выбор базы данных (PostgreSQL, MongoDB, Redis)
            - Система кэширования и оптимизация производительности
            - Безопасность и аутентификация пользователей
            - Масштабируемость и нагрузочное тестирование
            
            Технические детали:
            - Использование Docker для контейнеризации
            - CI/CD пайплайны с GitHub Actions
            - Мониторинг и логирование с помощью ELK Stack
            - API документация с помощью Swagger/OpenAPI
            - Тестирование: unit, integration, e2e тесты
            
            Планирование проекта:
            - Agile методология с двухнедельными спринтами
            - Управление задачами в Jira
            - Code review процесс
            - Документация проекта в Confluence
            - Командная работа и коммуникация в Slack
            
            Риски и митигация:
            - Технические риски и способы их минимизации
            - Временные ограничения и приоритизация задач
            - Ресурсы команды и распределение ролей
            - Качество кода и стандарты разработки
            - Поддержка и сопровождение после релиза
            """ * 3,  # Увеличиваем текст
            
            "russian_text": """
            Обсуждение криптовалют и блокчейн технологий в русскоязычном сообществе.
            Участники делятся мнениями о различных проектах, анализируют рынок,
            обсуждают технические аспекты и инвестиционные стратегии.
            
            Основные темы:
            - Анализ курсов Bitcoin, Ethereum и других криптовалют
            - Новые проекты в сфере DeFi и NFT
            - Регулирование криптовалют в разных странах
            - Безопасность хранения и торговли цифровыми активами
            - Майнинг и стейкинг как способы заработка
            
            Технические вопросы:
            - Смарт-контракты и их программирование
            - Слои масштабирования (Layer 2 решения)
            - Межблочные мосты и атомарные свопы
            - Конфиденциальность и анонимность транзакций
            - Интеграция блокчейна с традиционными системами
            """,
            
            "code_discussion": """
            Обсуждение технических вопросов разработки:
            
            ```python
            def process_telegram_messages(messages):
                # Обработка сообщений Telegram
                processed = []
                for msg in messages:
                    if msg.get('text'):
                        processed.append({
                            'id': msg['id'],
                            'text': msg['text'],
                            'timestamp': msg['date']
                        })
                return processed
            ```
            
            Нужно оптимизировать этот код для работы с большими объемами данных.
            Рассматриваются варианты использования asyncio, multiprocessing,
            и различных структур данных для ускорения обработки.
            """,
            
            "multilingual_text": """
            International discussion about technology trends:
            - Artificial Intelligence and Machine Learning developments
            - Cloud computing and serverless architectures
            - Mobile app development with React Native and Flutter
            - Web development with modern frameworks like Next.js
            
            Обсуждение на русском языке:
            - Развитие искусственного интеллекта и машинного обучения
            - Облачные вычисления и бессерверные архитектуры
            - Разработка мобильных приложений
            - Веб-разработка с современными фреймворками
            """
        }
        
        # Обрезаем длинные тексты до 7к токенов (примерно 28к символов)
        self._truncate_test_scenarios()
        
        # Тестовые запросы для эмбеддингов
        self.embedding_queries = [
            "криптовалюты и блокчейн",
            "разработка мобильных приложений", 
            "искусственный интеллект",
            "облачные технологии",
            "веб-разработка",
            "машинное обучение",
            "базы данных",
            "безопасность информации"
        ]

    def _setup_logging(self):
        """Настройка логирования"""
        # Создаем директорию для логов
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Настраиваем логирование
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"benchmark_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Запуск бенчмарка моделей Ollama")
        self.logger.info(f"📝 Логи сохраняются в: {log_file.absolute()}")

    def _truncate_test_scenarios(self):
        """Обрезает тестовые сценарии до 7к токенов (примерно 28к символов)"""
        max_chars = self.max_context_tokens * 4  # Примерно 4 символа на токен
        
        for key, text in self.test_scenarios.items():
            if len(text) > max_chars:
                self.test_scenarios[key] = text[:max_chars] + "\n\n[Текст обрезан для ограничения контекста]"

    async def unload_current_model(self):
        """Выгружает текущую модель из памяти Ollama"""
        try:
            self.logger.info("🔄 Выгружаем текущую модель из памяти...")
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_base_url}/api/generate", 
                                      json={"model": "", "prompt": "", "stream": False}) as response:
                    if response.status == 200:
                        print("  🔄 Модель выгружена из памяти")
                        self.logger.info("✅ Модель успешно выгружена")
                    else:
                        print(f"  ⚠️  Не удалось выгрузить модель: {response.status}")
                        self.logger.warning(f"⚠️ Не удалось выгрузить модель: {response.status}")
        except Exception as e:
            print(f"  ⚠️  Ошибка при выгрузке модели: {e}")
            self.logger.error(f"❌ Ошибка при выгрузке модели: {e}")

    async def load_model(self, model_name: str) -> bool:
        """Загружает модель в память Ollama"""
        try:
            print(f"  📥 Загружаем модель: {model_name}")
            self.logger.info(f"📥 Загружаем модель: {model_name}")
            
            async with aiohttp.ClientSession() as session:
                # Сначала выгружаем текущую модель
                await self.unload_current_model()
                
                # Загружаем новую модель простым запросом
                self.logger.info(f"🔄 Отправляем запрос загрузки модели {model_name}")
                async with session.post(f"{self.ollama_base_url}/api/generate", 
                                      json={
                                          "model": model_name, 
                                          "prompt": "test", 
                                          "stream": False,
                                          "options": {"num_ctx": self.max_context_tokens}
                                      }) as response:
                    if response.status == 200:
                        print(f"  ✅ Модель {model_name} загружена")
                        self.logger.info(f"✅ Модель {model_name} успешно загружена")
                        return True
                    else:
                        print(f"  ❌ Ошибка загрузки модели: {response.status}")
                        self.logger.error(f"❌ Ошибка загрузки модели {model_name}: {response.status}")
                        return False
        except Exception as e:
            print(f"  ❌ Ошибка при загрузке модели {model_name}: {e}")
            self.logger.error(f"❌ Ошибка при загрузке модели {model_name}: {e}")
            return False

    async def test_model_availability(self, model_name: str) -> bool:
        """Проверка доступности модели"""
        try:
            async with OllamaEmbeddingClient() as client:
                result = await client.test_connection()
                return result.get("model_available", False)
        except Exception as e:
            print(f"❌ Ошибка проверки модели {model_name}: {e}")
            return False

    async def benchmark_text_generation(self, model_name: str, text: str, test_name: str) -> BenchmarkResult:
        """Бенчмарк генерации текста/саммаризации"""
        start_time = time.time()
        self.logger.info(f"🔤 Запуск теста {test_name} для модели {model_name}")
        
        try:
            # Создаем клиент с указанной моделью и ограничением контекста
            async with OllamaEmbeddingClient(model_name=model_name) as client:
                # Генерируем саммаризацию
                self.logger.debug(f"📝 Генерируем саммаризацию для текста длиной {len(text)} символов")
                summary = await client.generate_summary(text)
                duration = time.time() - start_time
                
                # Оцениваем качество (простая эвристика)
                quality_score = self._evaluate_summary_quality(text, summary)
                
                # Подсчитываем токены в секунду (примерно)
                estimated_tokens = len(text.split()) + len(summary.split())
                tokens_per_second = estimated_tokens / duration if duration > 0 else 0
                
                self.logger.info(f"✅ Тест {test_name} завершен за {duration:.2f}с, качество: {quality_score:.3f}")
                
                return BenchmarkResult(
                    model_name=model_name,
                    test_name=test_name,
                    duration=duration,
                    success=True,
                    tokens_per_second=tokens_per_second,
                    quality_score=quality_score,
                    additional_metrics={
                        "input_length": len(text),
                        "output_length": len(summary),
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary
                    }
                )
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка в тесте {test_name}: {e}")
            return BenchmarkResult(
                model_name=model_name,
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    async def benchmark_embeddings(self, model_name: str, texts: List[str], test_name: str) -> BenchmarkResult:
        """Бенчмарк генерации эмбеддингов"""
        start_time = time.time()
        
        try:
            # Создаем клиент с указанной моделью
            async with OllamaEmbeddingClient(model_name=model_name) as client:
                embeddings = await client.generate_embeddings(texts)
                duration = time.time() - start_time
                
                # Оцениваем качество эмбеддингов (сходство между похожими текстами)
                quality_score = self._evaluate_embedding_quality(embeddings, texts)
                
                # Подсчитываем токены в секунду
                total_tokens = sum(len(text.split()) for text in texts)
                tokens_per_second = total_tokens / duration if duration > 0 else 0
                
                return BenchmarkResult(
                    model_name=model_name,
                    test_name=test_name,
                    duration=duration,
                    success=True,
                    tokens_per_second=tokens_per_second,
                    quality_score=quality_score,
                    additional_metrics={
                        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                        "num_embeddings": len(embeddings),
                        "total_input_length": sum(len(text) for text in texts)
                    }
                )
                
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _evaluate_summary_quality(self, original_text: str, summary: str) -> float:
        """Простая оценка качества саммаризации"""
        if not summary:
            return 0.0
            
        # Базовые метрики качества
        compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 0
        
        # Проверяем наличие ключевых слов из оригинала
        original_words = set(original_text.lower().split())
        summary_words = set(summary.lower().split())
        keyword_overlap = len(original_words.intersection(summary_words)) / len(original_words) if original_words else 0
        
        # Комбинированная оценка (0-1)
        quality_score = min(1.0, keyword_overlap * 0.7 + (1 - compression_ratio) * 0.3)
        return quality_score

    def _evaluate_embedding_quality(self, embeddings: List[List[float]], texts: List[str]) -> float:
        """Оценка качества эмбеддингов через анализ сходства"""
        if len(embeddings) < 2:
            return 0.0
            
        try:
            import numpy as np
            
            # Вычисляем косинусное сходство между эмбеддингами
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    emb1 = np.array(embeddings[i])
                    emb2 = np.array(embeddings[j])
                    
                    # Нормализуем векторы
                    norm1 = np.linalg.norm(emb1)
                    norm2 = np.linalg.norm(emb2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                        similarities.append(similarity)
            
            if similarities:
                # Оценка качества на основе разброса сходств
                mean_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                
                # Хорошие эмбеддинги должны иметь разумный разброс сходств
                quality_score = min(1.0, max(0.0, 1 - std_similarity))
                return quality_score
            
        except ImportError:
            # Если numpy недоступен, возвращаем базовую оценку
            return 0.5
            
        return 0.0

    async def run_comprehensive_benchmark(self) -> Dict[str, ModelBenchmark]:
        """Запуск комплексного бенчмарка всех моделей"""
        print("🚀 Запуск комплексного бенчмарка моделей Ollama")
        print(f"📏 Максимальный контекст: {self.max_context_tokens} токенов")
        print("=" * 60)
        
        self.logger.info("🚀 Начинаем комплексный бенчмарк всех моделей")
        self.logger.info(f"📏 Максимальный контекст: {self.max_context_tokens} токенов")
        
        results = {}
        total_models = len(self.models)
        
        # Создаем прогресс-бар для моделей
        model_progress = tqdm(self.models.items(), 
                            desc="🤖 Тестирование моделей", 
                            unit="модель",
                            total=total_models)
        
        for model_name, model_size in model_progress:
            model_progress.set_description(f"🤖 Тестируем {model_name.split('/')[-1]}")
            
            print(f"\n📊 Тестирование модели: {model_name}")
            print(f"📦 Размер: {model_size} GB")
            print("-" * 50)
            
            self.logger.info(f"📊 Начинаем тестирование модели: {model_name} ({model_size} GB)")
            
            # Загружаем модель
            if not await self.load_model(model_name):
                print(f"❌ Не удалось загрузить модель {model_name}, пропускаем")
                self.logger.error(f"❌ Не удалось загрузить модель {model_name}, пропускаем")
                continue
            
            model_results = []
            
            # Определяем список тестов для выполнения
            tests = [
                ("short_text_summarization", "🔤 Тест 1: Саммаризация короткого текста", "short_text"),
                ("medium_text_summarization", "📝 Тест 2: Саммаризация среднего текста", "medium_text"),
                ("long_text_summarization", "📄 Тест 3: Саммаризация длинного текста", "long_text"),
                ("russian_text_summarization", "🇷🇺 Тест 4: Саммаризация русского текста", "russian_text"),
                ("code_discussion_summarization", "💻 Тест 5: Саммаризация технического текста", "code_discussion"),
                ("short_embeddings", "🔢 Тест 6: Эмбеддинги коротких текстов", "embedding_short"),
                ("all_embeddings", "🔢 Тест 7: Эмбеддинги всех текстов", "embedding_all"),
                ("parallel_processing", "⚡ Тест 8: Параллельная обработка", "parallel")
            ]
            
            # Создаем прогресс-бар для тестов
            test_progress = tqdm(tests, desc="🧪 Тесты", unit="тест", leave=False)
            
            for test_id, test_description, test_type in test_progress:
                test_progress.set_description(test_description)
                print(f"{test_description}...")
                
                if test_type == "embedding_short":
                    result = await self.benchmark_embeddings(
                        model_name, 
                        self.embedding_queries[:4], 
                        test_id
                    )
                elif test_type == "embedding_all":
                    result = await self.benchmark_embeddings(
                        model_name, 
                        self.embedding_queries, 
                        test_id
                    )
                elif test_type == "parallel":
                    # Параллельная обработка
                    parallel_start = time.time()
                    tasks = []
                    for i, scenario_name in enumerate(["short_text", "medium_text", "russian_text"]):
                        task = self.benchmark_text_generation(
                            model_name, 
                            self.test_scenarios[scenario_name], 
                            f"parallel_test_{i}"
                        )
                        tasks.append(task)
                    
                    parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                    parallel_duration = time.time() - parallel_start
                    
                    # Создаем результат для параллельного теста
                    successful_parallel = sum(1 for r in parallel_results if isinstance(r, BenchmarkResult) and r.success)
                    result = BenchmarkResult(
                        model_name=model_name,
                        test_name=test_id,
                        duration=parallel_duration,
                        success=successful_parallel > 0,
                        tokens_per_second=sum(r.tokens_per_second for r in parallel_results if isinstance(r, BenchmarkResult)) / len(parallel_results),
                        quality_score=sum(r.quality_score for r in parallel_results if isinstance(r, BenchmarkResult)) / len(parallel_results),
                        additional_metrics={
                            "successful_tasks": successful_parallel,
                            "total_tasks": len(tasks)
                        }
                    )
                else:
                    # Обычные тесты саммаризации
                    result = await self.benchmark_text_generation(
                        model_name, 
                        self.test_scenarios[test_type], 
                        test_id
                    )
                
                model_results.append(result)
                self._print_test_result(result)
            
            
            # Создаем сводку по модели
            successful_results = [r for r in model_results if r.success]
            durations = [r.duration for r in successful_results]
            tokens_per_sec = [r.tokens_per_second for r in successful_results]
            quality_scores = [r.quality_score for r in successful_results]
            
            model_benchmark = ModelBenchmark(
                model_name=model_name,
                model_size_gb=model_size,
                total_tests=len(model_results),
                successful_tests=len(successful_results),
                average_duration=statistics.mean(durations) if durations else 0,
                median_duration=statistics.median(durations) if durations else 0,
                min_duration=min(durations) if durations else 0,
                max_duration=max(durations) if durations else 0,
                average_tokens_per_second=statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
                average_quality_score=statistics.mean(quality_scores) if quality_scores else 0,
                results=model_results
            )
            
            results[model_name] = model_benchmark
            
            print(f"\n✅ Модель {model_name} протестирована")
            print(f"📊 Успешных тестов: {model_benchmark.successful_tests}/{model_benchmark.total_tests}")
            print(f"⏱️  Среднее время: {model_benchmark.average_duration:.2f}с")
            print(f"🎯 Средняя оценка качества: {model_benchmark.average_quality_score:.3f}")
            
            self.logger.info(f"✅ Модель {model_name} протестирована")
            self.logger.info(f"📊 Успешных тестов: {model_benchmark.successful_tests}/{model_benchmark.total_tests}")
            self.logger.info(f"⏱️ Среднее время: {model_benchmark.average_duration:.2f}с")
            self.logger.info(f"🎯 Средняя оценка качества: {model_benchmark.average_quality_score:.3f}")
            
            # Выгружаем модель после тестирования
            await self.unload_current_model()
        
        return results

    def _print_test_result(self, result: BenchmarkResult):
        """Вывод результата теста"""
        if result.success:
            print(f"  ✅ {result.test_name}: {result.duration:.2f}с, "
                  f"{result.tokens_per_second:.1f} ток/с, качество: {result.quality_score:.3f}")
        else:
            print(f"  ❌ {result.test_name}: ОШИБКА - {result.error_message}")

    def generate_report(self, results: Dict[str, ModelBenchmark]) -> str:
        """Генерация отчета по результатам бенчмарка"""
        report = []
        report.append("# 📊 Отчет по бенчмарку моделей Ollama")
        report.append("=" * 60)
        report.append("")
        
        # Сводная таблица
        report.append("## 📋 Сводная таблица результатов")
        report.append("")
        report.append("| Модель | Размер (GB) | Тесты | Успех | Ср. время (с) | Ср. ток/с | Качество |")
        report.append("|--------|-------------|-------|-------|---------------|-----------|----------|")
        
        for model_name, benchmark in results.items():
            report.append(f"| {model_name.split('/')[-1]} | {benchmark.model_size_gb} | "
                         f"{benchmark.total_tests} | {benchmark.successful_tests} | "
                         f"{benchmark.average_duration:.2f} | {benchmark.average_tokens_per_second:.1f} | "
                         f"{benchmark.average_quality_score:.3f} |")
        
        report.append("")
        
        # Детальные результаты по каждой модели
        for model_name, benchmark in results.items():
            report.append(f"## 🤖 {model_name}")
            report.append("")
            report.append(f"**Размер модели:** {benchmark.model_size_gb} GB")
            report.append(f"**Успешных тестов:** {benchmark.successful_tests}/{benchmark.total_tests}")
            report.append(f"**Среднее время выполнения:** {benchmark.average_duration:.2f} секунд")
            report.append(f"**Медианное время:** {benchmark.median_duration:.2f} секунд")
            report.append(f"**Диапазон времени:** {benchmark.min_duration:.2f} - {benchmark.max_duration:.2f} секунд")
            report.append(f"**Средняя скорость:** {benchmark.average_tokens_per_second:.1f} токенов/секунда")
            report.append(f"**Средняя оценка качества:** {benchmark.average_quality_score:.3f}")
            report.append("")
            
            # Детали по каждому тесту
            report.append("### 📝 Детали тестов:")
            report.append("")
            for result in benchmark.results:
                status = "✅" if result.success else "❌"
                report.append(f"- {status} **{result.test_name}**: {result.duration:.2f}с")
                if result.success:
                    report.append(f"  - Скорость: {result.tokens_per_second:.1f} ток/с")
                    report.append(f"  - Качество: {result.quality_score:.3f}")
                    if result.additional_metrics:
                        for key, value in result.additional_metrics.items():
                            report.append(f"  - {key}: {value}")
                else:
                    report.append(f"  - Ошибка: {result.error_message}")
            report.append("")
        
        # Рекомендации
        report.append("## 🎯 Рекомендации")
        report.append("")
        
        # Лучшая модель по скорости
        fastest_model = min(results.values(), key=lambda x: x.average_duration)
        report.append(f"**⚡ Самая быстрая модель:** {fastest_model.model_name}")
        report.append(f"   - Среднее время: {fastest_model.average_duration:.2f} секунд")
        report.append("")
        
        # Лучшая модель по качеству
        best_quality_model = max(results.values(), key=lambda x: x.average_quality_score)
        report.append(f"**🏆 Лучшее качество:** {best_quality_model.model_name}")
        report.append(f"   - Оценка качества: {best_quality_model.average_quality_score:.3f}")
        report.append("")
        
        # Лучшая модель по соотношению скорость/качество
        balanced_scores = []
        for model_name, benchmark in results.items():
            if benchmark.average_quality_score > 0 and benchmark.average_tokens_per_second > 0:
                # Нормализуем метрики и вычисляем баланс
                speed_score = benchmark.average_tokens_per_second / max(b.average_tokens_per_second for b in results.values())
                quality_score = benchmark.average_quality_score / max(b.average_quality_score for b in results.values())
                balance_score = (speed_score + quality_score) / 2
                balanced_scores.append((model_name, balance_score))
        
        if balanced_scores:
            best_balanced = max(balanced_scores, key=lambda x: x[1])
            report.append(f"**⚖️ Лучший баланс скорость/качество:** {best_balanced[0]}")
            report.append(f"   - Сбалансированная оценка: {best_balanced[1]:.3f}")
            report.append("")
        
        # Рекомендации по использованию
        report.append("### 💡 Рекомендации по использованию:")
        report.append("")
        report.append("- **Для быстрой обработки:** используйте самую быструю модель")
        report.append("- **Для качественного анализа:** используйте модель с лучшим качеством")
        report.append("- **Для продакшена:** рассмотрите модель с лучшим балансом")
        report.append("- **Для экспериментов:** попробуйте разные модели для разных задач")
        report.append("")
        
        return "\n".join(report)

    async def save_results(self, results: Dict[str, ModelBenchmark], filename: str = None):
        """Сохранение результатов в JSON файл"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Конвертируем результаты в сериализуемый формат
        serializable_results = {}
        for model_name, benchmark in results.items():
            serializable_results[model_name] = {
                "model_name": benchmark.model_name,
                "model_size_gb": benchmark.model_size_gb,
                "total_tests": benchmark.total_tests,
                "successful_tests": benchmark.successful_tests,
                "average_duration": benchmark.average_duration,
                "median_duration": benchmark.median_duration,
                "min_duration": benchmark.min_duration,
                "max_duration": benchmark.max_duration,
                "average_tokens_per_second": benchmark.average_tokens_per_second,
                "average_quality_score": benchmark.average_quality_score,
                "results": [asdict(result) for result in benchmark.results]
            }
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в файл: {filepath.absolute()}")


async def main():
    """Основная функция для запуска бенчмарка"""
    benchmarker = ModelBenchmarker()
    
    try:
        # Запускаем комплексный бенчмарк
        benchmarker.logger.info("🚀 Начинаем выполнение бенчмарка")
        results = await benchmarker.run_comprehensive_benchmark()
        
        if not results:
            print("❌ Не удалось протестировать ни одной модели")
            benchmarker.logger.error("❌ Не удалось протестировать ни одной модели")
            return
        
        # Генерируем отчет
        benchmarker.logger.info("📊 Генерируем отчет по результатам")
        report = benchmarker.generate_report(results)
        
        # Сохраняем результаты
        await benchmarker.save_results(results)
        
        # Сохраняем отчет
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{timestamp}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Отчет сохранен в файл: {report_filename}")
        benchmarker.logger.info(f"📄 Отчет сохранен в файл: {report_filename}")
        
        # Выводим краткую сводку
        print("\n" + "="*60)
        print("🎉 БЕНЧМАРК ЗАВЕРШЕН!")
        print("="*60)
        benchmarker.logger.info("🎉 Бенчмарк успешно завершен!")
        
        for model_name, benchmark in results.items():
            print(f"\n🤖 {model_name.split('/')[-1]}:")
            print(f"   ⏱️  Время: {benchmark.average_duration:.2f}с")
            print(f"   🎯 Качество: {benchmark.average_quality_score:.3f}")
            print(f"   ⚡ Скорость: {benchmark.average_tokens_per_second:.1f} ток/с")
            print(f"   ✅ Успех: {benchmark.successful_tests}/{benchmark.total_tests}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Бенчмарк прерван пользователем")
        benchmarker.logger.warning("⏹️ Бенчмарк прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении бенчмарка: {e}")
        benchmarker.logger.error(f"❌ Ошибка при выполнении бенчмарка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
