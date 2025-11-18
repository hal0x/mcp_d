#!/usr/bin/env python3
"""CLI интерфейс для Telegram Dump Manager v2.0."""

import asyncio
import json
import logging
import math
import os
import re
import signal
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click

from ..utils.russian_tokenizer import tokenize_text as enhanced_tokenize

# Отключаем телеметрию ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = ""

from ..analysis.insight_graph import SummaryInsightAnalyzer
from ..analysis.instruction_manager import InstructionManager
from ..core.indexer import TwoLevelIndexer
from ..indexing import TelegramIndexer
from ..memory.ingest import MemoryIngestor
from ..memory.typed_graph import TypedGraphMemory
from ..utils.message_extractor import MessageExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """Класс для удаления дубликатов сообщений по полю 'id'."""

    def __init__(self, chats_dir: str = "chats"):
        self.chats_dir = Path(chats_dir)
        self.stats = {
            "total_chats": 0,
            "processed_chats": 0,
            "total_messages": 0,
            "duplicates_removed": 0,
            "unique_messages": 0,
            "errors": 0,
        }

    def deduplicate_chat(self, chat_dir: Path) -> Dict[str, int]:
        """Дедупликация сообщений в одном чате."""
        chat_stats = {
            "total_messages": 0,
            "duplicates_removed": 0,
            "unique_messages": 0,
            "errors": 0,
        }

        if not chat_dir.exists():
            return chat_stats

        all_messages = []
        for json_file in chat_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            message = json.loads(line.strip())
                            if isinstance(message, dict):
                                all_messages.append(message)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Ошибка чтения файла {json_file}: {e}")
                chat_stats["errors"] += 1

        chat_stats["total_messages"] = len(all_messages)

        from ..utils.deduplication import deduplicate_by_id

        unique_messages = deduplicate_by_id(all_messages)
        chat_stats["duplicates_removed"] = len(all_messages) - len(unique_messages)
        chat_stats["unique_messages"] = len(unique_messages)

        if unique_messages != all_messages:
            temp_file = chat_dir / "temp_dedup.json"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    for message in unique_messages:
                        f.write(json.dumps(message, ensure_ascii=False) + "\n")

                for json_file in chat_dir.glob("*.json"):
                    if json_file.name != "temp_dedup.json":
                        json_file.unlink()

                final_file = chat_dir / "messages.json"
                temp_file.rename(final_file)

            except Exception as e:
                logger.error(f"Ошибка записи файла для чата {chat_dir.name}: {e}")
                chat_stats["errors"] += 1
                if temp_file.exists():
                    temp_file.unlink()

        return chat_stats

    def deduplicate_all_chats(self) -> Dict[str, int]:
        """Дедупликация сообщений во всех чатах."""
        if not self.chats_dir.exists():
            logger.error(f"Директория {self.chats_dir} не найдена")
            return self.stats

        # Получаем список всех чатов
        chat_dirs = [d for d in self.chats_dir.iterdir() if d.is_dir()]
        self.stats["total_chats"] = len(chat_dirs)

        for chat_dir in chat_dirs:
            logger.info(f"Дедупликация чата: {chat_dir.name}")

            # Дедуплицируем сообщения в чате
            chat_stats = self.deduplicate_chat(chat_dir)

            # Обновляем общую статистику
            for key, value in chat_stats.items():
                self.stats[key] += value

            self.stats["processed_chats"] += 1

            logger.info(
                f"Чата {chat_dir.name}: {chat_stats['duplicates_removed']} дубликатов удалено"
            )

        return self.stats

    def print_stats(self):
        """Вывод статистики дедупликации."""
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА ДЕДУПЛИКАЦИИ")
        print("=" * 60)
        print(f"📁 Всего чатов: {self.stats['total_chats']}")
        print(f"✅ Обработано чатов: {self.stats['processed_chats']}")
        print(f"📨 Всего сообщений: {self.stats['total_messages']}")
        print(f"🔄 Дубликатов удалено: {self.stats['duplicates_removed']}")
        print(f"✨ Уникальных сообщений: {self.stats['unique_messages']}")
        print(f"❌ Ошибок: {self.stats['errors']}")
        print("=" * 60)


class ProcessManager:
    """Класс для управления процессами индексации."""

    @staticmethod
    def kill_processes_by_name(pattern: str) -> int:
        """Останавливает процессы по имени."""
        killed_count = 0
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            lines = result.stdout.split("\n")

            for line in lines:
                if pattern in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            killed_count += 1
                            logger.info(f"Процесс {pid} ({pattern}) остановлен")
                        except (ValueError, ProcessLookupError):
                            continue
        except Exception as e:
            logger.error(f"Ошибка при остановке процессов {pattern}: {e}")

        return killed_count

    @staticmethod
    def stop_ollama():
        """Остановка Ollama сервера."""
        logger.info("🛑 Остановка Ollama сервера...")

        try:
            result = subprocess.run(
                ["ollama", "stop"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("✅ Ollama сервер остановлен")
            else:
                logger.warning("⚠️ Ollama stop не сработал, пробуем kill")
                ProcessManager.kill_processes_by_name("ollama")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Timeout при остановке Ollama, пробуем kill")
            ProcessManager.kill_processes_by_name("ollama")
        except FileNotFoundError:
            logger.warning("⚠️ Ollama не найден в PATH, пробуем kill")
            ProcessManager.kill_processes_by_name("ollama")
        except Exception as e:
            logger.error(f"❌ Ошибка остановки Ollama: {e}")

    @staticmethod
    def stop_indexing_processes():
        """Остановка процессов индексации."""
        logger.info("🛑 Остановка процессов индексации...")

        patterns = [
            "tg_dump.py",
            "index_messages.py",
            "summarize_chats.py",
            "index_summaries.py",
            "cross_analyze.py",
            "ollama",
        ]

        total_killed = 0
        for pattern in patterns:
            killed = ProcessManager.kill_processes_by_name(pattern)
            total_killed += killed

        if total_killed > 0:
            logger.info(f"✅ Остановлено процессов: {total_killed}")
        else:
            logger.info("ℹ️ Процессы индексации не найдены")

    @staticmethod
    def check_remaining_processes():
        """Проверка оставшихся процессов."""
        logger.info("🔍 Проверка оставшихся процессов...")

        patterns = [
            "tg_dump.py",
            "index_messages.py",
            "summarize_chats.py",
            "index_summaries.py",
            "cross_analyze.py",
            "ollama",
        ]

        remaining = []
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            lines = result.stdout.split("\n")

            for line in lines:
                for pattern in patterns:
                    if pattern in line and "grep" not in line:
                        remaining.append(line.strip())
                        break
        except Exception as e:
            logger.error(f"Ошибка проверки процессов: {e}")

        if remaining:
            logger.warning(f"⚠️ Найдено {len(remaining)} оставшихся процессов:")
            for proc in remaining:
                logger.warning(f"   {proc}")
        else:
            logger.info("✅ Все процессы остановлены")

    @staticmethod
    def stop_all_indexing():
        """Остановка всех процессов индексации."""
        logger.info("🛑 ОСТАНОВКА ВСЕХ ПРОЦЕССОВ ИНДЕКСАЦИИ")
        logger.info("=" * 50)

        ProcessManager.stop_indexing_processes()
        ProcessManager.stop_ollama()

        import time
        time.sleep(2)

        ProcessManager.check_remaining_processes()

        logger.info("=" * 50)
        logger.info("✅ Остановка процессов завершена")


@click.group()
@click.version_option(version="2.0.0", prog_name="memory_mcp")
@click.option("--verbose", "-v", is_flag=True, help="Подробный вывод")
@click.option("--quiet", "-q", is_flag=True, help="Тихий режим")
def cli(verbose, quiet):
    """🚀 Telegram Dump Manager v2.0 - Управление дампами Telegram чатов

    Современный CLI для двухуровневой индексации и анализа Telegram чатов.

    Основные команды:
      • index              - Двухуровневая индексация (сессии + сообщения + задачи)
      • ingest-telegram    - Прямая загрузка чатов в граф памяти
      • indexing-progress  - Управление прогрессом инкрементальной индексации
      • update-summaries   - Обновление markdown-отчетов без полной индексации
      • review-summaries   - Автоматическое ревью и исправление саммаризаций
      • rebuild-vector-db  - Пересоздание векторной базы данных из существующих артефактов
      • search             - Поиск по индексированным данным
      • insight-graph      - Построение графа знаний
      • stats              - Статистика системы
      • check              - Проверка системы
      • extract-messages   - Извлечение новых сообщений из input в chats
      • deduplicate        - Удаление дубликатов сообщений
      • stop-indexing      - Остановка всех процессов индексации
      
    Управление данными:
      • backup-database    - Создание резервной копии (SQLite + ChromaDB)
      • restore-database   - Восстановление из резервной копии
      • optimize-database  - Оптимизация SQLite (VACUUM, ANALYZE, REINDEX)
      • validate-database  - Проверка целостности данных
      
    Система важности:
      • calculate-importance    - Вычисление важности записи
      • prune-memory            - Автоматическая очистка неважных записей
      • update-importance-scores - Массовый пересчёт важности
    """
    # Настройка логирования
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)


@cli.command("ingest-telegram")
@click.option(
    "--chats-dir",
    default="chats",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Директория с экспортами Telegram",
)
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Путь к SQLite базе типизированной памяти",
)
@click.option(
    "--chat",
    "selected_chats",
    multiple=True,
    help="Имя чата для выборочной индексации (можно указать несколько)",
)
def ingest_telegram(chats_dir: Path, db_path: Path, selected_chats: tuple[str, ...]):
    """📚 Загрузка сообщений Telegram напрямую в граф памяти."""

    chosen = [chat for chat in selected_chats if chat] or None
    indexer = TelegramIndexer(chats_dir=str(chats_dir), selected_chats=chosen)
    graph: TypedGraphMemory | None = None

    try:
        indexer.prepare()
        graph = TypedGraphMemory(db_path=str(db_path))
        ingestor = MemoryIngestor(graph)
        ingest_stats = ingestor.ingest(indexer.iter_records())
        index_stats = indexer.finalize()
    except Exception as exc:
        raise click.ClickException(f"Не удалось выполнить индексацию: {exc}") from exc
    try:
        indexer.close()
    except Exception:  # pragma: no cover - best effort
        logger.debug("Не удалось корректно закрыть индексатор", exc_info=True)

    try:
        if graph is not None:
            graph.conn.close()
    except Exception:  # pragma: no cover - best effort
        logger.debug("Не удалось закрыть соединение с БД графа", exc_info=True)

    skipped = max(0, index_stats.records_indexed - ingest_stats.records_ingested)

    click.echo("")
    click.echo("📥 Индексация Telegram завершена")
    click.echo(f"• Чатов обработано: {index_stats.sources_processed}")
    click.echo(
        f"• Записей создано: {ingest_stats.records_ingested} "
        f"(вложения: {ingest_stats.attachments_ingested})"
    )
    if skipped:
        click.echo(f"• Пропущено из-за дубликатов: {skipped}")
    if index_stats.warnings:
        click.echo("")
        click.echo("⚠️  Предупреждения:")
        for warning in index_stats.warnings:
            click.echo(f"  - {warning}")


@cli.command()
@click.option(
    "--embedding-model", default="text-embedding-qwen3-embedding-0.6b", help="Модель для эмбеддингов"
)
def check(embedding_model):
    """🔧 Проверка системы и подключений"""

    async def _check():
        import chromadb

        from ..core.lmstudio_client import LMStudioEmbeddingClient
        from ..config import get_settings

        click.echo("🔧 Проверка системы...")

        # Проверяем LM Studio Server
        try:
            settings = get_settings()
            lmstudio_client = LMStudioEmbeddingClient(
                model_name=embedding_model or settings.lmstudio_model,
                llm_model_name=settings.lmstudio_llm_model,
                base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
            )
            async with lmstudio_client:
                available = await lmstudio_client.test_connection()
                if not available or not available.get("lmstudio_available", False):
                    click.echo("❌ LM Studio Server недоступен")
                    click.echo(f"Убедитесь, что LM Studio Server запущен на {settings.lmstudio_host}:{settings.lmstudio_port}")
                    return False

                if not available.get("model_available", False):
                    click.echo("❌ Модель для эмбеддингов не найдена")
                    click.echo(f"Убедитесь, что модель {embedding_model or settings.lmstudio_model} загружена в LM Studio Server")
                    return False

                click.echo("✅ Ollama доступен")
        except Exception as e:
            click.echo(f"❌ Ошибка при проверке Ollama: {e}")
            return False

        # Проверяем ChromaDB коллекции
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # Проверяем новые коллекции
            collections_status = []
            try:
                sessions_collection = chroma_client.get_collection("chat_sessions")
                click.echo(
                    f"✅ ChromaDB chat_sessions: {sessions_collection.count()} записей"
                )
                collections_status.append(True)
            except:
                click.echo("⚠️  ChromaDB коллекция chat_sessions не найдена")
                collections_status.append(False)

            try:
                messages_collection = chroma_client.get_collection("chat_messages")
                click.echo(
                    f"✅ ChromaDB chat_messages: {messages_collection.count()} записей"
                )
                collections_status.append(True)
            except:
                click.echo("⚠️  ChromaDB коллекция chat_messages не найдена")
                collections_status.append(False)

            try:
                tasks_collection = chroma_client.get_collection("chat_tasks")
                click.echo(f"✅ ChromaDB chat_tasks: {tasks_collection.count()} записей")
                collections_status.append(True)
            except:
                click.echo("⚠️  ChromaDB коллекция chat_tasks не найдена")
                collections_status.append(False)

            if not any(collections_status):
                click.echo(
                    "\n💡 Подсказка: Запустите 'memory_mcp index' для создания индексов"
                )

        except Exception as e:
            click.echo(f"❌ Ошибка при проверке ChromaDB: {e}")

        # Проверяем файлы
        chats_path = Path("chats")
        if chats_path.exists():
            json_files = list(chats_path.glob("**/*.json"))
            click.echo(f"✅ Найдено JSON файлов: {len(json_files)}")
        else:
            click.echo("❌ Директория chats не найдена")

        # Проверяем саммаризации
        summaries_path = Path("artifacts/reports")
        if summaries_path.exists():
            md_files = list(summaries_path.glob("**/*.md"))
            click.echo(f"✅ Найдено MD файлов: {len(md_files)}")
        else:
            click.echo(
                "⚠️  Директория artifacts/reports не найдена (будет создана при индексации)"
            )

        click.echo("\n🎉 Система готова к работе!")
        return True

    asyncio.run(_check())


@cli.command()
@click.option(
    "--scope",
    default="all",
    type=click.Choice(["all", "chat"]),
    help="Область индексации: all (все чаты) или chat (один чат)",
)
@click.option("--chat", help="Название чата для индексации (если scope=chat)")
@click.option("--force-full", is_flag=True, help="Полная пересборка индекса")
@click.option(
    "--recent-days", default=7, type=int, help="Пересаммаризировать последние N дней"
)
@click.option(
    "--no-quality-check",
    is_flag=True,
    help="Отключить проверку качества саммаризации (быстрее)",
)
@click.option(
    "--no-improvement",
    is_flag=True,
    help="Отключить автоматическое улучшение саммаризации",
)
@click.option(
    "--min-quality", default=90.0, type=float, help="Минимальный балл качества (0-100)"
)
@click.option(
    "--enable-clustering",
    is_flag=True,
    help="Включить кластеризацию сессий для группировки",
)
@click.option(
    "--clustering-threshold",
    default=0.8,
    type=float,
    help="Порог сходства для кластеризации (0.0-1.0)",
)
@click.option(
    "--min-cluster-size", default=2, type=int, help="Минимальный размер кластера сессий"
)
@click.option(
    "--max-messages-per-group",
    default=200,
    type=int,
    help="Максимальное количество сообщений в группе (больше = меньше сессий)",
)
@click.option(
    "--max-session-hours",
    default=12,
    type=int,
    help="Максимальная длительность сессии в часах (больше = меньше сессий)",
)
@click.option(
    "--gap-minutes",
    default=120,
    type=int,
    help="Максимальный разрыв между сообщениями в минутах (больше = меньше сессий)",
)
@click.option(
    "--enable-smart-aggregation",
    is_flag=True,
    help="Включить умную группировку с скользящими окнами (NOW/FRESH/RECENT/OLD)",
)
@click.option(
    "--aggregation-strategy",
    default="smart",
    type=click.Choice(["smart", "channel", "legacy"]),
    help="Стратегия группировки: smart (умная), channel (для каналов), legacy (старая)",
)
@click.option(
    "--now-window-hours",
    default=24,
    type=int,
    help="Размер NOW окна в часах (по умолчанию: 24)",
)
@click.option(
    "--fresh-window-days",
    default=14,
    type=int,
    help="Размер FRESH окна в днях (по умолчанию: 14)",
)
@click.option(
    "--recent-window-days",
    default=30,
    type=int,
    help="Размер RECENT окна в днях (по умолчанию: 30)",
)
@click.option(
    "--strategy-threshold",
    default=1000,
    type=int,
    help="Порог количества сообщений для перехода между стратегиями (по умолчанию: 1000)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Принудительно пересоздать существующие артефакты",
)
@click.option(
    "--embedding-model", 
    default="text-embedding-qwen3-embedding-0.6b", 
    help="Модель для эмбеддингов"
)
def index(
    scope,
    chat,
    force_full,
    recent_days,
    no_quality_check,
    no_improvement,
    min_quality,
    enable_clustering,
    clustering_threshold,
    min_cluster_size,
    max_messages_per_group,
    max_session_hours,
    gap_minutes,
    enable_smart_aggregation,
    aggregation_strategy,
    now_window_hours,
    fresh_window_days,
    recent_window_days,
    strategy_threshold,
    force,
    embedding_model,
):
    """📚 Двухуровневая индексация чатов (L1: сессии + саммари, L2: сообщения, L3: задачи)

    Продвинутая индексация с умной группировкой сообщений в сессии,
    извлечением сущностей и задач, созданием Markdown отчётов.
    """

    async def _index():
        click.echo("=" * 80)
        click.echo("🚀 Telegram Dump Manager - Двухуровневая индексация v2.0")
        click.echo("=" * 80)
        click.echo()

        if scope == "chat" and not chat:
            click.echo("❌ Для scope='chat' необходимо указать --chat")
            return

        click.echo("📦 Инициализация индексатора...")
        from ..core.lmstudio_client import LMStudioEmbeddingClient
        from ..config import get_settings
        
        settings = get_settings()
        embedding_client = LMStudioEmbeddingClient(
            model_name=embedding_model or settings.lmstudio_model,
            llm_model_name=settings.lmstudio_llm_model,
            base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
        )
        chroma_path = os.getenv("MEMORY_MCP_CHROMA_PATH") or settings.chroma_path
        indexer = TwoLevelIndexer(
            chroma_path=chroma_path,
            artifacts_path=settings.artifacts_path,
            embedding_client=embedding_client,
            enable_quality_check=not no_quality_check,
            enable_iterative_refinement=not no_improvement,
            min_quality_score=min_quality,
            enable_clustering=enable_clustering,
            clustering_threshold=clustering_threshold,
            min_cluster_size=min_cluster_size,
            max_messages_per_group=max_messages_per_group,
            max_session_hours=max_session_hours,
            gap_minutes=gap_minutes,
            enable_smart_aggregation=enable_smart_aggregation,
            aggregation_strategy=aggregation_strategy,
            now_window_hours=now_window_hours,
            fresh_window_days=fresh_window_days,
            recent_window_days=recent_window_days,
            strategy_threshold=strategy_threshold,
            force=force,
        )
        click.echo("✅ Индексатор готов")
        click.echo()

        click.echo("⚙️  Параметры индексации:")
        click.echo(f"   - Scope: {scope}")
        click.echo(f"   - Chat: {chat or 'все чаты'}")
        click.echo(f"   - Force full rebuild: {force_full}")
        click.echo(f"   - Force artifacts: {force}")
        click.echo(f"   - Recent days resummary: {recent_days}")
        click.echo()
        click.echo("🎯 Параметры качества саммаризации:")
        click.echo(
            f"   - Проверка качества: {'❌ Отключена' if no_quality_check else '✅ Включена'}"
        )
        click.echo(
            f"   - Автоулучшение: {'❌ Отключено' if no_improvement else '✅ Включено'}"
        )
        click.echo(
            f"   - Минимальный балл: {min_quality}/100 {'(строгий режим)' if min_quality >= 80 else '(стандартный режим)' if min_quality >= 60 else '(мягкий режим)'}"
        )
        click.echo()
        click.echo("🔗 Параметры кластеризации сессий:")
        click.echo(
            f"   - Кластеризация: {'✅ Включена' if enable_clustering else '❌ Отключена'}"
        )
        if enable_clustering:
            click.echo(f"   - Порог сходства: {clustering_threshold}")
            click.echo(f"   - Минимальный размер кластера: {min_cluster_size}")
        click.echo()
        click.echo("📊 Параметры группировки сессий:")
        click.echo(f"   - Максимум сообщений в группе: {max_messages_per_group}")
        click.echo(f"   - Максимальная длительность сессии: {max_session_hours} часов")
        click.echo(f"   - Максимальный разрыв между сообщениями: {gap_minutes} минут")
        click.echo()

        if enable_smart_aggregation:
            click.echo("🧠 Умная группировка с скользящими окнами:")
            click.echo(f"   - Стратегия: {aggregation_strategy}")
            click.echo(f"   - NOW окно: {now_window_hours} часов (сегодня)")
            click.echo(f"   - FRESH окно: {fresh_window_days} дней (детально)")
            click.echo(f"   - RECENT окно: {recent_window_days} дней (по неделям)")
            click.echo(f"   - OLD окно: >{recent_window_days} дней (по месяцам)")
            click.echo(f"   - Порог перехода стратегий: {strategy_threshold} сообщений")
            click.echo("   - Контекстная саммаризация для NOW окна")
            click.echo("   - Оптимизация запросов к Ollama")
        else:
            click.echo("📊 Классический алгоритм группировки:")
            click.echo("   - Оптимизированная группировка по дням")
            click.echo("   - 10-100 сообщений в группе")
            click.echo("   - Естественные разрывы в обсуждениях (>4 часов)")
            click.echo("   - Фильтрация пустых и сервисных сообщений")
            click.echo("   - Дедупликация последовательных похожих сообщений")
            click.echo("   - Объединение маленьких групп")
        click.echo()

        click.echo("🔄 Начало индексации...")
        click.echo()

        try:
            stats = await indexer.build_index(
                scope=scope, chat=chat, force_full=force_full, recent_days=recent_days
            )

            click.echo()
            click.echo("=" * 80)
            click.echo("✅ Индексация завершена успешно!")
            click.echo("=" * 80)
            click.echo()
            click.echo("📊 Статистика:")
            click.echo(f"   - Проиндексировано чатов: {len(stats['indexed_chats'])}")
            click.echo(f"   - Сессий (L1): {stats['sessions_indexed']}")
            click.echo(f"   - Сообщений (L2): {stats['messages_indexed']}")
            click.echo(f"   - Задач (L3): {stats['tasks_indexed']}")
            click.echo()

            if stats["indexed_chats"]:
                click.echo("📁 Проиндексированные чаты:")
                for chat_name in stats["indexed_chats"]:
                    click.echo(f"   - {chat_name}")
                click.echo()

            click.echo("📂 Результаты сохранены в:")
            click.echo("   - Markdown отчёты: ./artifacts/reports/")
            click.echo("   - Векторная база: ./chroma_db/")
            click.echo("   - Коллекции: chat_sessions, chat_messages, chat_tasks")
            click.echo()

        except Exception as e:
            click.echo()
            click.echo("=" * 80)
            click.echo("❌ Ошибка при индексации!")
            click.echo("=" * 80)
            click.echo(f"Ошибка: {e}")
            click.echo()
            import traceback

            traceback.print_exc()

    asyncio.run(_index())


@cli.command("set-instruction")
@click.option(
    "--chat", help="Название чата (как папка в chats/) для индивидуальной инструкции"
)
@click.option(
    "--mode",
    type=click.Choice(["group", "channel"]),
    help="Общая инструкция для всех чатов выбранного типа",
)
@click.option("--text", help="Текст инструкции прямо в аргументе")
@click.option(
    "--file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к файлу с инструкцией",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Удалить сохранённую инструкцию для указанного чата или типа",
)
def set_instruction(chat, mode, text, file, clear):
    """📝 Сохранить или удалить специальную инструкцию саммаризации."""
    target_count = sum(1 for value in (chat, mode) if value)
    if target_count != 1:
        raise click.UsageError(
            "Нужно указать ровно один из параметров: --chat или --mode"
        )

    manager = InstructionManager()

    if clear:
        if chat:
            manager.clear_chat_instruction(chat)
            click.echo(f"🗑️ Индивидуальная инструкция для '{chat}' удалена")
        else:
            manager.clear_mode_instruction(mode)
            click.echo(f"🗑️ Общая инструкция для типа '{mode}' очищена")
        return

    instruction_text = text or ""
    if file:
        instruction_text = file.read_text(encoding="utf-8")
    if not instruction_text.strip():
        raise click.UsageError(
            "Необходимо передать текст инструкции через --text или --file (или используйте --clear)."
        )

    if chat:
        manager.set_chat_instruction(chat, instruction_text)
        click.echo(f"✅ Сохранена индивидуальная инструкция для чата '{chat}'")
    else:
        manager.set_mode_instruction(mode, instruction_text)
        click.echo(f"✅ Сохранена общая инструкция для типа '{mode}'")


@cli.command("list-instructions")
def list_instructions():
    """📋 Показать сохранённые инструкции саммаризации."""
    manager = InstructionManager()
    data = manager.export()

    click.echo("📌 Индивидуальные инструкции по чатам:")
    if data["chats"]:
        for name, instruction in sorted(data["chats"].items()):
            preview = instruction.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            click.echo(f"  • {name}: {preview}")
    else:
        click.echo("  (Нет индивидуальных инструкций)")

    click.echo("\n📌 Инструкции по типам чатов:")
    for mode in ("group", "channel"):
        instruction = data["modes"].get(mode, "").strip()
        if instruction:
            preview = instruction.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            click.echo(f"  • {mode}: {preview}")
        else:
            click.echo(f"  • {mode}: (не задано)")


def highlight_text(text: str, query: str) -> str:
    """Подсветка найденных терминов в тексте."""
    keywords = [
        word.strip().lower() for word in query.split() if len(word.strip()) >= 3
    ]

    if not keywords:
        return text

    result = text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(
            lambda m: click.style(m.group(0), fg="yellow", bold=True), result
        )

    return result


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
MIN_TOKEN_LENGTH = 3

HYBRID_WEIGHTS = {
    "messages": (0.65, 0.35),
    "sessions": (0.6, 0.4),
    "tasks": (0.6, 0.4),
}

RELEVANCE_THRESHOLDS = {
    "messages": 0.32,
    "sessions": 0.30,
    "tasks": 0.28,
}


def _tokenize(text: str) -> list[str]:
    """Токенизация для русского языка с fallback на простую."""
    if not text:
        return []

    try:
        return enhanced_tokenize(text)
    except Exception as e:
        logger.warning(f"Ошибка улучшенной токенизации, используем fallback: {e}")
        return [
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if len(token) >= MIN_TOKEN_LENGTH
        ]


def _bm25_scores(
    query_tokens: list[str], documents_tokens: list[list[str]]
) -> list[float]:
    """Вычисляет BM25 для корпуса документов."""
    if not query_tokens or not documents_tokens:
        return [0.0] * len(documents_tokens)

    num_docs = len(documents_tokens)
    doc_freq = Counter()
    doc_lengths = []
    for tokens in documents_tokens:
        unique_tokens = set(tokens)
        if unique_tokens:
            doc_freq.update(unique_tokens)
        doc_lengths.append(len(tokens))

    avgdl = sum(doc_lengths) / num_docs if num_docs else 0
    if avgdl == 0:
        return [0.0] * len(documents_tokens)

    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log(((num_docs - freq + 0.5) / (freq + 0.5)) + 1.0)

    scores = []
    k1, b = 1.5, 0.75  # Параметры BM25
    for tokens, doc_len in zip(documents_tokens, doc_lengths):
        if not tokens:
            scores.append(0.0)
            continue

        term_freq = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            token_idf = idf.get(token)
            tf = term_freq.get(token)
            if not token_idf or not tf:
                continue
            denom = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += token_idf * (tf * (k1 + 1) / denom)

        scores.append(score)

    return scores


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Лимит результатов")
@click.option(
    "--collection",
    "-c",
    type=click.Choice(["messages", "sessions", "tasks"]),
    default="messages",
    help="Коллекция для поиска",
)
@click.option("--chat", help="Фильтр по чату (название чата)")
@click.option(
    "--highlight/--no-highlight", default=True, help="Подсветка найденных терминов"
)
@click.option(
    "--embedding-model", 
    default="text-embedding-qwen3-embedding-0.6b", 
    help="Модель для эмбеддингов"
)
def search(query, limit, collection, chat, highlight, embedding_model):
    """🔍 Поиск по индексированным данным

    Поиск по трём уровням:
    - messages: Поиск по сообщениям
    - sessions: Поиск по саммаризациям сессий
    - tasks: Поиск по задачам (Action Items)
    """

    async def _search():
        import chromadb

        from ..core.lmstudio_client import LMStudioEmbeddingClient
        from ..config import get_settings

        click.echo(f"🔍 Поиск в коллекции '{collection}': '{query}'")
        if chat:
            click.echo(f"📋 Фильтр по чату: '{chat}'")

        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            settings = get_settings()
            embedding_client = LMStudioEmbeddingClient(
                model_name=embedding_model or settings.lmstudio_model,
                base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
            )

            collection_name = f"chat_{collection}"
            try:
                coll = chroma_client.get_collection(collection_name)
            except:
                click.echo(f"❌ Коллекция {collection_name} не найдена")
                click.echo("💡 Запустите 'memory_mcp index' для создания индексов")
                return

            async with embedding_client:
                query_embedding = await embedding_client._generate_single_embedding(query)

                if not query_embedding:
                    click.echo("❌ Не удалось сгенерировать эмбеддинг для запроса")
                    return

                where_filter = {"chat": chat} if chat else None
                vector_limit = max(limit * 4, 20)
                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=vector_limit,
                    where=where_filter,
                )

                documents = results.get("documents")
                if not documents or not documents[0]:
                    click.echo("❌ Результаты не найдены")
                    return

                raw_ids = results.get("ids") or [[]]
                raw_ids = raw_ids[0] if raw_ids else []
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                def resolve_doc_id(raw_id, metadata, doc_text):
                    if raw_id:
                        return raw_id
                    metadata = metadata or {}
                    for key in ("msg_id", "session_id", "task_id", "id"):
                        value = metadata.get(key)
                        if value:
                            return value
                    return f"doc-{abs(hash((doc_text or '')[:80]))}"

                vector_scores: dict[str, float] = {}
                vector_distances: dict[str, float] = {}
                vector_candidates = []

                for doc, metadata, distance, raw_id in zip(
                    documents[0], metadatas, distances, raw_ids
                ):
                    if not doc:
                        continue
                    doc_id = resolve_doc_id(raw_id, metadata, doc)
                    vector_candidates.append(
                        {
                            "id": doc_id,
                            "doc": doc,
                            "metadata": metadata or {},
                            "distance": distance,
                        }
                    )
                    vector_distances[doc_id] = distance

                if not vector_candidates:
                    click.echo("❌ Результаты не найдены")
                    return

                available_distances = [
                    item["distance"]
                    for item in vector_candidates
                    if item.get("distance") is not None
                ]
                if available_distances:
                    min_distance = min(available_distances)
                    max_distance = max(available_distances)
                    denominator = max(max_distance - min_distance, 1e-6)
                    for item in vector_candidates:
                        doc_id = item["id"]
                        distance = item.get("distance")
                        if distance is None:
                            continue
                        if max_distance == min_distance:
                            vector_scores[doc_id] = 1.0
                        else:
                            score = (max_distance - distance) / denominator
                            vector_scores[doc_id] = max(score, 0.0)

                get_kwargs = {"include": ["documents", "metadatas"]}
                if where_filter:
                    get_kwargs["where"] = where_filter

                corpus = coll.get(**get_kwargs)
                corpus_docs = corpus.get("documents", [])
                corpus_meta = corpus.get("metadatas", [])

                doc_store: dict[str, dict[str, object]] = {}
                lexical_entries: list[str] = []
                lexical_tokens: list[list[str]] = []

                for idx, (doc_text, metadata) in enumerate(
                    zip(corpus_docs, corpus_meta)
                ):
                    raw_id = f"doc_{idx}_{hash(doc_text or '')}"
                    resolved_id = resolve_doc_id(raw_id, metadata, doc_text)
                    doc_store[resolved_id] = {
                        "doc": doc_text or "",
                        "metadata": metadata or {},
                    }
                    lexical_entries.append(resolved_id)
                    lexical_tokens.append(_tokenize(doc_text or ""))

                query_tokens = _tokenize(query)
                lexical_scores_list = _bm25_scores(query_tokens, lexical_tokens)
                lexical_scores = dict(zip(lexical_entries, lexical_scores_list))
                max_lexical_score = (
                    max(lexical_scores.values()) if lexical_scores else 0.0
                )
                lexical_norm = {
                    doc_id: (score / max_lexical_score)
                    if max_lexical_score > 0
                    else 0.0
                    for doc_id, score in lexical_scores.items()
                }

                weight_vector, weight_lexical = HYBRID_WEIGHTS.get(
                    collection, (0.6, 0.4)
                )
                if not query_tokens or max_lexical_score == 0:
                    weight_vector, weight_lexical = 1.0, 0.0
                weight_sum = weight_vector + weight_lexical
                if weight_sum == 0:
                    weight_vector, weight_lexical = 1.0, 0.0
                else:
                    weight_vector /= weight_sum
                    weight_lexical /= weight_sum

                candidate_ids = set(vector_scores.keys())
                if lexical_scores:
                    sorted_lexical = sorted(
                        lexical_scores.items(), key=lambda item: item[1], reverse=True
                    )
                    top_lexical = [
                        doc_id for doc_id, score in sorted_lexical if score > 0
                    ][: max(limit * 3, 15)]
                    candidate_ids.update(top_lexical)

                final_candidates = []
                for doc_id in candidate_ids:
                    payload = doc_store.get(doc_id)
                    if not payload:
                        continue
                    vector_component = vector_scores.get(doc_id, 0.0)
                    lexical_component = lexical_norm.get(doc_id, 0.0)
                    hybrid_score = (
                        vector_component * weight_vector
                        + lexical_component * weight_lexical
                    )
                    final_candidates.append(
                        {
                            "id": doc_id,
                            "doc": payload["doc"],
                            "metadata": payload["metadata"],
                            "score": hybrid_score,
                            "vector_component": vector_component,
                            "lexical_component": lexical_component,
                            "vector_distance": vector_distances.get(doc_id),
                        }
                    )

                if not final_candidates:
                    click.echo("❌ Результаты не найдены")
                    return

                final_candidates.sort(key=lambda item: item["score"], reverse=True)

                threshold = RELEVANCE_THRESHOLDS.get(collection, 0.0)
                filtered_candidates = [
                    candidate
                    for candidate in final_candidates
                    if candidate["score"] >= threshold
                ]
                filtered_out = len(final_candidates) - len(filtered_candidates)

                if not filtered_candidates:
                    filtered_candidates = final_candidates[:limit]
                    filtered_out = 0
                else:
                    filtered_candidates = filtered_candidates[:limit]

                click.echo(f"✅ Найдено результатов: {len(filtered_candidates)}")
                if filtered_out > 0:
                    click.echo(f"   (отсечено по порогу релевантности: {filtered_out})")
                click.echo()

                for index, candidate in enumerate(filtered_candidates, 1):
                    metadata = candidate.get("metadata") or {}
                    chat_name = metadata.get(
                        "chat", metadata.get("chat_name", "Unknown")
                    )
                    signal_parts = []
                    if candidate.get("vector_component", 0) > 0:
                        signal_parts.append("vec")
                    if candidate.get("lexical_component", 0) > 0:
                        signal_parts.append("lex")
                    signals = "+".join(signal_parts) if signal_parts else "-"
                    header = f"{index}. {chat_name} (score: {candidate['score'] * 100:.1f} | signals: {signals}"
                    distance = candidate.get("vector_distance")
                    if distance is not None:
                        header += f" | distance: {distance:.1f}"
                    header += ")"
                    click.echo(header)

                    doc_text = candidate.get("doc") or ""

                    if collection == "messages":
                        text = (
                            doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                        )
                        if highlight:
                            text = highlight_text(text, query)
                        click.echo(f"   {text}")
                    elif collection == "sessions":
                        session_id = metadata.get("session_id", "N/A")
                        time_range = metadata.get("time_span", "N/A")
                        click.echo(f"   Session: {session_id}")
                        click.echo(f"   Time: {time_range}")
                        summary = (
                            doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
                        )
                        if highlight:
                            summary = highlight_text(summary, query)
                        click.echo(f"   Summary: {summary}")
                    elif collection == "tasks":
                        task_text = (
                            doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                        )
                        if highlight:
                            task_text = highlight_text(task_text, query)
                        owner = metadata.get("owner", "N/A")
                        due_date = metadata.get("due", "N/A")
                        priority = metadata.get("priority", "N/A")
                        click.echo(f"   Task: {task_text}")
                        click.echo(
                            f"   Owner: {owner} | Due: {due_date} | Priority: {priority}"
                        )

                    click.echo()

        except Exception as e:
            click.echo(f"❌ Ошибка при поиске: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(_search())


@cli.command()
@click.option(
    "--threshold", default=0.76, type=float, help="Порог схожести между чатами"
)
@click.option("--graphml", type=click.Path(), help="Путь для сохранения GraphML-файла")
def insight_graph(threshold, graphml):
    """🧠 Построение графа знаний

    Создает граф связей на основе саммаризаций, выделяя ключевые инсайты
    и связи между чатами.
    """

    async def _run():
        click.echo("🧠 Построение графа инсайтов...")
        click.echo(f"   Порог схожести: {threshold}")
        click.echo()

        analyzer = SummaryInsightAnalyzer(
            summaries_dir=Path("artifacts/reports"),
            similarity_threshold=threshold,
        )

        try:
            # Строим граф
            async with analyzer:
                result = await analyzer.analyze()

            # Выводим отчёт
            click.echo("\n" + "=" * 80)
            click.echo("✅ Граф инсайтов построен!")
            click.echo("=" * 80)
            click.echo()

            graph_metrics = result.metrics.get("graph", {})
            click.echo("📊 Метрики графа:")
            click.echo(f"   - Узлов (чатов): {graph_metrics.get('nodes', 0)}")
            click.echo(f"   - Рёбер (связей): {graph_metrics.get('edges', 0)}")
            click.echo(f"   - Компонентов: {graph_metrics.get('components', 0)}")
            click.echo(f"   - Плотность графа: {graph_metrics.get('density', 0.0):.3f}")
            click.echo()

            # Сохраняем отчёт
            report_path = Path("insight_graph_report.md")
            report_content = analyzer.generate_report(result)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            click.echo(f"📄 Отчёт сохранён: {report_path}")

            # Сохраняем GraphML если указан путь
            if graphml:
                export_path = analyzer.export_graphml(result, Path(graphml))
                if export_path:
                    click.echo(f"📁 GraphML сохранён: {export_path}")

        except Exception as e:
            click.echo(f"❌ Ошибка при построении графа: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(_run())


@cli.command()
def stats():
    """📊 Статистика системы"""

    async def _stats():
        import chromadb

        click.echo("📊 Статистика системы...")
        click.echo()

        # Проверяем ChromaDB коллекции
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # Статистика по коллекциям
            total_records = 0
            for coll_name in ["chat_sessions", "chat_messages", "chat_tasks"]:
                try:
                    coll = chroma_client.get_collection(coll_name)
                    count = coll.count()
                    total_records += count
                    icon = "✅" if count > 0 else "⚠️ "
                    click.echo(f"{icon} {coll_name}: {count} записей")
                except:
                    click.echo(f"❌ {coll_name}: не найдена")

            click.echo()
            click.echo(f"📦 Всего записей в индексах: {total_records}")

        except Exception as e:
            click.echo(f"❌ Ошибка при проверке ChromaDB: {e}")

        click.echo()

        # Статистика по файлам
        chats_path = Path("chats")
        if chats_path.exists():
            json_files = list(chats_path.glob("**/*.json"))
            click.echo(f"📁 JSON файлов: {len(json_files)}")

            # Количество чатов
            chat_dirs = [d for d in chats_path.iterdir() if d.is_dir()]
            click.echo(f"💬 Чатов: {len(chat_dirs)}")
        else:
            click.echo("📁 JSON файлов: 0")

        # Markdown файлы
        summaries_path = Path("artifacts/reports")
        if summaries_path.exists():
            md_files = list(summaries_path.glob("**/*.md"))
            session_files = list(summaries_path.glob("**/sessions/*.md"))
            click.echo(f"📄 MD файлов: {len(md_files)}")
            click.echo(f"📝 Саммаризаций сессий: {len(session_files)}")
        else:
            click.echo("📄 MD файлов: 0")

    asyncio.run(_stats())


@cli.command("indexing-progress")
@click.option("--chat", help="Показать прогресс для конкретного чата")
@click.option(
    "--reset",
    is_flag=True,
    help="Сбросить прогресс индексации (для повторной полной индексации)",
)
def indexing_progress(chat, reset):
    """🔄 Управление прогрессом инкрементальной индексации

    Показывает информацию о последней индексации каждого чата
    или сбрасывает прогресс для повторной индексации.
    """

    import chromadb

    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        try:
            progress_collection = chroma_client.get_collection("indexing_progress")
        except:
            click.echo("⚠️  Коллекция indexing_progress не найдена")
            click.echo("💡 Индексация ещё не запускалась или используется старая версия")
            return

        if reset:
            if chat:
                # Сбрасываем прогресс для конкретного чата
                from ..utils.naming import slugify

                progress_id = f"progress_{slugify(chat)}"
                try:
                    progress_collection.delete(ids=[progress_id])
                    click.echo(f"✅ Прогресс индексации для чата '{chat}' сброшен")
                    click.echo(
                        "💡 При следующем запуске чат будет проиндексирован заново"
                    )
                except Exception as e:
                    click.echo(f"❌ Ошибка при сбросе прогресса: {e}")
            else:
                # Сбрасываем весь прогресс
                try:
                    result = progress_collection.get()
                    if result["ids"]:
                        progress_collection.delete(ids=result["ids"])
                        click.echo(
                            f"✅ Прогресс индексации сброшен для {len(result['ids'])} чатов"
                        )
                        click.echo(
                            "💡 При следующем запуске все чаты будут проиндексированы заново"
                        )
                    else:
                        click.echo("⚠️  Нет записей о прогрессе индексации")
                except Exception as e:
                    click.echo(f"❌ Ошибка при сбросе прогресса: {e}")
        else:
            # Показываем прогресс
            click.echo("🔄 Прогресс инкрементальной индексации:")
            click.echo()

            try:
                if chat:
                    # Показываем прогресс для конкретного чата
                    from ..utils.naming import slugify

                    progress_id = f"progress_{slugify(chat)}"
                    result = progress_collection.get(
                        ids=[progress_id], include=["metadatas"]
                    )

                    if result["ids"]:
                        metadata = result["metadatas"][0]
                        click.echo(f"📋 Чат: {metadata.get('chat_name', chat)}")
                        click.echo(
                            f"   Последнее сообщение: {metadata.get('last_indexed_date', 'N/A')}"
                        )
                        click.echo(
                            f"   Последняя индексация: {metadata.get('last_indexing_time', 'N/A')}"
                        )
                        click.echo(
                            f"   Всего сообщений: {metadata.get('total_messages', 0)}"
                        )
                        click.echo(
                            f"   Всего сессий: {metadata.get('total_sessions', 0)}"
                        )
                    else:
                        click.echo(f"⚠️  Нет записей о прогрессе для чата '{chat}'")
                else:
                    # Показываем прогресс для всех чатов
                    result = progress_collection.get(include=["metadatas"])

                    if result["ids"]:
                        click.echo(f"Найдено записей: {len(result['ids'])}")
                        click.echo()

                        for i, metadata in enumerate(result["metadatas"], 1):
                            chat_name = metadata.get("chat_name", "Unknown")
                            last_date = metadata.get("last_indexed_date", "N/A")
                            last_time = metadata.get("last_indexing_time", "N/A")
                            total_msgs = metadata.get("total_messages", 0)
                            total_sessions = metadata.get("total_sessions", 0)

                            click.echo(f"{i}. {chat_name}")
                            click.echo(f"   Последнее сообщение: {last_date}")
                            click.echo(f"   Последняя индексация: {last_time}")
                            click.echo(
                                f"   Сообщений: {total_msgs}, Сессий: {total_sessions}"
                            )
                            click.echo()
                    else:
                        click.echo("⚠️  Нет записей о прогрессе индексации")
                        click.echo("💡 Запустите индексацию командой: memory_mcp index")
            except Exception as e:
                click.echo(f"❌ Ошибка при получении прогресса: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        click.echo(f"❌ Ошибка при подключении к ChromaDB: {e}")


@cli.command("update-summaries")
@click.option("--chat", help="Обновить отчеты только для конкретного чата")
@click.option(
    "--force",
    is_flag=True,
    help="Принудительно пересоздать существующие артефакты",
)
def update_summaries(chat, force):
    """📝 Обновление markdown-отчетов без полной индексации

    Читает существующие JSON-саммаризации и пересоздает markdown-отчеты,
    включая раздел "Актуально за 30 дней".
    """
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from ..analysis.markdown_renderer import MarkdownRenderer

    async def _update_summaries():
        click.echo("📝 Обновление markdown-отчетов...")
        click.echo()

        reports_dir = Path("artifacts/reports")

        if not reports_dir.exists():
            click.echo("❌ Директория artifacts/reports не найдена")
            click.echo("💡 Запустите индексацию: memory_mcp index")
            return

        # Находим чаты для обработки
        if chat:
            chat_dirs = [reports_dir / chat] if (reports_dir / chat).exists() else []
            if not chat_dirs:
                click.echo(f"❌ Чат '{chat}' не найден в artifacts/reports/")
                return
        else:
            chat_dirs = [
                d
                for d in reports_dir.iterdir()
                if d.is_dir() and (d / "sessions").exists()
            ]

        if not chat_dirs:
            click.echo("❌ Не найдено чатов с саммаризациями")
            return

        click.echo(f"📁 Найдено чатов: {len(chat_dirs)}")
        click.echo()

        # Создаем renderer
        renderer = MarkdownRenderer(output_dir=reports_dir)

        def parse_message_time(date_str: str) -> datetime:
            try:
                from ..utils.datetime_utils import parse_datetime_utc

                return parse_datetime_utc(date_str, default=datetime.now(ZoneInfo("UTC")), use_zoneinfo=True)
            except Exception:
                return datetime.now(ZoneInfo("UTC"))

        def load_session_summaries(chat_dir: Path) -> list:
            sessions = []
            sessions_dir = chat_dir / "sessions"
            if not sessions_dir.exists():
                return sessions

            json_files = list(sessions_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        session = json.load(f)
                        sessions.append(session)
                except Exception as e:
                    click.echo(f"⚠️  Ошибка чтения {json_file.name}: {e}")
                    continue
            return sessions

        # Обрабатываем каждый чат
        updated = 0

        for chat_dir in chat_dirs:
            chat_name = chat_dir.name.replace("_", " ").title()
            click.echo(f"📋 Обработка чата: {chat_name}")

            # Загружаем саммаризации
            sessions = load_session_summaries(chat_dir)

            if not sessions:
                click.echo("   ⚠️  Нет саммаризаций для обновления")
                continue

            click.echo(f"   📊 Найдено саммаризаций: {len(sessions)}")

            # Фильтруем сессии за последние 30 дней
            now = datetime.now(ZoneInfo("UTC"))
            thirty_days_ago = now - timedelta(days=30)

            recent_sessions = []
            for session in sessions:
                end_time_str = session.get("meta", {}).get("end_time_utc", "")
                if end_time_str:
                    end_time = parse_message_time(end_time_str)
                    if end_time >= thirty_days_ago:
                        recent_sessions.append(session)

            click.echo(f"   📅 Сессий за последние 30 дней: {len(recent_sessions)}")

            # Сортируем по качеству
            top_sessions = sorted(
                recent_sessions,
                key=lambda s: s.get("quality", {}).get("score", 0),
                reverse=True,
            )

            # Генерируем отчеты
            try:
                renderer.render_chat_summary(
                    chat_name, sessions, top_sessions=top_sessions, force=force
                )
                renderer.render_cumulative_context(chat_name, sessions, force=force)
                renderer.render_chat_index(chat_name, sessions, force=force)
                click.echo("   ✅ Обновлены все отчеты")
                updated += 1
            except Exception as e:
                click.echo(f"   ❌ Ошибка при обновлении: {e}")

        # Итоговая статистика
        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Обновление завершено!")
        click.echo("=" * 80)
        click.echo(f"📊 Обновлено чатов: {updated}")
        click.echo("📂 Обновленные файлы находятся в: ./artifacts/reports/")

    asyncio.run(_update_summaries())


@cli.command("rebuild-vector-db")
@click.option(
    "--force",
    is_flag=True,
    help="Принудительно удалить существующую базу данных без подтверждения",
)
@click.option(
    "--keep-reports",
    is_flag=True,
    help="Сохранить markdown отчеты и JSON саммаризации (только пересоздать ChromaDB)",
)
@click.option(
    "--backup",
    is_flag=True,
    help="Создать резервную копию существующей базы данных перед удалением",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Отключить прогресс-бар (полезно для автоматизации)",
)
def rebuild_vector_db(force, keep_reports, backup, no_progress):
    """🔄 Пересоздание векторной базы данных ChromaDB

    Удаляет существующую векторную базу данных и пересоздает её заново,
    используя существующие артефакты (JSON саммаризации, markdown отчеты).

    Полезно когда:
    - База данных повреждена
    - Нужно обновить схему коллекций
    - Произошла ошибка при индексации

    ВНИМАНИЕ: Эта команда удалит все данные из ChromaDB!
    """

    async def _rebuild():
        import json
        import shutil
        from pathlib import Path

        click.echo("=" * 80)
        click.echo("🔄 Пересоздание векторной базы данных ChromaDB")
        click.echo("=" * 80)
        click.echo()

        # Проверяем наличие артефактов
        reports_dir = Path("artifacts/reports")
        chroma_dir = Path("chroma_db")

        if not reports_dir.exists():
            click.echo("❌ Директория artifacts/reports не найдена")
            click.echo("💡 Сначала запустите индексацию: memory_mcp index")
            return

        # Проверяем наличие JSON саммаризаций
        json_files = list(reports_dir.glob("**/*.json"))
        if not json_files:
            click.echo("❌ Не найдено JSON файлов саммаризаций")
            click.echo("💡 Сначала запустите индексацию: memory_mcp index")
            return

        click.echo(f"📁 Найдено JSON файлов саммаризаций: {len(json_files)}")

        # Проверяем существующую базу данных
        if chroma_dir.exists():
            try:
                import chromadb

                chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

                # Получаем информацию о коллекциях
                collections_info = []
                for collection_name in [
                    "chat_sessions",
                    "chat_messages",
                    "chat_tasks",
                    "session_clusters",
                    "indexing_progress",
                ]:
                    try:
                        collection = chroma_client.get_collection(collection_name)
                        count = collection.count()
                        collections_info.append(
                            f"   - {collection_name}: {count} записей"
                        )
                    except:
                        collections_info.append(f"   - {collection_name}: не найдена")

                click.echo("📊 Текущее состояние ChromaDB:")
                for info in collections_info:
                    click.echo(info)
                click.echo()

            except Exception as e:
                click.echo(f"⚠️  Не удалось подключиться к ChromaDB: {e}")
                click.echo("   База данных может быть повреждена")
                click.echo()

        # Подтверждение удаления
        if not force:
            click.echo("⚠️  ВНИМАНИЕ: Эта операция удалит все данные из ChromaDB!")
            click.echo("   Существующие коллекции будут полностью пересозданы.")
            click.echo()

            if not click.confirm("Продолжить?"):
                click.echo("❌ Операция отменена")
                return

        # Создаем резервную копию если запрошено
        if backup and chroma_dir.exists():
            backup_dir = Path(
                f"chroma_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            click.echo(f"📦 Создание резервной копии: {backup_dir}")
            try:
                shutil.copytree(chroma_dir, backup_dir)
                click.echo(f"✅ Резервная копия создана: {backup_dir}")
            except Exception as e:
                click.echo(f"❌ Ошибка создания резервной копии: {e}")
                if not click.confirm("Продолжить без резервной копии?"):
                    return
            click.echo()

        # Удаляем существующую базу данных
        if chroma_dir.exists():
            click.echo("🗑️  Удаление существующей ChromaDB...")
            try:
                shutil.rmtree(chroma_dir)
                click.echo("✅ Существующая база данных удалена")
            except Exception as e:
                click.echo(f"❌ Ошибка удаления базы данных: {e}")
                return
            click.echo()

        # Пересоздаем базу данных из существующих артефактов
        click.echo("🔄 Пересоздание векторной базы из существующих артефактов...")
        click.echo()

        try:
            # Инициализируем индексатор
            from ..core.indexer import TwoLevelIndexer

            click.echo("📦 Инициализация индексатора...")
            indexer = TwoLevelIndexer()
            click.echo("✅ Индексатор готов")
            click.echo()

            # Загружаем существующие саммаризации
            click.echo("📚 Загрузка существующих саммаризаций...")

            sessions_data = []
            for json_file in json_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        session_data = json.load(f)
                        sessions_data.append(session_data)
                except Exception as e:
                    click.echo(f"⚠️  Ошибка чтения {json_file.name}: {e}")
                    continue

            click.echo(f"✅ Загружено саммаризаций: {len(sessions_data)}")
            click.echo()

            if not sessions_data:
                click.echo("❌ Нет валидных саммаризаций для пересоздания базы")
                return

            # Пересоздаем коллекции
            click.echo("🔄 Пересоздание коллекций ChromaDB...")

            # Группируем по чатам
            chats_data = {}
            for session in sessions_data:
                chat_name = session.get("meta", {}).get("chat_name", "Unknown")
                if chat_name not in chats_data:
                    chats_data[chat_name] = []
                chats_data[chat_name].append(session)

            click.echo(f"📋 Найдено чатов: {len(chats_data)}")

            # Индексируем каждую сессию с прогресс-баром
            total_sessions = len(sessions_data)
            indexed_sessions = 0
            indexed_messages = 0
            indexed_tasks = 0

            # Импортируем tqdm для прогресс-бара
            from tqdm import tqdm

            # Определяем, показывать ли прогресс-бар
            show_progress = not no_progress

            if show_progress:
                # Создаем прогресс-бар для всех сессий
                with tqdm(
                    total=total_sessions,
                    desc="Пересоздание векторной базы",
                    unit="сессия",
                ) as pbar:
                    for chat_name, chat_sessions in chats_data.items():
                        # Обновляем описание прогресс-бара
                        pbar.set_description(f"Обработка чата: {chat_name}")

                        for session in chat_sessions:
                            try:
                                # L1: Индексация саммари сессии
                                await indexer._index_session_l1(session)
                                indexed_sessions += 1

                                # L2: Индексация сообщений
                                messages_count = await indexer._index_messages_l2(
                                    session
                                )
                                indexed_messages += messages_count

                                # L3: Индексация задач
                                tasks_count = await indexer._index_tasks(session)
                                indexed_tasks += tasks_count

                            except Exception as e:
                                click.echo(
                                    f"⚠️  Ошибка индексации сессии {session.get('session_id', 'Unknown')}: {e}"
                                )
                                continue

                            # Обновляем прогресс-бар с дополнительной информацией
                            pbar.set_postfix(
                                {
                                    "сессий": indexed_sessions,
                                    "сообщений": indexed_messages,
                                    "задач": indexed_tasks,
                                }
                            )
                            pbar.update(1)
            else:
                # Обработка без прогресс-бара
                for chat_name, chat_sessions in chats_data.items():
                    click.echo(
                        f"📁 Обработка чата: {chat_name} ({len(chat_sessions)} сессий)"
                    )

                    for session in chat_sessions:
                        try:
                            # L1: Индексация саммари сессии
                            await indexer._index_session_l1(session)
                            indexed_sessions += 1

                            # L2: Индексация сообщений
                            messages_count = await indexer._index_messages_l2(session)
                            indexed_messages += messages_count

                            # L3: Индексация задач
                            tasks_count = await indexer._index_tasks(session)
                            indexed_tasks += tasks_count

                        except Exception as e:
                            click.echo(
                                f"⚠️  Ошибка индексации сессии {session.get('session_id', 'Unknown')}: {e}"
                            )
                            continue

                    click.echo(f"   ✅ Обработано сессий: {len(chat_sessions)}")

            click.echo()
            click.echo("=" * 80)
            click.echo("✅ Векторная база данных успешно пересоздана!")
            click.echo("=" * 80)
            click.echo()
            click.echo("📊 Статистика:")
            click.echo(f"   - Пересоздано сессий (L1): {indexed_sessions}")
            click.echo(f"   - Пересоздано сообщений (L2): {indexed_messages}")
            click.echo(f"   - Пересоздано задач (L3): {indexed_tasks}")
            click.echo()
            click.echo("📂 Результаты:")
            click.echo("   - Векторная база: ./chroma_db/")
            click.echo("   - Коллекции: chat_sessions, chat_messages, chat_tasks")
            if keep_reports:
                click.echo("   - Markdown отчеты: сохранены в ./artifacts/reports/")
            click.echo()
            click.echo("💡 Теперь можно использовать поиск: memory_mcp search")

        except Exception as e:
            click.echo()
            click.echo("=" * 80)
            click.echo("❌ Ошибка при пересоздании векторной базы!")
            click.echo("=" * 80)
            click.echo(f"Ошибка: {e}")
            click.echo()
            import traceback

            traceback.print_exc()

    asyncio.run(_rebuild())


@cli.command("extract-messages")
@click.option("--dry-run", is_flag=True, help="Только анализ, без изменения файлов")
@click.option("--no-date-filter", is_flag=True, help="Отключить фильтрацию по дате")
@click.option("--chat", help="Фильтр по названию чата")
@click.option("--input-dir", default="input", help="Директория с исходными данными")
@click.option(
    "--chats-dir", default="chats", help="Директория для сохранения сообщений"
)
def extract_messages(dry_run, no_date_filter, chat, input_dir, chats_dir):
    """📥 Извлечение новых сообщений из input в chats

    Извлекает новые сообщения из директории input и сохраняет их в chats,
    с фильтрацией по дате и дедупликацией.
    """

    async def _extract_messages():
        click.echo("📥 Извлечение новых сообщений...")
        click.echo(f"   Входная директория: {input_dir}")
        click.echo(f"   Выходная директория: {chats_dir}")
        click.echo(
            f"   Фильтр по дате: {'❌ Отключен' if no_date_filter else '✅ Включен'}"
        )
        click.echo(f"   Фильтр по чату: {chat or 'все чаты'}")
        click.echo(f"   Режим: {'🔸 DRY RUN' if dry_run else '✅ Реальное выполнение'}")
        click.echo()

        # Создаем экстрактор
        extractor = MessageExtractor(input_dir=input_dir, chats_dir=chats_dir)

        # Выполняем извлечение
        extractor.extract_all_messages(
            dry_run=dry_run, filter_by_date=not no_date_filter, chat_filter=chat
        )

        # Выводим статистику
        extractor.print_stats()

        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Извлечение сообщений завершено!")
        click.echo("=" * 80)

    asyncio.run(_extract_messages())


@cli.command("deduplicate")
@click.option(
    "--chats-dir", default="chats", help="Директория с сообщениями для дедупликации"
)
def deduplicate(chats_dir):
    """🧹 Удаление дубликатов сообщений

    Удаляет дубликаты сообщений по полю 'id' во всех чатах.
    """

    async def _deduplicate():
        click.echo("🧹 Удаление дубликатов сообщений...")
        click.echo(f"   Директория: {chats_dir}")
        click.echo()

        # Создаем дедупликатор
        deduplicator = MessageDeduplicator(chats_dir=chats_dir)

        # Выполняем дедупликацию
        deduplicator.deduplicate_all_chats()

        # Выводим статистику
        deduplicator.print_stats()

        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Дедупликация завершена!")
        click.echo("=" * 80)

    asyncio.run(_deduplicate())


@cli.command("sync-chromadb")
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Путь к SQLite базе типизированной памяти",
)
@click.option(
    "--chroma-path",
    default="chroma_db",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Путь к ChromaDB",
)
@click.option(
    "--chat",
    help="Синхронизировать только указанный чат",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Режим тестирования без изменений",
)
def sync_chromadb(db_path: Path, chroma_path: Path, chat: Optional[str], dry_run: bool):
    """Синхронизация записей из ChromaDB в граф памяти.
    
    Эта команда мигрирует существующие записи из ChromaDB коллекций
    (chat_messages, chat_sessions, chat_tasks) в граф памяти TypedGraphMemory.
    Эмбеддинги также синхронизируются.
    """
    import chromadb
    from ..memory.ingest import MemoryIngestor
    from ..indexing import MemoryRecord
    from ..utils.datetime_utils import parse_datetime_utc
    from datetime import datetime, timezone
    
    logger.info("🔄 Начало синхронизации ChromaDB → Граф памяти")
    
    if dry_run:
        logger.info("🔍 Режим тестирования (dry-run), изменения не будут сохранены")
    
    # Инициализация графа
    graph = TypedGraphMemory(db_path=str(db_path))
    ingestor = MemoryIngestor(graph)
    
    # Инициализация сервисов для эмбеддингов и Qdrant
    from ..memory.embeddings import build_embedding_service_from_env
    from ..memory.vector_store import build_vector_store_from_env
    
    embedding_service = build_embedding_service_from_env()
    vector_store = build_vector_store_from_env()
    
    if vector_store and embedding_service and embedding_service.dimension:
        vector_store.ensure_collection(embedding_service.dimension)
        logger.info("✅ Векторное хранилище инициализировано")
    else:
        logger.warning("⚠️  Векторное хранилище недоступно, эмбеддинги не будут сохранены в Qdrant")
    
    # Инициализация ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    
    total_synced = 0
    total_errors = 0
    
    collections_to_sync = ["chat_messages", "chat_sessions", "chat_tasks"]
    
    for collection_name in collections_to_sync:
        try:
            collection = chroma_client.get_collection(collection_name)
            total_count = collection.count()
            
            if total_count == 0:
                logger.info(f"  Коллекция {collection_name}: пуста, пропускаем")
                continue
            
            logger.info(f"  Коллекция {collection_name}: {total_count} записей")
            
            # Получаем все записи батчами
            offset = 0
            batch_size = 100
            synced_in_collection = 0
            
            while offset < total_count:
                try:
                    result = collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents", "metadatas", "embeddings"]
                    )
                    
                    ids = result.get("ids", [])
                    if not ids:
                        break
                    
                    documents = result.get("documents", [])
                    metadatas = result.get("metadatas", [])
                    embeddings = result.get("embeddings", [])
                    
                    records_to_ingest = []
                    
                    for idx, record_id in enumerate(ids):
                        try:
                            # Проверяем, существует ли уже запись в графе
                            if record_id in graph.graph:
                                continue
                            
                            # Фильтр по чату, если указан
                            metadata = metadatas[idx] if idx < len(metadatas) else {}
                            if chat and metadata.get("chat") != chat:
                                continue
                            
                            doc = documents[idx] if idx < len(documents) else ""
                            embedding = embeddings[idx] if idx < len(embeddings) else None
                            
                            # Парсим timestamp
                            date_utc = metadata.get("date_utc") or metadata.get("start_time_utc") or metadata.get("end_time_utc")
                            timestamp = None
                            if date_utc:
                                try:
                                    timestamp = parse_datetime_utc(date_utc, use_zoneinfo=True)
                                except Exception:
                                    timestamp = datetime.now(timezone.utc)
                            else:
                                timestamp = datetime.now(timezone.utc)
                            
                            # Извлекаем автора
                            author = metadata.get("sender") or metadata.get("author") or metadata.get("username")
                            
                            # Извлекаем теги и сущности
                            tags = metadata.get("tags", [])
                            if isinstance(tags, str):
                                tags = [tags] if tags else []
                            
                            entities = metadata.get("entities", [])
                            if isinstance(entities, str):
                                entities = [entities] if entities else []
                            
                            # Создаём MemoryRecord
                            record = MemoryRecord(
                                record_id=record_id,
                                source=metadata.get("chat", collection_name.replace("chat_", "")),
                                content=doc,
                                timestamp=timestamp,
                                author=author,
                                tags=tags if isinstance(tags, list) else [],
                                entities=entities if isinstance(entities, list) else [],
                                attachments=[],
                                metadata={
                                    "collection": collection_name,
                                    "chat": metadata.get("chat", ""),
                                    **metadata,
                                },
                            )
                            
                            records_to_ingest.append((record, embedding))
                            
                        except Exception as e:
                            logger.warning(f"Ошибка при подготовке записи {record_id}: {e}")
                            total_errors += 1
                            continue
                    
                    # Сохраняем записи в граф
                    if records_to_ingest and not dry_run:
                        try:
                            records_only = [r for r, _ in records_to_ingest]
                            ingestor.ingest(records_only)
                            
                            # Сохраняем эмбеддинги в граф и Qdrant
                            for record, embedding in records_to_ingest:
                                # Проверяем, что эмбеддинг существует и не пустой
                                if embedding is not None and len(embedding) > 0:
                                    try:
                                        # Преобразуем numpy массив в список, если нужно
                                        if hasattr(embedding, 'tolist'):
                                            embedding = embedding.tolist()
                                        elif not isinstance(embedding, list):
                                            embedding = list(embedding)
                                        
                                        # Сохраняем эмбеддинг в граф
                                        graph.update_node(record.record_id, embedding=embedding)
                                        
                                        # Сохраняем эмбеддинг в Qdrant
                                        if vector_store:
                                            payload_data = {
                                                "record_id": record.record_id,
                                                "source": record.source,
                                                "tags": record.tags,
                                                "timestamp": record.timestamp.timestamp(),
                                                "timestamp_iso": record.timestamp.isoformat(),
                                                "content_preview": record.content[:200],
                                            }
                                            chat_name = record.metadata.get("chat")
                                            if isinstance(chat_name, str):
                                                payload_data["chat"] = chat_name
                                            
                                            try:
                                                vector_store.upsert(record.record_id, embedding, payload_data)
                                            except Exception as e:
                                                logger.debug(f"Ошибка при сохранении эмбеддинга в Qdrant для {record.record_id}: {e}")
                                    except Exception as e:
                                        logger.debug(f"Ошибка при сохранении эмбеддинга для {record.record_id}: {e}")
                            
                            synced_in_collection += len(records_to_ingest)
                            total_synced += len(records_to_ingest)
                            
                        except Exception as e:
                            logger.error(f"Ошибка при сохранении записей в граф: {e}")
                            total_errors += len(records_to_ingest)
                    elif records_to_ingest and dry_run:
                        synced_in_collection += len(records_to_ingest)
                        total_synced += len(records_to_ingest)
                    
                    offset += len(ids)
                    if len(ids) < batch_size:
                        break
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке батча (offset={offset}): {e}")
                    total_errors += batch_size
                    offset += batch_size
            
            if synced_in_collection > 0:
                logger.info(f"  ✅ Синхронизировано {synced_in_collection} записей из {collection_name}")
            
        except Exception as e:
            logger.error(f"Ошибка при синхронизации коллекции {collection_name}: {e}")
            total_errors += 1
    
    if dry_run:
        logger.info(f"🔍 Режим тестирования: было бы синхронизировано {total_synced} записей")
    else:
        logger.info(f"✅ Синхронизация завершена: {total_synced} записей, {total_errors} ошибок")
        if vector_store:
            logger.info("✅ Эмбеддинги сохранены в Qdrant")
    
    graph.conn.close()
    if vector_store:
        vector_store.close()
    if embedding_service:
        embedding_service.close()


@cli.command("stop-indexing")
def stop_indexing():
    """🛑 Остановка всех процессов индексации

    Останавливает все процессы индексации и Ollama сервер.
    """

    async def _stop_indexing():
        click.echo("🛑 Остановка процессов индексации...")
        click.echo()

        # Останавливаем все процессы
        ProcessManager.stop_all_indexing()

        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Остановка процессов завершена!")
        click.echo("=" * 80)

    asyncio.run(_stop_indexing())


@cli.command("review-summaries")
@click.option("--dry-run", is_flag=True, help="Только анализ, без изменения файлов")
@click.option("--chat", help="Обработать только конкретный чат")
@click.option("--limit", type=int, help="Максимальное количество файлов для обработки")
def review_summaries(dry_run, chat, limit):
    """🔍 Автоматическое ревью и исправление саммаризаций с суффиксом -needs-review

    Находит файлы *-needs-review.md, анализирует их через LLM и создает
    исправленные версии без суффикса -needs-review.
    """
    import json

    from ..core.lmstudio_client import LMStudioEmbeddingClient
    from ..config import get_settings

    async def _review_summaries():
        click.echo("🔍 Автоматическое ревью и исправление саммаризаций")
        click.echo()

        if dry_run:
            click.echo("🔸 Режим DRY RUN - файлы не будут изменены")
            click.echo()

        reports_dir = Path("artifacts/reports")

        if not reports_dir.exists():
            click.echo("❌ Директория artifacts/reports не найдена")
            return

        # Находим файлы с -needs-review
        needs_review_files = []
        for md_file in reports_dir.rglob("*-needs-review.md"):
            json_file = md_file.with_suffix(".json")

            file_info = {
                "md_file": md_file,
                "json_file": json_file if json_file.exists() else None,
                "session_id": md_file.stem.replace("-needs-review", ""),
                "chat": md_file.parent.parent.name,
            }

            # Фильтруем по чату если указан
            if chat and chat.lower() not in file_info["chat"].lower():
                continue

            needs_review_files.append(file_info)

        # Ограничиваем количество если указан лимит
        if limit:
            needs_review_files = needs_review_files[:limit]

        if not needs_review_files:
            click.echo("✅ Не найдено файлов с суффиксом -needs-review")
            return

        click.echo(f"📁 Найдено файлов для обработки: {len(needs_review_files)}")
        click.echo()

        # Создаем LLM клиент
        settings = get_settings()
        embedding_client = LMStudioEmbeddingClient(
            model_name=settings.lmstudio_model,
            base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
        )

        async def review_summary(md_content: str) -> dict:
            prompt = f"""Ты - эксперт по анализу и улучшению саммаризаций чатов.

Проанализируй следующую саммаризацию и улучши её, если нужно:

{md_content}

Твоя задача:
1. Проверить структуру и полноту информации
2. Исправить грамматические и стилистические ошибки
3. Улучшить ясность и читаемость
4. Убедиться, что все секции заполнены корректно
5. Добавить отсутствующую важную информацию, если она очевидна из контекста

ВАЖНО:
- Сохрани оригинальную структуру markdown (заголовки, списки, и т.д.)
- Не добавляй информацию, которой нет в оригинале
- Сохрани все даты, имена и технические детали
- Если саммаризация хорошая - верни её без изменений

Верни ТОЛЬКО улучшенный markdown-текст БЕЗ дополнительных комментариев."""

            try:
                async with embedding_client:
                    improved = await embedding_client.generate_summary(
                        prompt,
                        temperature=0.3,
                        max_tokens=131072,  # Для gpt-oss-20b (максимальный лимит)
                    )
                    improved = improved.strip()

                    if improved:

                        # Анализируем изменения
                        issues_found = []
                        if (
                            "_(Нет данных)_" in md_content
                            or "_(отсутствуют)_" in md_content
                        ):
                            issues_found.append("Есть пустые секции")
                        if len(md_content) < 200:
                            issues_found.append("Слишком короткая саммаризация")
                        if md_content.count("##") < 2:
                            issues_found.append("Недостаточная структуризация")

                        improvements = []
                        if md_content != improved:
                            improvements.append("Исправлены ошибки")
                        if len(improved) > len(md_content) * 1.1:
                            improvements.append("Расширен контент")
                        if not improvements:
                            improvements.append("Изменений не требуется")

                        return {
                            "improved_content": improved,
                            "issues_found": issues_found,
                            "improvements": improvements,
                            "success": True,
                        }
                    else:
                        return {
                            "improved_content": md_content,
                            "issues_found": [],
                            "improvements": [],
                            "success": False,
                            "error": "Нет ответа от LLM",
                        }

            except Exception as e:
                return {
                    "improved_content": md_content,
                    "issues_found": [],
                    "improvements": [],
                    "success": False,
                    "error": str(e),
                }

        # Обрабатываем каждый файл

        for file_info in needs_review_files:
            md_file = file_info["md_file"]
            json_file = file_info["json_file"]
            session_id = file_info["session_id"]

            click.echo(f"📄 Обработка: {md_file.name}")

            # Читаем markdown
            try:
                with open(md_file, encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                click.echo(f"   ❌ Ошибка чтения MD: {e}")
                continue

            # Проводим ревью
            click.echo("   🔍 Анализ через LLM...")
            review_result = await review_summary(md_content)

            if not review_result["success"]:
                click.echo(
                    f"   ❌ Ошибка анализа: {review_result.get('error', 'Unknown')}"
                )
                continue

            # Выводим результаты анализа
            if review_result["issues_found"]:
                click.echo(
                    f"   ⚠️  Найдено проблем: {', '.join(review_result['issues_found'])}"
                )

            if review_result["improvements"]:
                click.echo(
                    f"   ✨ Улучшения: {', '.join(review_result['improvements'])}"
                )

            if dry_run:
                click.echo("   🔸 DRY RUN - файл не изменён")
                continue

            # Сохраняем улучшенную версию
            new_md_file = md_file.parent / f"{session_id}.md"
            new_json_file = md_file.parent / f"{session_id}.json"

            try:
                # Сохраняем улучшенный markdown
                with open(new_md_file, "w", encoding="utf-8") as f:
                    f.write(review_result["improved_content"])

                # Обновляем JSON если нужно
                if json_file:
                    try:
                        with open(json_file, encoding="utf-8") as f:
                            session_data = json.load(f)

                        session_data["session_id"] = session_id

                        with open(new_json_file, "w", encoding="utf-8") as f:
                            json.dump(session_data, f, ensure_ascii=False, indent=2)

                        if new_json_file != json_file:
                            json_file.unlink()
                            click.echo(f"   🗑️  Удалён старый JSON: {json_file.name}")
                    except Exception as e:
                        click.echo(f"   ⚠️  Ошибка обновления JSON: {e}")

                if new_md_file != md_file:
                    md_file.unlink()
                    click.echo(f"   🗑️  Удалён старый MD: {md_file.name}")

                click.echo(f"   ✅ Сохранён исправленный файл: {new_md_file.name}")

            except Exception as e:
                click.echo(f"   ❌ Ошибка сохранения: {e}")

            # Небольшая задержка между запросами
            await asyncio.sleep(1)

        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Обработка завершена!")
        click.echo("=" * 80)

    asyncio.run(_review_summaries())


@cli.command("backup-database")
@click.option(
    "--backup-path",
    type=click.Path(path_type=Path),
    help="Путь для сохранения backup (по умолчанию: backups/backup_YYYYMMDD_HHMMSS)",
)
@click.option(
    "--include-chromadb/--no-chromadb",
    default=True,
    help="Включить ChromaDB в backup",
)
@click.option(
    "--include-reports/--no-reports",
    default=False,
    help="Включить markdown отчеты в backup",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    help="Создать сжатый .tar.gz архив",
)
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--chroma-path",
    default="chroma_db",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Путь к ChromaDB",
)
def backup_database(backup_path, include_chromadb, include_reports, compress, db_path, chroma_path):
    """📦 Создание резервной копии базы данных (SQLite + ChromaDB)
    
    Создаёт полную резервную копию всех данных системы:
    - SQLite база данных (memory_graph.db)
    - ChromaDB векторное хранилище (опционально)
    - Markdown отчеты (опционально)
    """
    import shutil
    import tarfile
    
    click.echo("📦 Создание резервной копии базы данных")
    click.echo()
    
    # Определяем путь для backup
    if not backup_path:
        backups_dir = Path("backups")
        backups_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backups_dir / f"backup_{timestamp}"
        if compress:
            backup_path = backup_path.with_suffix(".tar.gz")
    
    backup_path = Path(backup_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    includes = []
    temp_backup_dir = None
    
    try:
        if compress:
            # Для сжатого архива создаём временную директорию
            temp_backup_dir = Path(f"/tmp/memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            temp_backup_dir.mkdir(exist_ok=True)
            actual_backup_path = temp_backup_dir
        else:
            actual_backup_path = backup_path
            actual_backup_path.mkdir(exist_ok=True)
        
        # Копируем SQLite БД
        if db_path.exists():
            click.echo(f"📄 Копирование SQLite БД: {db_path}")
            db_backup_path = actual_backup_path / "memory_graph.db"
            shutil.copy2(db_path, db_backup_path)
            includes.append("sqlite_database")
            click.echo(f"   ✅ Размер: {db_backup_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            click.echo(f"⚠️  SQLite БД не найдена: {db_path}")
        
        # Копируем ChromaDB
        if include_chromadb and chroma_path.exists():
            click.echo(f"🔍 Копирование ChromaDB: {chroma_path}")
            chroma_backup_path = actual_backup_path / "chroma_db"
            shutil.copytree(chroma_path, chroma_backup_path, dirs_exist_ok=True)
            includes.append("chromadb")
            # Подсчитываем размер
            total_size = sum(f.stat().st_size for f in chroma_backup_path.rglob('*') if f.is_file())
            click.echo(f"   ✅ Размер: {total_size / 1024 / 1024:.2f} MB")
        elif include_chromadb:
            click.echo(f"⚠️  ChromaDB не найдена: {chroma_path}")
        
        # Копируем отчеты
        if include_reports:
            reports_path = Path("artifacts/reports")
            if reports_path.exists():
                click.echo(f"📊 Копирование отчетов: {reports_path}")
                reports_backup_path = actual_backup_path / "reports"
                shutil.copytree(reports_path, reports_backup_path, dirs_exist_ok=True)
                includes.append("reports")
                total_size = sum(f.stat().st_size for f in reports_backup_path.rglob('*') if f.is_file())
                click.echo(f"   ✅ Размер: {total_size / 1024 / 1024:.2f} MB")
            else:
                click.echo(f"⚠️  Отчеты не найдены: {reports_path}")
        
        # Создаём архив если нужно
        if compress and temp_backup_dir:
            click.echo(f"🗜️  Создание архива: {backup_path}")
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(temp_backup_dir, arcname=backup_path.stem)
            backup_size = backup_path.stat().st_size
            click.echo(f"   ✅ Размер архива: {backup_size / 1024 / 1024:.2f} MB")
        
        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Резервная копия успешно создана!")
        click.echo("=" * 80)
        click.echo(f"📁 Путь: {backup_path}")
        click.echo(f"📦 Включено: {', '.join(includes)}")
        if compress:
            click.echo(f"📊 Размер: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при создании резервной копии!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()
    finally:
        # Удаляем временную директорию
        if temp_backup_dir and temp_backup_dir.exists():
            shutil.rmtree(temp_backup_dir)


@cli.command("restore-database")
@click.option(
    "--backup-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к резервной копии (файл .tar.gz или директория)",
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Подтвердить восстановление (удалит текущие данные)",
)
@click.option(
    "--restore-chromadb/--no-chromadb",
    default=True,
    help="Восстановить ChromaDB",
)
@click.option(
    "--restore-reports/--no-reports",
    default=False,
    help="Восстановить markdown отчеты",
)
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--chroma-path",
    default="chroma_db",
    type=click.Path(file_okay=False, path_type=Path),
    help="Путь к ChromaDB",
)
def restore_database(backup_path, confirm, restore_chromadb, restore_reports, db_path, chroma_path):
    """🔄 Восстановление базы данных из резервной копии
    
    ВНИМАНИЕ: Эта операция удалит текущие данные и заменит их данными из backup!
    """
    import shutil
    import tarfile
    
    click.echo("🔄 Восстановление базы данных из резервной копии")
    click.echo()
    
    if not confirm:
        click.echo("⚠️  ВНИМАНИЕ: Эта операция удалит текущие данные!")
        click.echo(f"   Будет восстановлено из: {backup_path}")
        if not click.confirm("Продолжить?"):
            click.echo("❌ Операция отменена")
            return
    
    backup_path = Path(backup_path)
    temp_extract_dir = None
    
    try:
        # Распаковываем архив если нужно
        if backup_path.suffix == ".gz" or backup_path.suffixes == [".tar", ".gz"]:
            click.echo(f"📦 Распаковка архива: {backup_path}")
            temp_extract_dir = Path(f"/tmp/memory_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            temp_extract_dir.mkdir(exist_ok=True)
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(temp_extract_dir)
            # Находим распакованную директорию
            extracted_dirs = [d for d in temp_extract_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                source_dir = extracted_dirs[0]
            else:
                source_dir = temp_extract_dir
        else:
            source_dir = backup_path
        
        # Восстанавливаем SQLite БД
        db_backup = source_dir / "memory_graph.db"
        if db_backup.exists():
            click.echo(f"📄 Восстановление SQLite БД: {db_path}")
            if db_path.exists():
                # Создаём backup текущей БД
                old_db_backup = Path(f"{db_path}.old_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copy2(db_path, old_db_backup)
                click.echo(f"   💾 Текущая БД сохранена как: {old_db_backup}")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_backup, db_path)
            click.echo("   ✅ SQLite БД восстановлена")
        else:
            click.echo(f"⚠️  SQLite БД не найдена в backup: {db_backup}")
        
        # Восстанавливаем ChromaDB
        if restore_chromadb:
            chroma_backup = source_dir / "chroma_db"
            if chroma_backup.exists() and chroma_backup.is_dir():
                click.echo(f"🔍 Восстановление ChromaDB: {chroma_path}")
                if chroma_path.exists():
                    shutil.rmtree(chroma_path)
                chroma_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(chroma_backup, chroma_path)
                click.echo("   ✅ ChromaDB восстановлена")
            else:
                click.echo(f"⚠️  ChromaDB не найдена в backup: {chroma_backup}")
        
        # Восстанавливаем отчеты
        if restore_reports:
            reports_backup = source_dir / "reports"
            if reports_backup.exists() and reports_backup.is_dir():
                reports_path = Path("artifacts/reports")
                click.echo(f"📊 Восстановление отчетов: {reports_path}")
                if reports_path.exists():
                    shutil.rmtree(reports_path)
                reports_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(reports_backup, reports_path)
                click.echo("   ✅ Отчеты восстановлены")
            else:
                click.echo(f"⚠️  Отчеты не найдены в backup: {reports_backup}")
        
        click.echo()
        click.echo("=" * 80)
        click.echo("✅ База данных успешно восстановлена!")
        click.echo("=" * 80)
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при восстановлении базы данных!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()
    finally:
        # Удаляем временную директорию
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)


@cli.command("optimize-database")
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--vacuum/--no-vacuum",
    default=True,
    help="Выполнить VACUUM для освобождения места",
)
@click.option(
    "--analyze/--no-analyze",
    default=True,
    help="Выполнить ANALYZE для обновления статистики",
)
@click.option(
    "--reindex/--no-reindex",
    default=False,
    help="Выполнить REINDEX для пересоздания индексов",
)
@click.option(
    "--optimize-fts/--no-optimize-fts",
    default=True,
    help="Оптимизировать FTS5 индекс",
)
def optimize_database(db_path, vacuum, analyze, reindex, optimize_fts):
    """⚡ Оптимизация SQLite базы данных
    
    Выполняет операции оптимизации для улучшения производительности:
    - VACUUM: освобождает место, удаляя неиспользуемые страницы
    - ANALYZE: обновляет статистику для оптимизатора запросов
    - REINDEX: пересоздаёт индексы
    - FTS5 оптимизация: оптимизирует полнотекстовый поиск
    """
    import sqlite3
    import time
    
    click.echo("⚡ Оптимизация SQLite базы данных")
    click.echo()
    
    if not db_path.exists():
        click.echo(f"❌ База данных не найдена: {db_path}")
        raise click.Abort()
    
    # Получаем размер до оптимизации
    size_before = db_path.stat().st_size
    
    operations_performed = []
    start_time = time.time()
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # VACUUM
        if vacuum:
            click.echo("🧹 Выполнение VACUUM...")
            cursor.execute("VACUUM")
            conn.commit()
            operations_performed.append("VACUUM")
            click.echo("   ✅ VACUUM выполнен")
        
        # ANALYZE
        if analyze:
            click.echo("📊 Выполнение ANALYZE...")
            cursor.execute("ANALYZE")
            conn.commit()
            operations_performed.append("ANALYZE")
            click.echo("   ✅ ANALYZE выполнен")
        
        # REINDEX
        if reindex:
            click.echo("🔄 Выполнение REINDEX...")
            cursor.execute("REINDEX")
            conn.commit()
            operations_performed.append("REINDEX")
            click.echo("   ✅ REINDEX выполнен")
        
        # FTS5 оптимизация
        if optimize_fts:
            click.echo("🔍 Оптимизация FTS5 индекса...")
            try:
                cursor.execute("INSERT INTO node_search(node_search) VALUES('optimize')")
                conn.commit()
                operations_performed.append("FTS5_optimize")
                click.echo("   ✅ FTS5 индекс оптимизирован")
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e).lower():
                    raise
                click.echo("   ⚠️  FTS5 таблица не найдена, пропускаем")
        
        conn.close()
        
        # Получаем размер после оптимизации
        size_after = db_path.stat().st_size
        space_freed = size_before - size_after
        duration = time.time() - start_time
        
        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Оптимизация завершена!")
        click.echo("=" * 80)
        click.echo(f"📊 Операции: {', '.join(operations_performed)}")
        click.echo(f"📦 Размер до: {size_before / 1024 / 1024:.2f} MB")
        click.echo(f"📦 Размер после: {size_after / 1024 / 1024:.2f} MB")
        if space_freed > 0:
            click.echo(f"💾 Освобождено: {space_freed / 1024 / 1024:.2f} MB")
        click.echo(f"⏱️  Время выполнения: {duration:.2f} сек")
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при оптимизации базы данных!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command("validate-database")
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--check-integrity/--no-check-integrity",
    default=True,
    help="Проверить целостность SQLite (PRAGMA integrity_check)",
)
@click.option(
    "--check-foreign-keys/--no-check-foreign-keys",
    default=True,
    help="Проверить внешние ключи (PRAGMA foreign_key_check)",
)
@click.option(
    "--check-orphaned-nodes/--no-check-orphaned-nodes",
    default=True,
    help="Проверить узлы без связей",
)
@click.option(
    "--check-orphaned-edges/--no-check-orphaned-edges",
    default=True,
    help="Проверить рёбра с несуществующими узлами",
)
def validate_database(db_path, check_integrity, check_foreign_keys, check_orphaned_nodes, check_orphaned_edges):
    """🔍 Проверка целостности базы данных
    
    Выполняет комплексную проверку целостности данных:
    - Проверка целостности SQLite
    - Проверка внешних ключей
    - Проверка графа знаний (сиротские узлы и рёбра)
    """
    import sqlite3
    
    click.echo("🔍 Проверка целостности базы данных")
    click.echo()
    
    if not db_path.exists():
        click.echo(f"❌ База данных не найдена: {db_path}")
        raise click.Abort()
    
    issues = []
    checks_performed = []
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Проверка целостности SQLite
        if check_integrity:
            click.echo("🔍 Проверка целостности SQLite...")
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            checks_performed.append("integrity_check")
            if result == "ok":
                click.echo("   ✅ Целостность SQLite: OK")
            else:
                click.echo(f"   ❌ Проблемы с целостностью: {result}")
                issues.append({
                    "type": "integrity",
                    "severity": "error",
                    "message": f"SQLite integrity check failed: {result}",
                    "details": {"result": result}
                })
        
        # Проверка внешних ключей
        if check_foreign_keys:
            click.echo("🔗 Проверка внешних ключей...")
            cursor.execute("PRAGMA foreign_key_check")
            foreign_key_issues = cursor.fetchall()
            checks_performed.append("foreign_key_check")
            if not foreign_key_issues:
                click.echo("   ✅ Внешние ключи: OK")
            else:
                click.echo(f"   ❌ Найдено проблем с внешними ключами: {len(foreign_key_issues)}")
                for issue in foreign_key_issues:
                    issues.append({
                        "type": "foreign_key",
                        "severity": "error",
                        "message": f"Foreign key violation: {dict(issue)}",
                        "details": dict(issue)
                    })
        
        # Проверка графа знаний
        if check_orphaned_nodes or check_orphaned_edges:
            click.echo("🕸️  Проверка графа знаний...")
            from ..memory.typed_graph import TypedGraphMemory
            graph = TypedGraphMemory(db_path=str(db_path))
            
            # Проверка сиротских узлов
            if check_orphaned_nodes:
                cursor.execute("""
                    SELECT id, type 
                    FROM nodes 
                    WHERE id NOT IN (
                        SELECT DISTINCT source_id FROM edges
                        UNION
                        SELECT DISTINCT target_id FROM edges
                    )
                """)
                orphaned_nodes = cursor.fetchall()
                checks_performed.append("orphaned_nodes")
                if not orphaned_nodes:
                    click.echo("   ✅ Сиротские узлы: не найдено")
                else:
                    click.echo(f"   ⚠️  Найдено сиротских узлов: {len(orphaned_nodes)}")
                    for node in orphaned_nodes[:10]:  # Показываем первые 10
                        issues.append({
                            "type": "orphaned_node",
                            "severity": "warning",
                            "message": f"Node '{node['id']}' has no connections",
                            "details": {"node_id": node["id"], "node_type": node["type"]}
                        })
            
            # Проверка сиротских рёбер
            if check_orphaned_edges:
                cursor.execute("""
                    SELECT e.id, e.source_id, e.target_id, e.type
                    FROM edges e
                    LEFT JOIN nodes n1 ON e.source_id = n1.id
                    LEFT JOIN nodes n2 ON e.target_id = n2.id
                    WHERE n1.id IS NULL OR n2.id IS NULL
                """)
                orphaned_edges = cursor.fetchall()
                checks_performed.append("orphaned_edges")
                if not orphaned_edges:
                    click.echo("   ✅ Сиротские рёбра: не найдено")
                else:
                    click.echo(f"   ❌ Найдено сиротских рёбер: {len(orphaned_edges)}")
                    for edge in orphaned_edges[:10]:  # Показываем первые 10
                        issues.append({
                            "type": "orphaned_edge",
                            "severity": "error",
                            "message": f"Edge '{edge['id']}' references non-existent node",
                            "details": {
                                "edge_id": edge["id"],
                                "source_id": edge["source_id"],
                                "target_id": edge["target_id"],
                                "edge_type": edge["type"]
                            }
                        })
        
        conn.close()
        
        click.echo()
        click.echo("=" * 80)
        if not issues:
            click.echo("✅ База данных валидна! Проблем не обнаружено.")
        else:
            click.echo(f"⚠️  Найдено проблем: {len(issues)}")
            click.echo()
            for issue in issues[:20]:  # Показываем первые 20
                severity_icon = "❌" if issue["severity"] == "error" else "⚠️"
                click.echo(f"{severity_icon} [{issue['type']}] {issue['message']}")
        click.echo("=" * 80)
        click.echo(f"📊 Выполнено проверок: {', '.join(checks_performed)}")
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при проверке базы данных!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command("calculate-importance")
@click.option(
    "--record-id",
    required=True,
    help="ID записи для оценки важности",
)
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--entity-weight",
    type=float,
    default=0.1,
    help="Вес за каждую сущность",
)
@click.option(
    "--task-weight",
    type=float,
    default=0.3,
    help="Вес за наличие задачи",
)
@click.option(
    "--length-weight",
    type=float,
    default=0.2,
    help="Вес за длину сообщения",
)
@click.option(
    "--search-hits-weight",
    type=float,
    default=0.4,
    help="Вес за частоту поиска",
)
def calculate_importance(record_id, db_path, entity_weight, task_weight, length_weight, search_hits_weight):
    """📊 Вычисление важности записи
    
    Вычисляет importance score (0.0-1.0) для указанной записи на основе:
    - Наличия сущностей
    - Наличия задач/action items
    - Длины контента
    - Частоты поиска
    """
    from ..memory.importance_scoring import ImportanceScorer
    from ..memory.typed_graph import TypedGraphMemory
    import sqlite3
    
    click.echo(f"📊 Вычисление важности записи: {record_id}")
    click.echo()
    
    try:
        # Инициализируем граф и scorer
        graph = TypedGraphMemory(db_path=str(db_path))
        scorer = ImportanceScorer(
            entity_weight=entity_weight,
            task_weight=task_weight,
            length_weight=length_weight,
            search_hits_weight=search_hits_weight
        )
        
        # Получаем запись из БД
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (record_id,))
        node = cursor.fetchone()
        
        if not node:
            click.echo(f"❌ Запись не найдена: {record_id}")
            raise click.Abort()
        
        # Преобразуем узел в словарь
        node_dict = dict(node)
        properties = json.loads(node_dict.get("properties", "{}") or "{}")
        node_dict.update(properties)
        
        # Получаем метаданные (частота поиска и т.д.)
        metadata = {
            "_search_hits": properties.get("_search_hits", 0)
        }
        
        # Вычисляем важность
        importance_score = scorer.compute_importance(node_dict, metadata)
        
        # Вычисляем факторы отдельно для детализации
        factors = {}
        entities = node_dict.get("entities") or properties.get("entities", [])
        if entities:
            factors["entities"] = min(len(entities) * entity_weight, 0.5)
        if node_dict.get("has_task") or node_dict.get("is_action_item") or properties.get("has_task") or properties.get("is_action_item"):
            factors["task"] = task_weight
        text = node_dict.get("text", "") or node_dict.get("content", "") or properties.get("content", "")
        if len(text) > 500:
            factors["length"] = length_weight
        elif len(text) > 200:
            factors["length"] = length_weight * 0.5
        if metadata.get("_search_hits", 0) > 0:
            factors["search_hits"] = min(metadata["_search_hits"] / 10.0, 1.0) * search_hits_weight
        
        conn.close()
        
        click.echo("=" * 80)
        click.echo("📊 Результаты оценки важности")
        click.echo("=" * 80)
        click.echo(f"📝 Запись: {record_id}")
        click.echo(f"⭐ Importance Score: {importance_score:.3f} (0.0 - 1.0)")
        click.echo()
        click.echo("📈 Факторы:")
        for factor, value in factors.items():
            click.echo(f"   • {factor}: {value:.3f}")
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при вычислении важности!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command("prune-memory")
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--max-records",
    type=int,
    default=100000,
    help="Максимальное количество записей",
)
@click.option(
    "--eviction-threshold",
    type=float,
    default=0.7,
    help="Порог eviction score для удаления (0.0-1.0)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Только анализ, без удаления",
)
@click.option(
    "--source",
    help="Фильтр по источнику (опционально)",
)
def prune_memory(db_path, max_records, eviction_threshold, dry_run, source):
    """🧹 Автоматическая очистка неважных записей
    
    Удаляет записи с низкой важностью для управления размером БД.
    Использует систему оценки важности (Importance Scoring).
    """
    from ..memory.importance_scoring import MemoryPruner, EvictionScorer
    from ..memory.typed_graph import TypedGraphMemory
    import sqlite3
    
    click.echo("🧹 Автоматическая очистка памяти")
    click.echo()
    
    if dry_run:
        click.echo("🔸 Режим DRY RUN - записи не будут удалены")
        click.echo()
    
    try:
        graph = TypedGraphMemory(db_path=str(db_path))
        eviction_scorer = EvictionScorer()
        pruner = MemoryPruner(
            eviction_scorer=eviction_scorer,
            max_messages=max_records,
            eviction_threshold=eviction_threshold
        )
        
        # Получаем все записи
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM nodes"
        params = []
        if source:
            query += " WHERE properties LIKE ?"
            params.append(f'%"source": "{source}"%')
        
        cursor.execute(query, params)
        nodes = cursor.fetchall()
        
        current_count = len(nodes)
        click.echo(f"📊 Текущее количество записей: {current_count}")
        
        if not pruner.should_prune(current_count):
            click.echo("✅ Очистка не требуется (количество записей в пределах лимита)")
            conn.close()
            return
        
        click.echo(f"⚠️  Превышен лимит ({max_records}), требуется очистка")
        click.echo()
        
        # Преобразуем узлы в словари
        messages = []
        for node in nodes:
            node_dict = dict(node)
            properties = json.loads(node_dict.get("properties", "{}") or "{}")
            node_dict.update(properties)
            # Убеждаемся, что есть поле id или msg_id для get_eviction_candidates
            if "id" not in node_dict and "msg_id" not in node_dict:
                node_dict["id"] = node_dict.get("node_id") or node["id"]
            messages.append(node_dict)
        
        # Получаем кандидатов на удаление
        candidates = pruner.get_eviction_candidates(
            messages,
            threshold=eviction_threshold
        )
        
        click.echo(f"🎯 Найдено кандидатов на удаление: {len(candidates)}")
        click.echo()
        
        if not dry_run and candidates:
            click.echo("🗑️  Удаление записей...")
            removed_count = 0
            for candidate in candidates:
                try:
                    # get_eviction_candidates возвращает msg_id
                    node_id = candidate.get("msg_id") or candidate.get("message", {}).get("id")
                    if node_id:
                        graph.delete_node(node_id)
                        removed_count += 1
                except Exception as e:
                    click.echo(f"   ⚠️  Ошибка удаления {node_id}: {e}")
            
            click.echo(f"   ✅ Удалено записей: {removed_count}")
        
        conn.close()
        
        click.echo()
        click.echo("=" * 80)
        if dry_run:
            click.echo("🔸 DRY RUN завершён")
            click.echo(f"📊 Будет удалено записей: {len(candidates)}")
        else:
            click.echo("✅ Очистка памяти завершена")
        click.echo("=" * 80)
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при очистке памяти!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command("update-importance-scores")
@click.option(
    "--db-path",
    default="data/memory_graph.db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Путь к SQLite базе данных",
)
@click.option(
    "--source",
    help="Обновить только для указанного источника",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Размер батча для обработки",
)
def update_importance_scores(db_path, source, batch_size):
    """🔄 Массовый пересчёт важности записей
    
    Пересчитывает importance scores для всех записей в базе данных.
    Полезно после изменения весов факторов или обновления системы оценки.
    """
    from ..memory.importance_scoring import ImportanceScorer
    from ..memory.typed_graph import TypedGraphMemory
    import sqlite3
    
    click.echo("🔄 Массовый пересчёт важности записей")
    click.echo()
    
    try:
        graph = TypedGraphMemory(db_path=str(db_path))
        scorer = ImportanceScorer()
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем все записи
        query = "SELECT * FROM nodes"
        params = []
        if source:
            query += " WHERE properties LIKE ?"
            params.append(f'%"source": "{source}"%')
        
        cursor.execute(query, params)
        nodes = cursor.fetchall()
        
        total_nodes = len(nodes)
        click.echo(f"📊 Всего записей для обработки: {total_nodes}")
        click.echo()
        
        updated_count = 0
        importance_scores = []
        
        for i, node in enumerate(nodes, 1):
            try:
                node_dict = dict(node)
                properties = json.loads(node_dict.get("properties", "{}"))
                node_dict.update(properties)
                
                metadata = {
                    "_search_hits": properties.get("_search_hits", 0)
                }
                
                importance_score = scorer.compute_importance(node_dict, metadata)
                importance_scores.append(importance_score)
                
                # Обновляем properties с новым importance_score
                properties["_importance_score"] = importance_score
                
                # Обновляем узел в графе
                graph.update_node(
                    node_id=node_dict["id"],
                    properties=properties
                )
                
                updated_count += 1
                
                if i % batch_size == 0:
                    click.echo(f"   ⏳ Обработано: {i}/{total_nodes} ({i*100//total_nodes}%)")
            
            except Exception as e:
                node_id_str = node_dict.get("id", "unknown")
                click.echo(f"   ⚠️  Ошибка обработки {node_id_str}: {e}")
        
        conn.close()
        
        # Статистика
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0
        min_importance = min(importance_scores) if importance_scores else 0
        max_importance = max(importance_scores) if importance_scores else 0
        
        click.echo()
        click.echo("=" * 80)
        click.echo("✅ Пересчёт важности завершён")
        click.echo("=" * 80)
        click.echo(f"📊 Обновлено записей: {updated_count}")
        click.echo(f"⭐ Средняя важность: {avg_importance:.3f}")
        click.echo(f"📉 Минимальная важность: {min_importance:.3f}")
        click.echo(f"📈 Максимальная важность: {max_importance:.3f}")
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo("=" * 80)
        click.echo("❌ Ошибка при пересчёте важности!")
        click.echo("=" * 80)
        click.echo(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


def main():
    """Главная функция CLI"""
    cli()


if __name__ == "__main__":
    main()
