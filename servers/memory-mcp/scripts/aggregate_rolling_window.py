#!/usr/bin/env python3
"""
CLI для системы континуальной агрегации с скользящим окном

Использование:
    python aggregate_rolling_window.py --help
    python aggregate_rolling_window.py --dry-run
    python aggregate_rolling_window.py --strategy conservative
    python aggregate_rolling_window.py --chat "ChatName" --report
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_mcp.analysis.rolling_window_aggregator import (
    AGGRESSIVE_STRATEGY,
    CONSERVATIVE_STRATEGY,
    MINIMAL_STRATEGY,
    RollingWindowAggregator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


STRATEGIES = {
    "conservative": CONSERVATIVE_STRATEGY,
    "aggressive": AGGRESSIVE_STRATEGY,
    "minimal": MINIMAL_STRATEGY,
}


def print_strategy_info(strategy_name: str):
    """Выводит информацию о стратегии"""
    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        return

    print(f"\n📋 Стратегия: {strategy_name.upper()}")
    print("=" * 70)

    for window in strategy:
        print(f"\n🪟 Окно: {window.name}")
        print(f"   Возраст: {window.age_days_min}-{window.age_days_max} дней")
        print(f"   Группировка: {window.group_by}")
        print(f"   Сохранять оригинал: {'Да' if window.keep_original else 'Нет'}")
        print(f"   Саммаризация: {'Да' if window.summarize else 'Нет'}")


def print_stats(stats: dict, detailed: bool = False):
    """Выводит статистику агрегации"""
    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА АГРЕГАЦИИ")
    print("=" * 70)

    print(f"\n✅ Обработано чатов: {stats['processed_chats']}/{stats['total_chats']}")
    print(f"📝 Создано саммаризаций: {stats['total_summaries']}")
    print(f"📦 Агрегировано блоков: {stats['total_blocks']}")

    if detailed and stats.get("chats"):
        print("\n" + "-" * 70)
        print("Детальная статистика по чатам:")
        print("-" * 70)

        for chat_name, chat_stats in stats["chats"].items():
            if "error" in chat_stats:
                print(f"\n❌ {chat_name}: {chat_stats['error']}")
                continue

            print(f"\n💬 {chat_name}")
            print(f"   Всего сообщений: {chat_stats.get('total_messages', 0)}")
            print(f"   Создано саммаризаций: {chat_stats.get('summaries_created', 0)}")
            print(f"   Агрегировано блоков: {chat_stats.get('blocks_aggregated', 0)}")

            if chat_stats.get("windows"):
                print("   Окна:")
                for window_name, window_stats in chat_stats["windows"].items():
                    msg_count = window_stats.get("messages_count", 0)
                    blocks = window_stats.get("blocks_count", 0)
                    print(
                        f"      - {window_name}: {msg_count} сообщений, {blocks} блоков"
                    )


def print_report(report: dict):
    """Выводит отчет об агрегации чата"""
    print("\n" + "=" * 70)
    print(f"📋 ОТЧЕТ ОБ АГРЕГАЦИИ: {report['chat_name']}")
    print("=" * 70)

    print(f"\n⏰ Последняя агрегация: {report.get('last_aggregation', 'Никогда')}")
    print(f"📦 Всего блоков: {report.get('total_blocks', 0)}")
    print(f"📝 Всего саммаризаций: {report.get('total_summaries', 0)}")

    if report.get("window_boundaries"):
        print("\n🪟 Границы окон:")
        for window, boundary in report["window_boundaries"].items():
            print(f"   {window}: {boundary}")

    recent_blocks = report.get("recent_blocks", [])
    if recent_blocks:
        print(f"\n📚 Последние {len(recent_blocks)} блоков:")
        for block in recent_blocks:
            print(f"\n   ID: {block['block_id']}")
            print(f"   Окно: {block['window']}")
            print(f"   Сообщений: {block['message_count']}")
            print(f"   Саммаризация: {block['summary'][:100]}...")


async def main():
    parser = argparse.ArgumentParser(
        description="Континуальная агрегация сообщений с скользящим окном",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Показать доступные стратегии
  python aggregate_rolling_window.py --list-strategies

  # Тестовый запуск с консервативной стратегией
  python aggregate_rolling_window.py --dry-run --strategy conservative

  # Реальная агрегация всех чатов
  python aggregate_rolling_window.py --strategy conservative

  # Агрегация конкретного чата
  python aggregate_rolling_window.py --chat "ChatName"

  # Получить отчет по чату
  python aggregate_rolling_window.py --chat "ChatName" --report

  # Агрессивная стратегия с подробным выводом
  python aggregate_rolling_window.py --strategy aggressive --detailed
        """,
    )

    parser.add_argument(
        "--chats-dir", default="chats", help="Директория с чатами (по умолчанию: chats)"
    )
    parser.add_argument(
        "--state-dir",
        default="aggregation_state",
        help="Директория для состояния (по умолчанию: aggregation_state)",
    )
    parser.add_argument(
        "--strategy",
        choices=["conservative", "aggressive", "minimal"],
        default="conservative",
        help="Стратегия агрегации (по умолчанию: conservative)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Размер батча для саммаризации (по умолчанию: 50)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Максимум параллельных агрегаций (по умолчанию: 3)",
    )

    parser.add_argument("--chat", type=str, help="Обработать только указанный чат")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Показать отчет об агрегации (используется с --chat)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Тестовый запуск без сохранения"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Подробный вывод статистики"
    )
    parser.add_argument(
        "--list-strategies", action="store_true", help="Показать доступные стратегии"
    )

    args = parser.parse_args()

    # Показать стратегии
    if args.list_strategies:
        print("\n🎯 ДОСТУПНЫЕ СТРАТЕГИИ АГРЕГАЦИИ")
        print("=" * 70)
        for strategy_name in STRATEGIES.keys():
            print_strategy_info(strategy_name)
        return

    # Показать информацию о выбранной стратегии
    if not args.report:
        print_strategy_info(args.strategy)

    # Создаем агрегатор
    aggregator = RollingWindowAggregator(
        chats_dir=Path(args.chats_dir),
        state_dir=Path(args.state_dir),
        strategy=STRATEGIES[args.strategy],
        batch_size=args.batch_size,
    )

    # Режим отчета
    if args.report:
        if not args.chat:
            print("❌ Ошибка: --report требует указания --chat")
            sys.exit(1)

        report = aggregator.get_aggregation_report(args.chat)
        print_report(report)
        return

    # Агрегация
    if args.chat:
        # Один чат
        logger.info(f"Агрегация чата: {args.chat}")
        stats = await aggregator.aggregate_chat(args.chat, dry_run=args.dry_run)

        print("\n" + "=" * 70)
        print(f"📊 СТАТИСТИКА ДЛЯ ЧАТА: {args.chat}")
        print("=" * 70)

        if "error" in stats:
            print(f"\n❌ Ошибка: {stats['error']}")
        else:
            print(f"\n✅ Всего сообщений: {stats.get('total_messages', 0)}")
            print(f"📝 Создано саммаризаций: {stats.get('summaries_created', 0)}")
            print(f"📦 Агрегировано блоков: {stats.get('blocks_aggregated', 0)}")

            if stats.get("windows"):
                print("\n🪟 Распределение по окнам:")
                for window_name, window_stats in stats["windows"].items():
                    print(f"\n   {window_name}:")
                    print(f"      Сообщений: {window_stats.get('messages_count', 0)}")
                    print(f"      Блоков: {window_stats.get('blocks_count', 0)}")
    else:
        # Все чаты
        logger.info("Агрегация всех чатов")
        stats = await aggregator.aggregate_all_chats(
            dry_run=args.dry_run, max_concurrent=args.max_concurrent
        )
        print_stats(stats, detailed=args.detailed)

    if args.dry_run:
        print(
            "\n💡 Это был тестовый запуск. Для реального выполнения запустите без --dry-run"
        )
    else:
        print(f"\n✅ Агрегация завершена! Состояние сохранено в {args.state_dir}/")
        print(
            f'   Для просмотра отчета: python {sys.argv[0]} --chat "ChatName" --report'
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
