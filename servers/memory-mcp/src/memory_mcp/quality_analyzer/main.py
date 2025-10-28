#!/usr/bin/env python3
"""Convenience helpers to run quality analysis programmatically."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .quality_analyzer import QualityAnalyzer
from .utils import load_chats_from_directory

logger = logging.getLogger(__name__)


class QualityAnalysisError(RuntimeError):
    """Raised when quality analysis cannot be completed."""


async def _analyze(
    chats: dict[str, list[dict[str, Any]]],
    analyzer: QualityAnalyzer,
    *,
    max_queries: int | None = None,
    batch_size: int | None = None,
    custom_queries: list[Any] | None = None,
):
    if len(chats) == 1:
        chat_name = next(iter(chats))
        return await analyzer.analyze_chat_quality(
            chat_name,
            chats[chat_name],
            batch_size=batch_size,
            max_queries=max_queries,
            custom_queries=custom_queries,
        )
    return await analyzer.analyze_multiple_chats(
        chats,
        batch_size=batch_size,
        max_queries=max_queries,
        custom_queries=custom_queries,
    )


def run_quality_analysis(
    chats_data: dict[str, list[dict]] | None = None,
    selected_chats: Iterable[str] | None = None,
    config_path: Path | None = None,
    *,
    max_queries: int | None = None,
    batch_size: int | None = None,
    custom_queries_path: Path | None = None,
):
    """Runs quality analysis using provided chats data."""

    logger.info(
        "Запуск анализа качества (config=%s, chats=%s, max_queries=%s, batch_size=%s)",
        str(config_path) if config_path else "<default>",
        list(selected_chats) if selected_chats else "<all>",
        max_queries,
        batch_size,
    )
    try:
        config = load_config(config_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load quality analysis config", exc_info=True)
        raise QualityAnalysisError(
            "Не удалось загрузить конфигурацию качества"
        ) from exc

    logger.info(
        "Конфигурация загружена: chats_dir=%s, reports_dir=%s, history_dir=%s",
        config.chats_dir,
        config.reports_dir,
        config.history_dir,
    )

    try:
        analyzer = QualityAnalyzer(
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
            max_context_tokens=config.max_context_tokens,
            ollama_temperature=config.temperature,
            ollama_max_tokens=config.max_response_tokens,
            ollama_thinking_level=config.thinking_level,
            reports_dir=config.reports_dir,
            history_dir=config.history_dir,
            reports_subdir=config.quality_reports_subdir,
            results_per_query=config.results_per_query,
            chroma_path=config.chroma_path,
            search_collection=config.search_collection,
            hybrid_alpha=config.hybrid_alpha,
            batch_max_size=config.batch_max_size,
            system_prompt_reserve=config.system_prompt_reserve,
            max_query_tokens=config.max_query_tokens,
        )
    except Exception as exc:  # pragma: no cover - initialization errors
        logger.error("Failed to initialize QualityAnalyzer", exc_info=True)
        raise QualityAnalysisError(
            "Не удалось инициализировать анализатор качества"
        ) from exc

    try:
        if custom_queries_path:
            custom_queries = _load_custom_queries(custom_queries_path)
        else:
            custom_queries = None
    except QualityAnalysisError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected format
        logger.error("Failed to load custom queries", exc_info=True)
        raise QualityAnalysisError(
            "Не удалось прочитать файл пользовательских запросов"
        ) from exc

    if custom_queries_path:
        logger.info(
            "Пользовательские запросы: %d элементов из %s",
            len(custom_queries or []),
            custom_queries_path,
        )

    try:
        filtered = load_chats_from_directory(
            config.chats_dir,
            selected_chats,
        )

        if chats_data:
            for name, data in chats_data.items():
                filtered.setdefault(name, []).extend(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to prepare chat data", exc_info=True)
        raise QualityAnalysisError(
            "Ошибка подготовки данных чатов для анализа"
        ) from exc

    logger.info("Загружено чатов: %d", len(filtered))

    if filtered:
        sample_name = next(iter(filtered))
        logger.info(
            "Чат '%s': %d сообщений",
            sample_name,
            len(filtered[sample_name]),
        )

    if not filtered:
        raise QualityAnalysisError("Не найдены данные чатов для анализа")

    if selected_chats:
        missing = [
            name
            for name in selected_chats
            if name.lower() not in {c.lower() for c in filtered}
        ]
        if missing:
            logger.warning("Недоступные чаты: %s", ", ".join(sorted(set(missing))))
            raise QualityAnalysisError(
                f"Чаты не найдены: {', '.join(sorted(set(missing)))}"
            )

    try:
        logger.info("Запускаем анализ (%d чатов)...", len(filtered))
        result = asyncio.run(
            _analyze(
                filtered,
                analyzer,
                max_queries=max_queries or config.max_queries_per_chat,
                batch_size=batch_size or config.batch_size,
                custom_queries=custom_queries,
            )
        )
        logger.info("Анализ завершён")
        return result
    except QualityAnalysisError:
        raise
    except RuntimeError as exc:
        if "asyncio.run" in str(exc):
            raise QualityAnalysisError(
                "Запуск анализа невозможен внутри активного asyncio цикла"
            ) from exc
        raise QualityAnalysisError("Не удалось выполнить анализ качества") from exc
    except Exception as exc:  # pragma: no cover - catch unexpected errors
        logger.error("Quality analysis run failed", exc_info=True)
        raise QualityAnalysisError("Не удалось выполнить анализ качества") from exc


def _load_custom_queries(path: Path) -> list[Any]:
    if not path.exists():
        raise QualityAnalysisError(
            f"Файл с пользовательскими запросами не найден: {path}"
        )

    with path.open("r", encoding="utf-8") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as exc:
            raise QualityAnalysisError(
                f"Некорректный JSON в файле пользовательских запросов: {path}"
            ) from exc

    if not isinstance(data, list):
        raise QualityAnalysisError(
            "Файл пользовательских запросов должен содержать список объектов"
        )

    return data


def _print_summary(result: Any) -> None:
    if not isinstance(result, dict):
        print("✅ Анализ завершён")
        return

    errors = result.get("errors")
    if errors:
        print("⚠️  Были зафиксированы ошибки во время анализа")

    if "overall_metrics" in result:
        overall_metrics = result.get("overall_metrics", {})
        avg_score = overall_metrics.get("average_score", 0.0)
        median_score = overall_metrics.get("median_score", 0.0)
        total_chats = result.get("total_chats", 0)
        print(
            f"📈 Средняя оценка качества: {avg_score:.2f}/10 (медиана {median_score:.2f})"
        )
        print(f"📊 Проанализировано чатов: {total_chats}")

        chat_results = result.get("chat_results", {})
        if chat_results:
            print("\n📋 Результаты по чатам:")
            for chat_name, chat_data in chat_results.items():
                if chat_data.get("error"):
                    print(f"   ❌ {chat_name}: {chat_data['error']}")
                    continue
                metrics = chat_data.get("metrics", {})
                basic = (
                    metrics.get("details", {}).get("basic", {})
                    if "details" in metrics
                    else metrics.get("basic", {})
                )
                avg = basic.get("average_score", 0.0)
                success = basic.get("success_rate", 0.0) * 100
                total_queries = basic.get("total_queries", 0)
                print(
                    f"   ✅ {chat_name}: {avg:.2f}/10 — {success:.1f}% успеха, запросов: {total_queries}"
                )
        return

    metrics = result.get("metrics", {})
    basic = (
        metrics.get("details", {}).get("basic", {})
        if isinstance(metrics.get("details"), dict)
        else metrics
    )
    avg_score = basic.get("average_score", 0.0)
    success_rate = basic.get("success_rate", 0.0) * 100
    total_queries = basic.get("total_queries", 0)
    print(f"📈 Средняя оценка качества: {avg_score:.2f}/10")
    print(f"📊 Успешные поиски: {success_rate:.1f}%")
    print(f"🔍 Тестовых запросов: {total_queries}")


def main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    parser = argparse.ArgumentParser(
        description="Запуск анализа качества индексации и поиска",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help="Путь к config/quality_analysis.json",
    )
    parser.add_argument(
        "--chat",
        dest="selected_chats",
        action="append",
        help="Название чата для анализа (можно указать несколько раз)",
    )
    parser.add_argument("--max-queries", type=int, help="Максимум запросов на чат")
    parser.add_argument("--batch-size", type=int, help="Размер батча обработки")
    parser.add_argument(
        "--custom-queries",
        type=Path,
        help="JSON-файл с пользовательскими запросами",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результат в формате JSON",
    )

    args = parser.parse_args(argv)

    try:
        result = run_quality_analysis(
            selected_chats=args.selected_chats,
            config_path=args.config_path,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            custom_queries_path=args.custom_queries,
        )
    except QualityAnalysisError as exc:
        logger.error("Quality analysis failed: %s", exc)
        print(f"❌ {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - unexpected issues
        logger.exception("Unexpected quality analysis failure")
        print(f"❌ Неожиданная ошибка: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result)

    return 0


__all__ = ["run_quality_analysis", "QualityAnalysisError", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
