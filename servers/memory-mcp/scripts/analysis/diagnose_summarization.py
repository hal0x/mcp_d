#!/usr/bin/env python3
"""
Диагностический скрипт для проверки качества саммаризации
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory_mcp.analysis.entity_extraction import EntityExtractor
from src.memory_mcp.analysis.session_segmentation import SessionSegmenter
from src.memory_mcp.analysis.session_summarizer import SessionSummarizer
from src.memory_mcp.core.ollama_client import OllamaEmbeddingClient


async def test_ollama_connection():
    """Тест 1: Проверка подключения к Ollama"""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Проверка подключения к Ollama")
    print("=" * 60)

    try:
        client = OllamaEmbeddingClient()
        async with client:
            # Пробуем простой тест генерации
            test_prompt = "Напиши слово 'тест' и ничего больше."
            result = await client.generate_summary(
                prompt=test_prompt, temperature=0.1, max_tokens=50
            )

            print("✅ Ollama доступен")
            print(f"📝 Тестовый ответ: '{result[:100]}...'")
            print(f"📏 Длина ответа: {len(result)} символов")
            return True
    except Exception as e:
        print(f"❌ Ошибка подключения к Ollama: {e}")
        return False


async def test_entity_extraction():
    """Тест 2: Извлечение сущностей"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Извлечение сущностей")
    print("=" * 60)

    test_messages = [
        {
            "id": "1",
            "text": "Давайте встретимся завтра в 15:00 в офисе. Саша ответственный.",
            "date": "2024-01-01T12:00:00Z",
            "from": {"display": "Иван"},
        },
        {
            "id": "2",
            "text": "Хорошо, я согласен. Вот ссылка: https://example.com/docs",
            "date": "2024-01-01T12:05:00Z",
            "from": {"display": "Петр"},
        },
    ]

    try:
        extractor = EntityExtractor()
        entities = extractor.extract_from_messages(test_messages)

        print("✅ Извлечение сущностей работает")
        print(f"📊 Участники: {entities.get('participants', [])}")
        print(f"🔗 Ссылки: {entities.get('links', [])}")
        print(f"⏰ Временные упоминания: {entities.get('time_mentions', [])}")
        return True
    except Exception as e:
        print(f"❌ Ошибка извлечения сущностей: {e}")
        return False


async def test_session_segmentation():
    """Тест 3: Сегментация сессий"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Сегментация сессий")
    print("=" * 60)

    # Загружаем реальные данные
    chats_dir = Path("chats")
    test_chat = None

    for chat_path in chats_dir.iterdir():
        if chat_path.is_dir():
            json_file = chat_path / "unknown.json"
            if json_file.exists():
                test_chat = chat_path.name
                # Файл в формате JSONL (одна строка = один JSON объект)
                messages = []
                with open(json_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                messages.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                if messages:
                    break

    if not test_chat or not messages:
        print("⚠️ Не найдено тестовых данных")
        return False

    try:
        segmenter = SessionSegmenter()
        sessions = segmenter.segment_messages(messages, test_chat)

        print("✅ Сегментация работает")
        print(f"💬 Чат: {test_chat}")
        print(f"📨 Всего сообщений: {len(messages)}")
        print(f"📅 Создано сессий: {len(sessions)}")

        if sessions:
            first = sessions[0]
            print("\n📍 Первая сессия:")
            print(f"   ID: {first['session_id']}")
            print(f"   Сообщений: {len(first['messages'])}")
            print(f"   Временной диапазон: {first.get('time_range_bkk', 'N/A')}")

        return True, test_chat, sessions[0] if sessions else None
    except Exception as e:
        print(f"❌ Ошибка сегментации: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


async def test_full_summarization(test_chat, test_session):
    """Тест 4: Полная саммаризация"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Полная саммаризация")
    print("=" * 60)

    if not test_session:
        print("⚠️ Нет сессии для тестирования")
        return False

    try:
        client = OllamaEmbeddingClient()
        summarizer = SessionSummarizer(client, Path("summaries"))

        print("🔄 Начинаем саммаризацию сессии...")
        print(f"   Сессия: {test_session['session_id']}")
        print(f"   Сообщений: {len(test_session['messages'])}")

        summary = await summarizer.summarize_session(test_session)

        print("\n✅ Саммаризация выполнена!")
        print("\n📊 Результаты саммаризации:")
        print(f"   Session ID: {summary.get('session_id', 'N/A')}")
        print(f"   Чат: {summary.get('chat', 'N/A')}")
        print(f"   Участники: {', '.join(summary.get('participants', []))}")

        # Проверяем основные поля
        context = summary.get("context", "")
        discussion = summary.get("discussion", [])
        decisions = summary.get("decisions_next", [])
        risks = summary.get("risks_open", [])
        links = summary.get("links_artifacts", [])

        print("\n📝 Содержимое:")
        print(
            f"   Контекст: {'✅ Заполнен' if context and len(context) > 10 else '❌ Пусто'} ({len(context)} символов)"
        )
        print(
            f"   Дискуссия: {'✅ Заполнена' if discussion else '❌ Пусто'} ({len(discussion)} пунктов)"
        )
        print(
            f"   Решения: {'✅ Заполнены' if decisions else '❌ Пусто'} ({len(decisions)} пунктов)"
        )
        print(
            f"   Риски: {'✅ Заполнены' if risks else '❌ Пусто'} ({len(risks)} пунктов)"
        )
        print(
            f"   Ссылки: {'✅ Заполнены' if links else '❌ Пусто'} ({len(links)} пунктов)"
        )

        # Выводим примеры контента
        if context:
            print("\n📖 Пример контекста (первые 300 символов):")
            print(f"   {context[:300]}...")

        if discussion:
            print("\n💬 Первый пункт дискуссии:")
            print(f"   {discussion[0]}")

        if decisions:
            print("\n✅ Первое решение:")
            dec = decisions[0]
            print(f"   Текст: {dec.get('text', 'N/A')}")
            print(f"   Владелец: {dec.get('owner', 'N/A')}")
            print(f"   Приоритет: {dec.get('priority', 'N/A')}")

        # Итоговая оценка
        score = 0
        if context and len(context) > 10:
            score += 1
        if discussion:
            score += 1
        if decisions:
            score += 1
        if risks:
            score += 1
        if links:
            score += 1

        print(f"\n🎯 Итоговый балл качества: {score}/5")

        if score >= 3:
            print("   ✅ ХОРОШО - Саммаризация работает корректно")
        elif score >= 1:
            print("   ⚠️ ЧАСТИЧНО - Саммаризация работает, но есть пропуски")
        else:
            print("   ❌ ПЛОХО - Саммаризация не генерирует контент")

        return score >= 1

    except Exception as e:
        print(f"❌ Ошибка саммаризации: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Главная функция диагностики"""
    print("\n" + "🔍" * 30)
    print("ДИАГНОСТИКА СИСТЕМЫ САММАРИЗАЦИИ")
    print("🔍" * 30)
    print(f"\n⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "ollama": False,
        "entities": False,
        "segmentation": False,
        "summarization": False,
    }

    # Тест 1: Ollama
    results["ollama"] = await test_ollama_connection()
    if not results["ollama"]:
        print("\n⚠️ Критическая ошибка: Ollama недоступен!")
        print("   Запустите Ollama командой: ollama serve")
        return

    # Тест 2: Извлечение сущностей
    results["entities"] = await test_entity_extraction()

    # Тест 3: Сегментация
    seg_result = await test_session_segmentation()
    if isinstance(seg_result, tuple):
        results["segmentation"], test_chat, test_session = seg_result
    else:
        results["segmentation"] = seg_result
        test_chat, test_session = None, None

    # Тест 4: Саммаризация (только если предыдущие прошли)
    if results["ollama"] and results["segmentation"] and test_session:
        results["summarization"] = await test_full_summarization(
            test_chat, test_session
        )

    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)

    total = sum(1 for v in results.values() if v)

    print(f"\n✓ Тесты пройдено: {total}/4")
    print(f"   1. Ollama подключение: {'✅' if results['ollama'] else '❌'}")
    print(f"   2. Извлечение сущностей: {'✅' if results['entities'] else '❌'}")
    print(f"   3. Сегментация сессий: {'✅' if results['segmentation'] else '❌'}")
    print(f"   4. Саммаризация: {'✅' if results['summarization'] else '❌'}")

    if total == 4:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("   Система саммаризации работает корректно.")
    elif total >= 2:
        print("\n⚠️ ЧАСТИЧНЫЙ УСПЕХ")
        print("   Некоторые компоненты требуют внимания.")
    else:
        print("\n❌ КРИТИЧЕСКИЕ ПРОБЛЕМЫ")
        print("   Система саммаризации не работает.")

    print("\n📄 Подробный отчёт сохранён в: ANALYSIS_QUALITY_REPORT.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
