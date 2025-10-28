#!/usr/bin/env python3
"""
Утилита для просмотра замененных URL в индексе
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import chromadb

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_replaced_urls_from_index(
    chroma_path: str = "./artifacts/chroma_db",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Получает все замененные URL из индекса

    Args:
        chroma_path: Путь к ChromaDB

    Returns:
        Словарь с замененными URL по чатам
    """
    client = chromadb.PersistentClient(path=chroma_path)

    # Получаем коллекции
    sessions_collection = client.get_collection("chat_sessions")
    messages_collection = client.get_collection("chat_messages")

    replaced_urls = {"sessions": {}, "messages": {}}

    # Проверяем сессии
    try:
        sessions_data = sessions_collection.get(include=["metadatas"])
        for i, metadata in enumerate(sessions_data["metadatas"]):
            if metadata.get("replaced_urls"):
                session_id = metadata.get("session_id", f"unknown_{i}")
                chat = metadata.get("chat", "unknown")
                if chat not in replaced_urls["sessions"]:
                    replaced_urls["sessions"][chat] = []

                replaced_urls["sessions"][chat].append(
                    {
                        "session_id": session_id,
                        "replaced_urls": metadata["replaced_urls"],
                        "message_count": metadata.get("message_count", 0),
                        "quality_score": metadata.get("quality_score", 0),
                    }
                )
    except Exception as e:
        print(f"Ошибка при получении данных сессий: {e}")

    # Проверяем сообщения
    try:
        messages_data = messages_collection.get(include=["metadatas"])
        for i, metadata in enumerate(messages_data["metadatas"]):
            if metadata.get("replaced_urls"):
                msg_id = metadata.get("msg_id", f"unknown_{i}")
                session_id = metadata.get("session_id", "unknown")
                chat = metadata.get("chat", "unknown")

                if chat not in replaced_urls["messages"]:
                    replaced_urls["messages"][chat] = []

                replaced_urls["messages"][chat].append(
                    {
                        "msg_id": msg_id,
                        "session_id": session_id,
                        "replaced_urls": metadata["replaced_urls"],
                        "date_utc": metadata.get("date_utc", ""),
                        "has_context": metadata.get("has_context", False),
                    }
                )
    except Exception as e:
        print(f"Ошибка при получении данных сообщений: {e}")

    return replaced_urls


def print_replaced_urls_summary(replaced_urls: Dict[str, List[Dict[str, Any]]]):
    """Выводит сводку по замененным URL"""

    print("🔍 Сводка по замененным URL в индексе")
    print("=" * 50)

    # Статистика по сессиям
    sessions_with_urls = sum(
        1 for chat_data in replaced_urls["sessions"].values() for _ in chat_data
    )
    total_sessions_urls = sum(
        len(session["replaced_urls"])
        for chat_data in replaced_urls["sessions"].values()
        for session in chat_data
    )

    print(f"📊 Сессии с замененными URL: {sessions_with_urls}")
    print(f"📊 Всего замененных URL в сессиях: {total_sessions_urls}")

    # Статистика по сообщениям
    messages_with_urls = sum(
        1 for chat_data in replaced_urls["messages"].values() for _ in chat_data
    )
    total_messages_urls = sum(
        len(msg["replaced_urls"])
        for chat_data in replaced_urls["messages"].values()
        for msg in chat_data
    )

    print(f"📊 Сообщения с замененными URL: {messages_with_urls}")
    print(f"📊 Всего замененных URL в сообщениях: {total_messages_urls}")

    print(f"\n📈 Общий итог: {total_sessions_urls + total_messages_urls} замененных URL")


def print_detailed_report(replaced_urls: Dict[str, List[Dict[str, Any]]]):
    """Выводит детальный отчет по замененным URL"""

    print("\n📋 Детальный отчет по чатам")
    print("=" * 50)

    all_chats = set(replaced_urls["sessions"].keys()) | set(
        replaced_urls["messages"].keys()
    )

    for chat in sorted(all_chats):
        print(f"\n💬 Чат: {chat}")

        # Сессии
        if chat in replaced_urls["sessions"]:
            print(
                f"  📁 Сессии с замененными URL: {len(replaced_urls['sessions'][chat])}"
            )
            for session in replaced_urls["sessions"][chat]:
                print(f"    - {session['session_id']}: {session['replaced_urls']}")
                print(
                    f"      Сообщений: {session['message_count']}, Качество: {session['quality_score']}"
                )

        # Сообщения
        if chat in replaced_urls["messages"]:
            print(
                f"  💬 Сообщения с замененными URL: {len(replaced_urls['messages'][chat])}"
            )
            for msg in replaced_urls["messages"][chat][
                :5
            ]:  # Показываем только первые 5
                print(f"    - {msg['msg_id']}: {msg['replaced_urls']}")
                print(f"      Дата: {msg['date_utc']}, Контекст: {msg['has_context']}")

            if len(replaced_urls["messages"][chat]) > 5:
                print(
                    f"    ... и еще {len(replaced_urls['messages'][chat]) - 5} сообщений"
                )


def get_unique_urls(replaced_urls: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Получает список уникальных замененных URL"""

    unique_urls = set()

    for chat_data in replaced_urls["sessions"].values():
        for session in chat_data:
            unique_urls.update(session["replaced_urls"])

    for chat_data in replaced_urls["messages"].values():
        for msg in chat_data:
            unique_urls.update(msg["replaced_urls"])

    return sorted(unique_urls)


def main():
    """Основная функция"""

    print("🔍 Анализ замененных URL в индексе Telegram Dump Manager")
    print("=" * 60)

    try:
        # Получаем данные
        replaced_urls = get_replaced_urls_from_index()

        # Выводим сводку
        print_replaced_urls_summary(replaced_urls)

        # Выводим детальный отчет
        print_detailed_report(replaced_urls)

        # Выводим уникальные URL
        unique_urls = get_unique_urls(replaced_urls)
        if unique_urls:
            print("\n🔗 Уникальные замененные URL:")
            print("=" * 30)
            for url in unique_urls:
                print(f"  - {url}")

        # Сохраняем отчет в файл
        report_file = Path("artifacts/reports/replaced_urls_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "sessions_with_urls": sum(
                            1
                            for chat_data in replaced_urls["sessions"].values()
                            for _ in chat_data
                        ),
                        "total_sessions_urls": sum(
                            len(session["replaced_urls"])
                            for chat_data in replaced_urls["sessions"].values()
                            for session in chat_data
                        ),
                        "messages_with_urls": sum(
                            1
                            for chat_data in replaced_urls["messages"].values()
                            for _ in chat_data
                        ),
                        "total_messages_urls": sum(
                            len(msg["replaced_urls"])
                            for chat_data in replaced_urls["messages"].values()
                            for msg in chat_data
                        ),
                        "unique_urls": unique_urls,
                    },
                    "details": replaced_urls,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n💾 Отчет сохранен в: {report_file}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
