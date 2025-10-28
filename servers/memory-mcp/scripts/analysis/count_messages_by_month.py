#!/usr/bin/env python3
"""
Скрипт для подсчета сообщений по месяцам в чатах
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_date(date_str):
    """Парсит дату из строки в формате ISO 8601"""
    try:
        # Заменяем Z на +00:00 для UTC
        if date_str.endswith("Z"):
            date_str = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def count_messages_by_month(file_path):
    """
    Подсчитывает сообщения по месяцам в файле чата

    Returns:
        dict: словарь с количеством сообщений по месяцам (YYYY-MM)
    """
    monthly_counts = Counter()

    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        message = json.loads(line)
                        if "date_utc" in message:
                            msg_date = parse_date(message["date_utc"])
                            if msg_date:
                                month_key = msg_date.strftime("%Y-%m")
                                monthly_counts[month_key] += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")

    return monthly_counts


def main():
    parser = argparse.ArgumentParser(description="Подсчет сообщений по месяцам в чатах")
    parser.add_argument(
        "--chats-dir",
        default="./chats",
        help="Путь к директории с чатами (по умолчанию: ./chats)",
    )
    parser.add_argument(
        "--output",
        choices=["table", "csv", "json"],
        default="table",
        help="Формат вывода (по умолчанию: table)",
    )
    parser.add_argument(
        "--sort",
        choices=["month", "count"],
        default="month",
        help="Сортировка по месяцу или количеству (по умолчанию: month)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Показать только топ N месяцев по количеству сообщений",
    )

    args = parser.parse_args()

    chats_dir = Path(args.chats_dir)
    if not chats_dir.exists():
        print(f"Ошибка: директория '{chats_dir}' не существует")
        sys.exit(1)

    print("Подсчет сообщений по месяцам")
    print(f"Директория чатов: {chats_dir}")
    print("-" * 60)

    # Находим все файлы unknown.json
    json_files = list(chats_dir.glob("**/unknown.json"))

    if not json_files:
        print("Не найдено файлов unknown.json")
        return

    # Общий счетчик по месяцам
    total_monthly_counts = Counter()
    processed_files = 0

    # Обрабатываем каждый файл
    for json_file in json_files:
        monthly_counts = count_messages_by_month(json_file)

        # Добавляем к общему счетчику
        for month, count in monthly_counts.items():
            total_monthly_counts[month] += count

        processed_files += 1

    print(f"Обработано файлов: {processed_files}")
    print(f"Всего месяцев с сообщениями: {len(total_monthly_counts)}")
    print("-" * 60)

    # Подготавливаем данные для вывода
    if args.sort == "month":
        sorted_data = sorted(total_monthly_counts.items())
    else:  # sort by count
        sorted_data = sorted(
            total_monthly_counts.items(), key=lambda x: x[1], reverse=True
        )

    # Ограничиваем вывод если нужно
    if args.top:
        sorted_data = sorted_data[: args.top]

    # Выводим результаты
    if args.output == "table":
        print(f"{'Месяц':<10} | {'Сообщений':<10} | {'Процент':<8}")
        print("-" * 35)

        total_messages = sum(total_monthly_counts.values())

        for month, count in sorted_data:
            percentage = (count / total_messages) * 100 if total_messages > 0 else 0
            print(f"{month:<10} | {count:<10} | {percentage:>6.1f}%")

        print("-" * 35)
        print(f"{'ИТОГО':<10} | {total_messages:<10} | {'100.0%':<8}")

    elif args.output == "csv":
        print("Месяц,Сообщений")
        for month, count in sorted_data:
            print(f"{month},{count}")

    elif args.output == "json":
        result = {
            "total_files_processed": processed_files,
            "total_months": len(total_monthly_counts),
            "total_messages": sum(total_monthly_counts.values()),
            "monthly_counts": dict(sorted_data),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
