"""Утилиты для загрузки JSON и JSONL файлов.

Централизованные функции для чтения JSON и JSONL (newline-delimited JSON) файлов.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _try_fix_missing_commas(content: str, error: json.JSONDecodeError) -> dict | list | None:
    """
    Пытается исправить отсутствующие запятые между объектами в JSON массиве.
    
    Args:
        content: Содержимое файла
        error: Ошибка парсинга JSON
        
    Returns:
        Распарсенные данные или None при неудаче
    """
    if not content.strip().startswith('['):
        return None
    
    import re
    # Заменяем '}\s*{' на '},\n    {' для объектов верхнего уровня
    fixed_content = re.sub(r'\}\s*\n\s*\{', '},\n    {', content)
    try:
        return json.loads(fixed_content)
    except json.JSONDecodeError:
        return None


def _try_load_partial(content: str, error: json.JSONDecodeError) -> dict | list | None:
    """
    Пытается загрузить частичное содержимое до позиции ошибки.
    
    Args:
        content: Содержимое файла
        error: Ошибка парсинга JSON
        
    Returns:
        Распарсенные данные или None при неудаче
    """
    error_pos = getattr(error, 'pos', None)
    if not error_pos or error_pos <= 0:
        return None
    
    # Пытаемся найти последний валидный объект
    partial_content = content[:error_pos]
    # Находим последнюю закрывающую скобку объекта
    last_brace = partial_content.rfind('}')
    if last_brace <= 0:
        return None
    
    partial_content = partial_content[:last_brace + 1] + '\n]'
    try:
        return json.loads(partial_content)
    except json.JSONDecodeError:
        return None


def load_json_file(path: Path) -> dict | list | None:
    """
    Загружает JSON файл (обычный JSON, не JSONL).
    
    Если файл поврежден (например, отсутствуют запятые между объектами),
    пытается исправить и загрузить частично.

    Args:
        path: Путь к JSON файлу

    Returns:
        Загруженные данные (dict или list) или None при ошибке

    Examples:
        >>> data = load_json_file(Path("data.json"))
        >>> if data:
        ...     print(data)
    """
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError as e:
        logger.debug("Файл %s не является валидным JSON: %s", path, e)
        # Пытаемся загрузить частично, исправляя формат
        try:
            with open(path, encoding="utf-8") as fp:
                content = fp.read()
            
            # Сначала пытаемся исправить отсутствующие запятые
            result = _try_fix_missing_commas(content, e)
            if result is not None:
                return result
            
            # Если не получилось, пытаемся загрузить частично
            result = _try_load_partial(content, e)
            if result is not None:
                return result
        except Exception as repair_error:
            logger.debug("Не удалось исправить файл %s: %s", path, repair_error)
        return None
    except Exception as e:
        logger.error("Ошибка при загрузке JSON файла %s: %s", path, e)
        return None


def load_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    """
    Загружает JSONL файл (newline-delimited JSON).

    Каждая строка файла должна быть отдельным JSON объектом.

    Args:
        path: Путь к JSONL файлу

    Yields:
        Словари из каждой строки файла

    Examples:
        >>> for message in load_jsonl_file(Path("messages.jsonl")):
        ...     print(message)
    """
    try:
        with open(path, encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Некорректная JSON-строка в %s:%d — %s",
                        path,
                        line_number,
                        e,
                    )
                    continue
    except Exception as e:
        logger.error("Ошибка при загрузке JSONL файла %s: %s", path, e)


def load_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """
    Загружает файл как JSON или JSONL (автоопределение).

    Сначала пытается загрузить как обычный JSON. Если не получается,
    пробует загрузить как JSONL.

    Args:
        path: Путь к файлу

    Returns:
        Кортеж (список объектов, is_jsonl), где is_jsonl=True если файл был JSONL

    Examples:
        >>> messages, is_jsonl = load_json_or_jsonl(Path("data.json"))
        >>> if is_jsonl:
        ...     print("Файл был в формате JSONL")
    """
    # Пробуем загрузить как обычный JSON
    data = load_json_file(path)
    if data is not None:
        # Если это список, возвращаем как есть
        if isinstance(data, list):
            return data, False
        # Если это словарь, оборачиваем в список
        if isinstance(data, dict):
            return [data], False
        # Для других типов возвращаем пустой список
        logger.warning("Неожиданный тип данных в JSON файле %s: %s", path, type(data))
        return [], False

    # Если не получилось, пробуем как JSONL
    messages = list(load_jsonl_file(path))
    return messages, True


def load_json_file_safe(path: Path, default: dict | list | None = None) -> dict | list:
    """
    Безопасная загрузка JSON файла с значением по умолчанию.

    Args:
        path: Путь к JSON файлу
        default: Значение по умолчанию при ошибке

    Returns:
        Загруженные данные или default

    Examples:
        >>> config = load_json_file_safe(Path("config.json"), default={})
    """
    result = load_json_file(path)
    if result is None:
        return default if default is not None else {}
    return result

