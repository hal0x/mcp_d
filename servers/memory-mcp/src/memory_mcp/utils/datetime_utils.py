"""Утилиты для работы с датами и временем.

Централизованные функции для парсинга и нормализации временных меток.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def parse_datetime_utc(
    value: str | None,
    *,
    default: Optional[datetime] = None,
    return_none_on_error: bool = False,
    use_zoneinfo: bool = False,
) -> datetime | None:
    """
    Унифицированный парсинг дат в UTC.

    Поддерживает различные форматы ISO 8601 и автоматически нормализует в UTC.
    Обрабатывает строки с 'Z' суффиксом и без timezone информации.

    Args:
        value: Строка с датой/временем или None
        default: Значение по умолчанию при ошибке (если return_none_on_error=False)
        return_none_on_error: Если True, возвращает None при ошибке вместо default
        use_zoneinfo: Если True, использует ZoneInfo("UTC"), иначе timezone.utc

    Returns:
        datetime объект в UTC или None (если return_none_on_error=True и произошла ошибка)

    Examples:
        >>> parse_datetime_utc("2024-01-01T12:00:00Z")
        datetime.datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        >>> parse_datetime_utc("2024-01-01T12:00:00+03:00")
        datetime.datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        >>> parse_datetime_utc(None, default=datetime.now(timezone.utc))
        datetime.datetime(...)
    """
    if not value:
        if return_none_on_error:
            return None
        return default or datetime.now(timezone.utc if not use_zoneinfo else ZoneInfo("UTC"))

    try:
        # Обработка 'Z' суффикса (ISO 8601 UTC indicator)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        # Парсинг ISO формата
        dt = datetime.fromisoformat(value)

        # Нормализация в UTC
        if dt.tzinfo is None:
            # Если timezone не указан, считаем что это UTC
            dt = dt.replace(tzinfo=ZoneInfo("UTC") if use_zoneinfo else timezone.utc)
        else:
            # Конвертируем в UTC
            dt = dt.astimezone(ZoneInfo("UTC") if use_zoneinfo else timezone.utc)

        return dt

    except Exception as e:
        logger.debug("Не удалось распарсить временную метку %s: %s", value, e)
        if return_none_on_error:
            return None
        return default or datetime.now(timezone.utc if not use_zoneinfo else ZoneInfo("UTC"))


def parse_message_time(
    msg: dict,
    *,
    default: Optional[datetime] = None,
    use_zoneinfo: bool = True,
) -> datetime:
    """
    Парсит время из сообщения Telegram.

    Извлекает дату из полей 'date_utc' или 'date' и нормализует в UTC.

    Args:
        msg: Словарь с данными сообщения
        default: Значение по умолчанию при ошибке
        use_zoneinfo: Использовать ZoneInfo вместо timezone

    Returns:
        datetime объект в UTC

    Examples:
        >>> msg = {"date_utc": "2024-01-01T12:00:00Z", "id": "123"}
        >>> parse_message_time(msg)
        datetime.datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo('UTC'))
    """
    date_str = msg.get("date_utc") or msg.get("date", "")
    if not date_str:
        logger.warning("Сообщение без даты: %s", msg.get("id", "unknown"))
        return default or datetime.now(ZoneInfo("UTC") if use_zoneinfo else timezone.utc)

    result = parse_datetime_utc(
        date_str,
        default=default,
        return_none_on_error=False,
        use_zoneinfo=use_zoneinfo,
    )
    if result is None:
        # Fallback на default если что-то пошло не так
        return default or datetime.now(ZoneInfo("UTC") if use_zoneinfo else timezone.utc)
    return result


def parse_datetime_flexible(
    date_str: str,
    *,
    default: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Гибкий парсинг дат с поддержкой множества форматов.

    Пробует различные форматы через strptime, затем ISO формат.
    Используется для парсинга дат из разных источников.

    Args:
        date_str: Строка с датой
        default: Значение по умолчанию при ошибке

    Returns:
        datetime объект в UTC или None
    """
    if not date_str:
        return default

    # Список форматов для попытки парсинга
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]

    # Пробуем каждый формат
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            continue

    # Если ничего не сработало, пробуем ISO формат
    try:
        return parse_datetime_utc(date_str, default=default, return_none_on_error=True)
    except Exception:
        logger.debug("Не удалось распарсить дату: %s", date_str)
        return default


def format_datetime_display(
    dt: datetime | str | None,
    *,
    format_type: str = "default",
    timezone_name: Optional[str] = None,
    fallback: str = "—",
) -> str:
    """
    Форматирует datetime для отображения в пользовательском интерфейсе.

    Поддерживает различные предустановленные форматы и кастомные форматы через format_type.
    Автоматически конвертирует в указанную временную зону.

    Args:
        dt: datetime объект, ISO строка или None
        format_type: Тип формата:
            - "default" или "datetime": "%Y-%m-%d %H:%M"
            - "datetime_seconds": "%Y-%m-%d %H:%M:%S"
            - "date": "%Y-%m-%d"
            - "time": "%H:%M"
            - "time_seconds": "%H:%M:%S"
            - "short": "%d.%m %H:%M"
            - "iso": ISO формат
            - или кастомный формат strftime (например, "%Y-%m-%d %H:%M BKK")
        timezone_name: Название временной зоны (например, "Asia/Bangkok").
                      Если None, используется UTC.
        fallback: Строка для возврата при ошибке или None

    Returns:
        Отформатированная строка с датой/временем

    Examples:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        >>> format_datetime_display(dt)
        "2024-01-15 14:30"
        >>> format_datetime_display(dt, format_type="time")
        "14:30"
        >>> format_datetime_display(dt, format_type="date")
        "2024-01-15"
        >>> format_datetime_display("2024-01-15T14:30:00Z", format_type="datetime")
        "2024-01-15 14:30"
        >>> format_datetime_display(None)
        "—"
    """
    # Парсим входное значение, если это строка
    if isinstance(dt, str):
        parsed_dt = parse_datetime_utc(dt, return_none_on_error=True, use_zoneinfo=True)
        if parsed_dt is None:
            return fallback
        dt = parsed_dt
    elif dt is None:
        return fallback

    # Конвертируем в указанную временную зону
    if timezone_name:
        try:
            target_tz = ZoneInfo(timezone_name)
            dt = dt.astimezone(target_tz)
        except Exception as e:
            logger.debug(f"Ошибка конвертации в timezone {timezone_name}: {e}")

    # Определяем формат
    format_map = {
        "default": "%Y-%m-%d %H:%M",
        "datetime": "%Y-%m-%d %H:%M",
        "datetime_seconds": "%Y-%m-%d %H:%M:%S",
        "date": "%Y-%m-%d",
        "time": "%H:%M",
        "time_seconds": "%H:%M:%S",
        "short": "%d.%m %H:%M",
        "iso": None,  # Специальная обработка для ISO
    }

    if format_type in format_map:
        fmt = format_map[format_type]
        if fmt is None:  # ISO формат
            return dt.isoformat()
        return dt.strftime(fmt)
    else:
        # Кастомный формат strftime
        try:
            return dt.strftime(format_type)
        except Exception as e:
            logger.debug(f"Ошибка форматирования с форматом '{format_type}': {e}")
            return fallback

