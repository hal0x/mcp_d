"""Тесты для модуля datetime_utils."""

import pytest
from datetime import datetime, timezone

from memory_mcp.utils.processing.datetime_utils import format_datetime_display


class TestFormatDatetimeDisplay:
    """Тесты для функции format_datetime_display."""

    def test_format_default(self):
        """Тест: форматирование с форматом по умолчанию."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt)
        assert result == "2024-01-15 14:30"

    def test_format_datetime(self):
        """Тест: форматирование с форматом datetime."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="datetime")
        assert result == "2024-01-15 14:30"

    def test_format_datetime_seconds(self):
        """Тест: форматирование с секундами."""
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="datetime_seconds")
        assert result == "2024-01-15 14:30:45"

    def test_format_date(self):
        """Тест: форматирование только даты."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="date")
        assert result == "2024-01-15"

    def test_format_time(self):
        """Тест: форматирование только времени."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="time")
        assert result == "14:30"

    def test_format_time_seconds(self):
        """Тест: форматирование времени с секундами."""
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="time_seconds")
        assert result == "14:30:45"

    def test_format_short(self):
        """Тест: короткий формат."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="short")
        assert result == "15.01 14:30"

    def test_format_iso(self):
        """Тест: ISO формат."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="iso")
        assert "2024-01-15T14:30:00" in result

    def test_format_custom(self):
        """Тест: кастомный формат strftime."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = format_datetime_display(dt, format_type="%Y-%m-%d %H:%M BKK")
        assert result == "2024-01-15 14:30 BKK"

    def test_format_from_string(self):
        """Тест: форматирование из ISO строки."""
        result = format_datetime_display("2024-01-15T14:30:00Z")
        assert result == "2024-01-15 14:30"

    def test_format_with_timezone(self):
        """Тест: форматирование с конвертацией timezone."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        # Asia/Bangkok = UTC+7
        result = format_datetime_display(dt, format_type="time", timezone_name="Asia/Bangkok")
        assert result == "21:30"  # 14:30 UTC = 21:30 BKK

    def test_format_none(self):
        """Тест: форматирование None."""
        result = format_datetime_display(None)
        assert result == "—"

    def test_format_none_custom_fallback(self):
        """Тест: форматирование None с кастомным fallback."""
        result = format_datetime_display(None, fallback="N/A")
        assert result == "N/A"

    def test_format_invalid_string(self):
        """Тест: форматирование невалидной строки."""
        result = format_datetime_display("invalid-date")
        assert result == "—"

    def test_format_invalid_string_custom_fallback(self):
        """Тест: форматирование невалидной строки с кастомным fallback."""
        result = format_datetime_display("invalid-date", fallback="??:??")
        assert result == "??:??"

    def test_format_invalid_timezone(self):
        """Тест: форматирование с невалидной timezone."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        # Должно вернуть результат без конвертации timezone
        result = format_datetime_display(dt, format_type="time", timezone_name="Invalid/Timezone")
        assert result == "14:30"  # Fallback на UTC

