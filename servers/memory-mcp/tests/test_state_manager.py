"""Тесты для модуля state_manager."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from memory_mcp.utils.system.state_manager import (
    StateBase,
    StateManager,
    deserialize_datetime,
    deserialize_datetime_dict,
    serialize_datetime,
    serialize_datetime_dict,
)


class TestState(StateBase):
    """Тестовый класс состояния."""

    def __init__(self, name: str, value: int = 0, timestamp: datetime | None = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        """Сериализация."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": serialize_datetime(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestState":
        """Десериализация."""
        return cls(
            name=data["name"],
            value=data.get("value", 0),
            timestamp=deserialize_datetime(data.get("timestamp")),
        )


class TestStateManager:
    """Тесты для StateManager."""

    def test_save_and_load_state(self):
        """Тест: сохранение и загрузка состояния."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            state = TestState("test1", value=42, timestamp=datetime.now(timezone.utc))

            # Сохраняем
            manager.save_state(state, "test1")

            # Загружаем
            loaded = manager.load_state("test1", TestState, default_factory=lambda: TestState("default"))

            assert loaded.name == "test1"
            assert loaded.value == 42
            assert loaded.timestamp is not None

    def test_load_nonexistent_state(self):
        """Тест: загрузка несуществующего состояния."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            # Загружаем с default_factory
            loaded = manager.load_state("nonexistent", TestState, default_factory=lambda: TestState("default", 0))
            assert loaded.name == "default"
            assert loaded.value == 0

    def test_load_nonexistent_state_no_factory(self):
        """Тест: загрузка несуществующего состояния без factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            with pytest.raises(ValueError, match="Не удалось загрузить состояние"):
                manager.load_state("nonexistent", TestState)

    def test_state_exists(self):
        """Тест: проверка существования состояния."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            assert not manager.state_exists("test1")

            state = TestState("test1")
            manager.save_state(state, "test1")

            assert manager.state_exists("test1")

    def test_delete_state(self):
        """Тест: удаление состояния."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            state = TestState("test1")
            manager.save_state(state, "test1")

            assert manager.state_exists("test1")

            deleted = manager.delete_state("test1")
            assert deleted is True
            assert not manager.state_exists("test1")

    def test_delete_nonexistent_state(self):
        """Тест: удаление несуществующего состояния."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            deleted = manager.delete_state("nonexistent")
            assert deleted is False

    def test_list_states(self):
        """Тест: список состояний."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            assert manager.list_states() == []

            manager.save_state(TestState("test1"), "test1")
            manager.save_state(TestState("test2"), "test2")
            manager.save_state(TestState("test3"), "test3")

            states = manager.list_states()
            assert len(states) == 3
            assert "test1" in states
            assert "test2" in states
            assert "test3" in states
            assert states == sorted(states)  # Должны быть отсортированы

    def test_save_state_with_chat_name(self):
        """Тест: сохранение состояния с атрибутом chat_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            state = TestState("test1")
            state.chat_name = "chat1"  # Добавляем атрибут chat_name

            manager.save_state(state)  # Без указания state_id

            assert manager.state_exists("chat1")

    def test_save_state_with_state_id(self):
        """Тест: сохранение состояния с атрибутом state_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            state = TestState("test1")
            state.state_id = "custom_id"  # Добавляем атрибут state_id

            manager.save_state(state)  # Без указания state_id

            assert manager.state_exists("custom_id")

    def test_save_state_no_id(self):
        """Тест: сохранение состояния без идентификатора."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            state = TestState("test1")
            # Нет chat_name и state_id

            with pytest.raises(ValueError, match="Не указан state_id"):
                manager.save_state(state)


class TestSerializeDatetime:
    """Тесты для функций сериализации datetime."""

    def test_serialize_datetime(self):
        """Тест: сериализация datetime."""
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = serialize_datetime(dt)
        assert result == "2024-01-15T14:30:00+00:00"

    def test_serialize_datetime_none(self):
        """Тест: сериализация None."""
        assert serialize_datetime(None) is None

    def test_deserialize_datetime(self):
        """Тест: десериализация datetime."""
        iso_str = "2024-01-15T14:30:00Z"
        result = deserialize_datetime(iso_str)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_deserialize_datetime_none(self):
        """Тест: десериализация None."""
        assert deserialize_datetime(None) is None
        assert deserialize_datetime("") is None

    def test_deserialize_datetime_default(self):
        """Тест: десериализация с default."""
        default = datetime.now(timezone.utc)
        result = deserialize_datetime(None, default=default)
        assert result == default

    def test_serialize_datetime_dict(self):
        """Тест: сериализация словаря с datetime."""
        dt_dict = {
            "start": datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
            "end": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        result = serialize_datetime_dict(dt_dict)
        assert "start" in result
        assert "end" in result
        assert isinstance(result["start"], str)
        assert isinstance(result["end"], str)

    def test_deserialize_datetime_dict(self):
        """Тест: десериализация словаря с ISO строками."""
        iso_dict = {
            "start": "2024-01-15T14:30:00Z",
            "end": "2024-01-15T15:30:00Z",
        }
        result = deserialize_datetime_dict(iso_dict)
        assert "start" in result
        assert "end" in result
        assert isinstance(result["start"], datetime)
        assert isinstance(result["end"], datetime)

    def test_deserialize_datetime_dict_invalid(self):
        """Тест: десериализация словаря с невалидными значениями."""
        iso_dict = {
            "start": "2024-01-15T14:30:00Z",
            "end": None,
            "invalid": "not-a-date",
        }
        result = deserialize_datetime_dict(iso_dict)
        assert "start" in result
        assert "end" not in result  # None пропускается
        assert "invalid" not in result  # Невалидное значение пропускается

