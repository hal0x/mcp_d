"""Утилита для управления состоянием (загрузка/сохранение JSON)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from .datetime_utils import parse_datetime_utc

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="StateBase")


class StateBase:
    """Базовый класс для состояний с сериализацией/десериализацией."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация состояния в словарь.

        Должен быть переопределен в подклассах.
        """
        raise NotImplementedError("Subclasses must implement to_dict()")

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Десериализация состояния из словаря.

        Должен быть переопределен в подклассах.
        """
        raise NotImplementedError("Subclasses must implement from_dict()")


class StateManager:
    """Менеджер для загрузки и сохранения состояния в JSON файлы."""

    def __init__(self, state_dir: Path):
        """
        Инициализация менеджера состояния.

        Args:
            state_dir: Директория для хранения файлов состояния
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load_state(
        self,
        state_id: str,
        state_class: Type[T],
        default_factory: Optional[callable] = None,
    ) -> T:
        """
        Загружает состояние из JSON файла.

        Args:
            state_id: Идентификатор состояния (имя файла без расширения)
            state_class: Класс состояния (должен иметь метод from_dict)
            default_factory: Фабрика для создания состояния по умолчанию (если файл не найден)

        Returns:
            Экземпляр состояния

        Example:
            >>> manager = StateManager(Path("state"))
            >>> state = manager.load_state("chat1", AggregationState, lambda: AggregationState("chat1"))
        """
        state_file = self.state_dir / f"{state_id}.json"

        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    data = json.load(f)
                return state_class.from_dict(data)
            except Exception as e:
                logger.warning(f"Не удалось загрузить состояние для {state_id}: {e}")

        # Создаем состояние по умолчанию
        if default_factory:
            return default_factory()
        raise ValueError(f"Не удалось загрузить состояние для {state_id} и не указана default_factory")

    def save_state(self, state: StateBase, state_id: Optional[str] = None):
        """
        Сохраняет состояние в JSON файл.

        Args:
            state: Экземпляр состояния (должен иметь метод to_dict)
            state_id: Идентификатор состояния (если None, берется из state.chat_name или state.state_id)

        Example:
            >>> manager = StateManager(Path("state"))
            >>> state = AggregationState("chat1")
            >>> manager.save_state(state)
        """
        # Определяем идентификатор состояния
        if state_id is None:
            # Пытаемся получить из атрибута chat_name или state_id
            state_id = getattr(state, "chat_name", None) or getattr(state, "state_id", None)
            if state_id is None:
                raise ValueError("Не указан state_id и не найден атрибут chat_name или state_id")

        state_file = self.state_dir / f"{state_id}.json"

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Сохранено состояние для {state_id}")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния для {state_id}: {e}")

    def state_exists(self, state_id: str) -> bool:
        """
        Проверяет существование файла состояния.

        Args:
            state_id: Идентификатор состояния

        Returns:
            True если файл существует
        """
        state_file = self.state_dir / f"{state_id}.json"
        return state_file.exists()

    def delete_state(self, state_id: str) -> bool:
        """
        Удаляет файл состояния.

        Args:
            state_id: Идентификатор состояния

        Returns:
            True если файл был удален
        """
        state_file = self.state_dir / f"{state_id}.json"
        if state_file.exists():
            try:
                state_file.unlink()
                logger.debug(f"Удалено состояние для {state_id}")
                return True
            except Exception as e:
                logger.error(f"Ошибка удаления состояния для {state_id}: {e}")
                return False
        return False

    def list_states(self) -> list[str]:
        """
        Возвращает список всех идентификаторов состояний.

        Returns:
            Список идентификаторов (имена файлов без расширения .json)
        """
        states = []
        for state_file in self.state_dir.glob("*.json"):
            states.append(state_file.stem)
        return sorted(states)


def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """
    Сериализует datetime в ISO строку.

    Args:
        dt: datetime объект или None

    Returns:
        ISO строка или None
    """
    if dt is None:
        return None
    return dt.isoformat()


def deserialize_datetime(iso_str: Optional[str], default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Десериализует ISO строку в datetime.

    Args:
        iso_str: ISO строка или None
        default: Значение по умолчанию при ошибке

    Returns:
        datetime объект или None
    """
    if not iso_str:
        return default
    return parse_datetime_utc(iso_str, default=default, return_none_on_error=True, use_zoneinfo=True)


def serialize_datetime_dict(dt_dict: Dict[str, datetime]) -> Dict[str, str]:
    """
    Сериализует словарь с datetime значениями.

    Args:
        dt_dict: Словарь с datetime значениями

    Returns:
        Словарь с ISO строками
    """
    return {k: serialize_datetime(v) for k, v in dt_dict.items()}


def deserialize_datetime_dict(
    iso_dict: Dict[str, str], default: Optional[datetime] = None
) -> Dict[str, datetime]:
    """
    Десериализует словарь с ISO строками в datetime.

    Args:
        iso_dict: Словарь с ISO строками
        default: Значение по умолчанию при ошибке

    Returns:
        Словарь с datetime значениями
    """
    result = {}
    for k, v in iso_dict.items():
        dt = deserialize_datetime(v, default)
        if dt is not None:
            result[k] = dt
    return result

