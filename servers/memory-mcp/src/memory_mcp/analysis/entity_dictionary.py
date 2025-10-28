#!/usr/bin/env python3
"""
Модуль для автоматического обучения словарей сущностей
Отслеживает частоту появления терминов и автоматически добавляет их в словари
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Пороги для автоматического добавления в словари
ENTITY_THRESHOLDS = {
    "crypto_tokens": 3,      # Криптовалютные токены
    "persons": 5,            # Имена людей
    "organizations": 4,      # Организации
    "locations": 4,          # Места
    "telegram_channels": 2,  # Telegram каналы
    "telegram_bots": 2,      # Telegram боты
    "crypto_addresses": 2,   # Криптовалютные адреса
    "domains": 3,            # Домены
}

# Типы сущностей для отслеживания
ENTITY_TYPES = list(ENTITY_THRESHOLDS.keys())


class EntityDictionary:
    """Класс для автоматического обучения словарей сущностей"""

    def __init__(self, storage_path: Path = Path("config/entity_dictionaries")):
        """
        Инициализация словаря сущностей

        Args:
            storage_path: Путь к директории для хранения словарей
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Счетчики частоты появления сущностей
        self.entity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Счетчики по чатам для контекстного анализа
        self.chat_entity_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        
        # Загруженные словари
        self.learned_dictionaries: Dict[str, Set[str]] = {
            entity_type: set() for entity_type in ENTITY_TYPES
        }
        
        # Загружаем существующие словари
        self.load_dictionaries()

    def track_entity(self, entity_type: str, value: str, chat_name: str) -> bool:
        """
        Отслеживание появления сущности

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            chat_name: Название чата

        Returns:
            True если сущность добавлена в словарь, False иначе
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return False

        if not value or not value.strip():
            return False

        # Нормализуем значение
        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return False

        # Увеличиваем счетчики
        self.entity_counts[entity_type][normalized_value] += 1
        self.chat_entity_counts[chat_name][entity_type][normalized_value] += 1

        # Проверяем, нужно ли добавить в словарь
        threshold = ENTITY_THRESHOLDS[entity_type]
        total_count = self.entity_counts[entity_type][normalized_value]

        if total_count >= threshold and normalized_value not in self.learned_dictionaries[entity_type]:
            self.learned_dictionaries[entity_type].add(normalized_value)
            logger.info(f"Добавлена новая сущность в словарь: {entity_type}={normalized_value} (встречается {total_count} раз)")
            return True

        return False

    def is_known_entity(self, entity_type: str, value: str) -> bool:
        """
        Проверка, известна ли сущность

        Args:
            entity_type: Тип сущности
            value: Значение сущности

        Returns:
            True если сущность известна
        """
        if entity_type not in ENTITY_TYPES:
            return False

        normalized_value = self._normalize_entity_value(value)
        return normalized_value in self.learned_dictionaries[entity_type]

    def get_top_entities(self, entity_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Получение топ-N сущностей по частоте

        Args:
            entity_type: Тип сущности
            limit: Максимальное количество

        Returns:
            Список словарей с сущностями и их частотами
        """
        if entity_type not in ENTITY_TYPES:
            return []

        entities = self.entity_counts[entity_type]
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)

        result = []
        for value, count in sorted_entities[:limit]:
            result.append({
                "value": value,
                "count": count,
                "is_learned": value in self.learned_dictionaries[entity_type],
                "threshold": ENTITY_THRESHOLDS[entity_type]
            })

        return result

    def get_entity_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по словарям

        Returns:
            Словарь со статистикой
        """
        stats = {}
        for entity_type in ENTITY_TYPES:
            total_tracked = len(self.entity_counts[entity_type])
            learned_count = len(self.learned_dictionaries[entity_type])
            threshold = ENTITY_THRESHOLDS[entity_type]

            stats[entity_type] = {
                "total_tracked": total_tracked,
                "learned_count": learned_count,
                "threshold": threshold,
                "candidates": total_tracked - learned_count
            }

        return stats

    def get_chat_entity_stats(self, chat_name: str) -> Dict[str, Any]:
        """
        Получение статистики по сущностям конкретного чата

        Args:
            chat_name: Название чата

        Returns:
            Словарь со статистикой по чату
        """
        if chat_name not in self.chat_entity_counts:
            return {}

        chat_stats = {}
        for entity_type in ENTITY_TYPES:
            entities = self.chat_entity_counts[chat_name][entity_type]
            if entities:
                top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]
                chat_stats[entity_type] = {
                    "total_unique": len(entities),
                    "top_entities": [{"value": val, "count": cnt} for val, cnt in top_entities]
                }

        return chat_stats

    def save_dictionaries(self) -> None:
        """Сохранение словарей в файлы"""
        try:
            # Сохраняем счетчики частоты
            counts_file = self.storage_path / "entity_counts.json"
            with open(counts_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.entity_counts), f, ensure_ascii=False, indent=2)

            # Сохраняем счетчики по чатам
            chat_counts_file = self.storage_path / "chat_entity_counts.json"
            with open(chat_counts_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.chat_entity_counts), f, ensure_ascii=False, indent=2)

            # Сохраняем обученные словари
            for entity_type in ENTITY_TYPES:
                dict_file = self.storage_path / f"{entity_type}.json"
                entities_list = sorted(list(self.learned_dictionaries[entity_type]))
                with open(dict_file, 'w', encoding='utf-8') as f:
                    json.dump(entities_list, f, ensure_ascii=False, indent=2)

            logger.info(f"Словари сохранены в {self.storage_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении словарей: {e}")

    def load_dictionaries(self) -> None:
        """Загрузка словарей из файлов"""
        try:
            # Загружаем счетчики частоты
            counts_file = self.storage_path / "entity_counts.json"
            if counts_file.exists():
                with open(counts_file, 'r', encoding='utf-8') as f:
                    counts_data = json.load(f)
                    for entity_type, entities in counts_data.items():
                        if entity_type in ENTITY_TYPES:
                            self.entity_counts[entity_type].update(entities)

            # Загружаем счетчики по чатам
            chat_counts_file = self.storage_path / "chat_entity_counts.json"
            if chat_counts_file.exists():
                with open(chat_counts_file, 'r', encoding='utf-8') as f:
                    chat_counts_data = json.load(f)
                    for chat_name, chat_data in chat_counts_data.items():
                        for entity_type, entities in chat_data.items():
                            if entity_type in ENTITY_TYPES:
                                self.chat_entity_counts[chat_name][entity_type].update(entities)

            # Загружаем обученные словари
            for entity_type in ENTITY_TYPES:
                dict_file = self.storage_path / f"{entity_type}.json"
                if dict_file.exists():
                    with open(dict_file, 'r', encoding='utf-8') as f:
                        entities_list = json.load(f)
                        self.learned_dictionaries[entity_type] = set(entities_list)

            logger.info(f"Словари загружены из {self.storage_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке словарей: {e}")

    def _normalize_entity_value(self, value: str) -> Optional[str]:
        """
        Нормализация значения сущности

        Args:
            value: Исходное значение

        Returns:
            Нормализованное значение или None
        """
        if not value:
            return None

        # Базовая нормализация
        normalized = value.strip().lower()

        # Убираем лишние символы
        normalized = normalized.replace('@', '').replace('#', '')

        # Ограничиваем длину
        if len(normalized) > 100:
            normalized = normalized[:100]

        # Проверяем, что значение не пустое после нормализации
        if not normalized or len(normalized) < 2:
            return None

        return normalized

    def export_dictionary(self, entity_type: str, format: str = "json") -> Optional[str]:
        """
        Экспорт словаря в различных форматах

        Args:
            entity_type: Тип сущности
            format: Формат экспорта (json, txt, csv)

        Returns:
            Экспортированные данные или None
        """
        if entity_type not in ENTITY_TYPES:
            return None

        entities = sorted(list(self.learned_dictionaries[entity_type]))

        if format == "json":
            return json.dumps(entities, ensure_ascii=False, indent=2)
        elif format == "txt":
            return "\n".join(entities)
        elif format == "csv":
            return f"entity_type,value\n" + "\n".join(f"{entity_type},{entity}" for entity in entities)
        else:
            return None

    def clear_dictionary(self, entity_type: str) -> None:
        """
        Очистка словаря определенного типа

        Args:
            entity_type: Тип сущности для очистки
        """
        if entity_type in ENTITY_TYPES:
            self.learned_dictionaries[entity_type].clear()
            self.entity_counts[entity_type].clear()
            logger.info(f"Словарь {entity_type} очищен")

    def reset_all(self) -> None:
        """Сброс всех словарей и счетчиков"""
        for entity_type in ENTITY_TYPES:
            self.learned_dictionaries[entity_type].clear()
            self.entity_counts[entity_type].clear()
        
        self.chat_entity_counts.clear()
        logger.info("Все словари и счетчики сброшены")


# Глобальный экземпляр для использования в других модулях
_global_entity_dictionary: Optional[EntityDictionary] = None


def get_entity_dictionary() -> EntityDictionary:
    """Получение глобального экземпляра словаря сущностей"""
    global _global_entity_dictionary
    if _global_entity_dictionary is None:
        _global_entity_dictionary = EntityDictionary()
    return _global_entity_dictionary


if __name__ == "__main__":
    # Тест модуля
    dict_manager = EntityDictionary()
    
    # Тестируем отслеживание сущностей
    test_entities = [
        ("crypto_tokens", "BTC", "test_chat"),
        ("crypto_tokens", "ETH", "test_chat"),
        ("crypto_tokens", "BTC", "test_chat"),
        ("crypto_tokens", "BTC", "test_chat"),  # Должен быть добавлен в словарь
        ("persons", "Алексей", "test_chat"),
        ("persons", "Алексей", "test_chat"),
        ("persons", "Алексей", "test_chat"),
        ("persons", "Алексей", "test_chat"),
        ("persons", "Алексей", "test_chat"),  # Должен быть добавлен в словарь
    ]
    
    for entity_type, value, chat_name in test_entities:
        dict_manager.track_entity(entity_type, value, chat_name)
    
    # Показываем статистику
    print("Статистика словарей:")
    stats = dict_manager.get_entity_stats()
    for entity_type, stat in stats.items():
        print(f"  {entity_type}: {stat}")
    
    # Показываем топ сущности
    print("\nТоп криптовалютные токены:")
    top_crypto = dict_manager.get_top_entities("crypto_tokens", 5)
    for entity in top_crypto:
        print(f"  {entity['value']}: {entity['count']} раз")
    
    # Сохраняем словари
    dict_manager.save_dictionaries()
