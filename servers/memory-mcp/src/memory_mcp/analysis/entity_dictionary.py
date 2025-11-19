#!/usr/bin/env python3
"""Автоматическое обучение словарей сущностей с LLM-валидацией."""

import asyncio
import concurrent.futures
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

ENTITY_THRESHOLDS = {
    "crypto_tokens": 3,
    "persons": 5,
    "organizations": 4,
    "locations": 4,
    "telegram_channels": 2,
    "telegram_bots": 2,
    "crypto_addresses": 2,
    "domains": 3,
}

ENTITY_TYPES = list(ENTITY_THRESHOLDS.keys())


class EntityDictionary:
    """Автоматическое обучение словарей сущностей с LLM-валидацией.
    
    Отслеживает частоту появления терминов и добавляет их в словари после валидации.
    """

    def __init__(
        self, 
        storage_path: Path = Path("config/entity_dictionaries"),
        enable_llm_validation: bool = True,
        llm_client: Optional[Any] = None,
        batch_validation_size: int = 10,
        enable_description_generation: bool = True,
        graph: Optional[Any] = None,
    ):
        """Инициализирует словарь сущностей.

        Args:
            storage_path: Путь к директории для хранения словарей
            enable_llm_validation: Включить валидацию через LLM
            llm_client: Клиент для LLM (опционально, создается автоматически если не указан)
            batch_validation_size: Размер батча для валидации сущностей (по умолчанию 10)
            enable_description_generation: Включить генерацию описаний сущностей (по умолчанию True)
            graph: Граф памяти для доступа к контексту сообщений (опционально)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.entity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.chat_entity_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.username_to_names: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.name_to_usernames: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.learned_dictionaries: Dict[str, Set[str]] = {
            entity_type: set() for entity_type in ENTITY_TYPES
        }
        self.entity_descriptions: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.enable_llm_validation = enable_llm_validation
        self.enable_description_generation = enable_description_generation
        self._llm_client = llm_client
        self._llm_client_initialized = False
        self.batch_validation_size = batch_validation_size
        self._validation_queue: List[Dict[str, Any]] = []
        self.graph = graph
        
        self.load_dictionaries()

    def link_username_to_name(self, chat_name: str, username: str, display_name: str) -> None:
        """Связывает никнейм с отображаемым именем пользователя в чате.
        
        Args:
            chat_name: Название чата
            username: Никнейм пользователя (без @)
            display_name: Отображаемое имя пользователя
        """
        if not username or not display_name:
            return
        
        username_normalized = username.lower().strip().replace('@', '')
        display_name_normalized = display_name.strip()
        
        if username_normalized and display_name_normalized:
            self.username_to_names[chat_name][username_normalized].add(display_name_normalized)
            self.name_to_usernames[chat_name][display_name_normalized].add(username_normalized)
            logger.debug(f"Связан никнейм @{username_normalized} с именем '{display_name_normalized}' в чате '{chat_name}'")

    def is_username_in_chat(self, chat_name: str, value: str) -> bool:
        """Проверяет, является ли значение никнеймом пользователя из чата.
        
        Args:
            chat_name: Название чата
            value: Проверяемое значение
            
        Returns:
            True если это никнейм пользователя из чата
        """
        normalized_value = value.lower().strip().replace('@', '')
        return normalized_value in self.username_to_names.get(chat_name, {})

    def track_entity(self, entity_type: str, value: str, chat_name: str, author_username: Optional[str] = None, author_display_name: Optional[str] = None, use_batch: bool = True) -> bool:
        """Отслеживает появление сущности и добавляет в словарь при достижении порога.

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            chat_name: Название чата
            author_username: Никнейм автора сообщения (опционально, для связывания)
            author_display_name: Отображаемое имя автора (опционально, для связывания)
            use_batch: Использовать батч-валидацию (по умолчанию True)

        Returns:
            True если сущность добавлена в словарь, False иначе
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return False

        if not value or not value.strip():
            return False

        if author_username and author_display_name:
            self.link_username_to_name(chat_name, author_username, author_display_name)

        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return False
        self.entity_counts[entity_type][normalized_value] += 1
        self.chat_entity_counts[chat_name][entity_type][normalized_value] += 1

        threshold = ENTITY_THRESHOLDS[entity_type]
        total_count = self.entity_counts[entity_type][normalized_value]

        if total_count >= threshold and normalized_value not in self.learned_dictionaries[entity_type]:
            # Предварительная валидация (быстрая проверка на стоп-слова и явно неправильные сущности)
            if not self._prevalidate_entity(entity_type, normalized_value, value):
                logger.debug(
                    f"Сущность отклонена предварительной проверкой: {entity_type}={normalized_value} "
                    f"(встречается {total_count} раз, но не прошла предварительную валидацию)"
                )
                return False

            # Валидация через LLM перед добавлением в словарь
            if self.enable_llm_validation:
                if use_batch:
                    # Добавляем в очередь для батч-валидации
                    self._validation_queue.append({
                        "entity_type": entity_type,
                        "normalized_value": normalized_value,
                        "original_value": value,
                        "chat_name": chat_name,
                    })
                    # Не добавляем сразу, вернем False
                    # Сущность будет добавлена после flush_validation_queue
                    return False
                else:
                    # Старый способ - немедленная валидация
                    is_valid = self._validate_entity_with_llm(entity_type, normalized_value, value, chat_name)
                    if not is_valid:
                        logger.debug(
                            f"Сущность отклонена LLM: {entity_type}={normalized_value} "
                            f"(встречается {total_count} раз, но не прошла валидацию)"
                        )
                        return False
            
            self.learned_dictionaries[entity_type].add(normalized_value)
            logger.info(f"Добавлена новая сущность в словарь: {entity_type}={normalized_value} (встречается {total_count} раз)")
            
            # Генерируем описание, если включено (в фоновом режиме, не блокируем)
            if self.enable_description_generation:
                # Добавляем в очередь для генерации описаний (будет обработано позже)
                # Описание будет сгенерировано при следующем flush или при обновлении узла в графе
                pass
            
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
            # Убеждаемся, что директория существует и доступна для записи
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем счетчики частоты
            counts_file = self.storage_path / "entity_counts.json"
            try:
                with open(counts_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(self.entity_counts), f, ensure_ascii=False, indent=2)
            except PermissionError as e:
                logger.error(f"Ошибка прав доступа при сохранении {counts_file}: {e}")
                # Не прерываем выполнение, продолжаем с другими файлами

            # Сохраняем счетчики по чатам
            chat_counts_file = self.storage_path / "chat_entity_counts.json"
            try:
                with open(chat_counts_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(self.chat_entity_counts), f, ensure_ascii=False, indent=2)
            except PermissionError as e:
                logger.error(f"Ошибка прав доступа при сохранении {chat_counts_file}: {e}")

            # Сохраняем обученные словари
            for entity_type in ENTITY_TYPES:
                dict_file = self.storage_path / f"{entity_type}.json"
                entities_list = sorted(list(self.learned_dictionaries[entity_type]))
                try:
                    with open(dict_file, 'w', encoding='utf-8') as f:
                        json.dump(entities_list, f, ensure_ascii=False, indent=2)
                except PermissionError as e:
                    logger.error(f"Ошибка прав доступа при сохранении {dict_file}: {e}")

            # Сохраняем описания сущностей
            descriptions_file = self.storage_path / "entity_descriptions.json"
            try:
                with open(descriptions_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(self.entity_descriptions), f, ensure_ascii=False, indent=2)
            except PermissionError as e:
                logger.error(f"Ошибка прав доступа при сохранении {descriptions_file}: {e}")

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

            # Загружаем описания сущностей
            descriptions_file = self.storage_path / "entity_descriptions.json"
            if descriptions_file.exists():
                with open(descriptions_file, 'r', encoding='utf-8') as f:
                    descriptions_data = json.load(f)
                    for entity_type, descriptions in descriptions_data.items():
                        if entity_type in ENTITY_TYPES:
                            self.entity_descriptions[entity_type].update(descriptions)

            logger.info(f"Словари загружены из {self.storage_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке словарей: {e}")

    def _normalize_entity_value(self, value: str) -> Optional[str]:
        """
        Нормализация значения сущности с дополнительной фильтрацией

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

        # Дополнительная фильтрация для имен (persons)
        # Исключаем явно не-имена: глаголы, мат, обычные слова
        stop_words = {
            'бля', 'блять', 'блядь', 'хуй', 'пизда', 'ебан', 'ебанутый',
            'весна', 'лето', 'осень', 'зима', 'день', 'ночь', 'утро', 'вечер',
            'походила', 'позвоню', 'сказал', 'сказала', 'говорил', 'говорила',
            'делал', 'делала', 'ходил', 'ходила', 'пришел', 'пришла',
            'саш', 'аллой', 'снежаны',  # Примеры из логов - явно не имена
        }
        
        if normalized in stop_words:
            return None

        return normalized

    def _prevalidate_entity(self, entity_type: str, normalized_value: str, original_value: str) -> bool:
        """
        Предварительная валидация сущности перед LLM-валидацией.
        Отсеивает явно неправильные сущности (предлоги, обычные слова и т.д.)

        Args:
            entity_type: Тип сущности
            normalized_value: Нормализованное значение
            original_value: Оригинальное значение

        Returns:
            True если сущность прошла предварительную проверку, False иначе
        """
        # Импортируем стоп-слова из токенизатора
        try:
            from ..utils.russian_tokenizer import RUSSIAN_STOP_WORDS, ENGLISH_STOP_WORDS
            stop_words = RUSSIAN_STOP_WORDS | ENGLISH_STOP_WORDS
        except ImportError:
            # Fallback стоп-слова, если токенизатор недоступен
            stop_words = {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
                'против', 'про', 'для', 'от', 'до', 'из', 'к', 'у', 'по', 'за', 'над', 'под',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
            }

        # Проверка на стоп-слова (предлоги, союзы и т.д.)
        if normalized_value in stop_words:
            logger.debug(f"Сущность отклонена (стоп-слово): {entity_type}={normalized_value}")
            return False

        # Специфичные проверки для разных типов сущностей
        if entity_type == "locations":
            # Предлоги и наречия, которые не являются локациями
            invalid_locations = {
                'против', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три',
                'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
                'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
                'конечно', 'всю', 'между'
            }
            if normalized_value in invalid_locations:
                logger.debug(f"Сущность отклонена (не локация): {entity_type}={normalized_value}")
                return False

        elif entity_type == "organizations":
            # Слова, которые не являются организациями
            invalid_organizations = {
                'киберспорт',  # Это индустрия, а не организация
                'спорт', 'игра', 'игры', 'компьютер', 'компьютеры',
                'технология', 'технологии', 'разработка', 'программирование'
            }
            if normalized_value in invalid_organizations:
                logger.debug(f"Сущность отклонена (не организация): {entity_type}={normalized_value}")
                return False

        elif entity_type == "persons":
            # Проверка на предлоги и обычные слова в родительном падеже
            # "андрея" может быть валидным (родительный падеж от "Андрей"),
            # но нужно проверить, не является ли это обычным словом
            # Эта проверка более мягкая, так как имена могут быть в разных падежах

            # Слишком короткие значения (менее 3 символов) - подозрительны
            if len(normalized_value) < 3:
                logger.debug(f"Сущность отклонена (слишком короткая): {entity_type}={normalized_value}")
                return False

        return True

    def _get_llm_client(self):
        """Получение или создание LLM клиента для валидации"""
        if self._llm_client is not None:
            return self._llm_client
        
        if not self._llm_client_initialized:
            try:
                from ..config import get_settings, get_quality_analysis_settings
                from ..core.lmstudio_client import LMStudioEmbeddingClient
                from ..core.ollama_client import OllamaEmbeddingClient
                
                settings = get_settings()
                
                # Пытаемся использовать LM Studio, если указана LLM модель
                if settings.lmstudio_llm_model:
                    self._llm_client = LMStudioEmbeddingClient(
                        model_name=settings.lmstudio_model,  # Для эмбеддингов (не используется здесь)
                        llm_model_name=settings.lmstudio_llm_model,  # Для генерации текста
                        base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
                    )
                    logger.debug("Используется LM Studio для валидации сущностей")
                else:
                    # Используем Ollama как fallback
                    qa_settings = get_quality_analysis_settings()
                    self._llm_client = OllamaEmbeddingClient(
                        llm_model_name=qa_settings.ollama_model,
                        base_url=qa_settings.ollama_base_url
                    )
                    logger.debug("Используется Ollama для валидации сущностей")
                
                self._llm_client_initialized = True
            except Exception as e:
                logger.warning(f"Не удалось инициализировать LLM клиент для валидации: {e}")
                self._llm_client = None
                self._llm_client_initialized = True
        
        return self._llm_client

    async def _validate_entity_with_llm_async(
        self, entity_type: str, normalized_value: str, original_value: str, chat_name: Optional[str] = None
    ) -> bool:
        """
        Асинхронная валидация сущности через LLM
        
        Args:
            entity_type: Тип сущности (persons, organizations, locations и т.д.)
            normalized_value: Нормализованное значение
            original_value: Оригинальное значение (для контекста)
            
        Returns:
            True если сущность валидна, False иначе
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            # Если LLM недоступен, пропускаем валидацию (разрешаем добавление)
            return True
        
        # Промпт для валидации сущностей
        entity_type_names = {
            "persons": "имя человека",
            "organizations": "название организации",
            "locations": "название места/локации",
            "crypto_tokens": "криптовалютный токен",
            "telegram_channels": "Telegram канал",
            "telegram_bots": "Telegram бот",
            "crypto_addresses": "криптовалютный адрес",
            "domains": "доменное имя",
        }
        
        type_name = entity_type_names.get(entity_type, entity_type)
        
        # Проверяем, не является ли это никнеймом пользователя из чата
        is_username = False
        username_context = ""
        if chat_name and entity_type in ("telegram_channels", "telegram_bots"):
            is_username = self.is_username_in_chat(chat_name, normalized_value)
            if is_username:
                # Получаем связанные имена для контекста
                usernames_in_chat = list(self.username_to_names.get(chat_name, {}).keys())
                if usernames_in_chat:
                    username_context = f"\n\nВАЖНО: В этом чате есть пользователи с никнеймами: {', '.join(usernames_in_chat[:10])}. Если '{normalized_value}' является никнеймом пользователя из этого чата, а не каналом/ботом - отклони."
        
        # Специальные инструкции для разных типов сущностей
        type_specific_rules = ""
        if entity_type == "locations":
            # Проверяем, начинается ли оригинальное значение с заглавной буквы (маркер имени собственного)
            starts_with_capital = original_value and len(original_value) > 0 and original_value[0].isupper()
            capital_hint = ""
            if starts_with_capital:
                capital_hint = "\nПОДСКАЗКА: Оригинальное значение начинается с заглавной буквы, что может указывать на имя собственное (название места)."
            
            type_specific_rules = f"""
ВАЖНО для локаций:
- Учитывай склонения и падежи русских названий мест (географические объекты могут быть в разных падежах)
- Если корень слова совпадает с известным географическим названием, но окончание изменено (склонение) - прими
- Сленговые и неофициальные названия городов (например, "Питер" для Санкт-Петербурга) и их склонения являются локациями
- Составные названия с дефисом в разных падежах - это одно и то же географическое название
- Названия географических объектов (города, страны, регионы, реки, озера, моря, горы) могут склоняться
- Названия объектов городской инфраструктуры (станции, вокзалы, аэропорты, площади, улицы, парки, районы) являются локациями, даже если они выглядят как прилагательные
- Названия коммерческих и культурных объектов (торговые центры, магазины, рестораны, кафе, кинотеатры, театры) являются локациями
- Прилагательные, которые являются частью названий объектов инфраструктуры, считаются локациями
- Если слово начинается с заглавной буквы в оригинале, это может быть маркером имени собственного (названия места)
- Склонения и падежи не должны быть причиной отклонения, если это явно географическое название или название объекта инфраструктуры
- Если в названии есть дефис и одна из частей - известное географическое название, это может быть составное название места{capital_hint}"""
        elif entity_type == "persons":
            type_specific_rules = """
ВАЖНО для имен:
- Учитывай уменьшительные, ласкательные и сокращенные формы имен
- Учитывай склонения имен (родительный, дательный, творительный, предложный падежи) - это тоже валидные формы имен
- Если корень слова совпадает с известным именем, но окончание изменено (склонение) - прими
- Короткие имена могут быть валидными уменьшительными формами
- В русском языке широко используются уменьшительные формы имен с различными суффиксами
- Склонения и падежи не должны быть причиной отклонения, если это явно форма известного имени
- Не отклоняй имя только из-за его длины или звучания, если оно может быть формой реального имени"""
        
        prompt = f"""Ты эксперт по анализу текста. Определи, является ли данное слово/фраза действительно {type_name}.

Слово/фраза для проверки: "{original_value}" (нормализованное: "{normalized_value}"){username_context}

Правила:
1. Для типа "{entity_type}": это должно быть действительно {type_name}, а не обычное слово, глагол, прилагательное или другое слово общего назначения.
2. Исключи мат, грубые слова, времена года, дни недели, обычные глаголы и прилагательные.
3. Если это опечатка или случайное сокращение - отклони. Официальные сокращения стран, городов и других географических объектов (например, "США", "СПб") - принимай.
4. Если это никнейм пользователя из чата (не канал/бот) - отклони.{type_specific_rules}

Ответь ТОЛЬКО одним словом: "ДА" если это валидная {type_name}, или "НЕТ" если это не валидная {type_name}.
Не добавляй никаких объяснений, только "ДА" или "НЕТ"."""

        try:
            # Используем async контекстный менеджер
            async with llm_client:
                if hasattr(llm_client, 'generate_summary'):
                    # LMStudioEmbeddingClient или OllamaEmbeddingClient
                    # Для reasoning-моделей увеличиваем max_tokens, так как они генерируют reasoning перед ответом
                    # Минимум 200 токенов для reasoning-моделей, 100 для обычных
                    base_max_tokens = 100
                    # Проверяем, является ли модель reasoning-моделью
                    if hasattr(llm_client, '_is_reasoning_model') and hasattr(llm_client, 'llm_model_name'):
                        if llm_client._is_reasoning_model(llm_client.llm_model_name or ""):
                            base_max_tokens = 4096  # Увеличено для reasoning-моделей, чтобы хватало на reasoning + ответ
                    
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.1,  # Низкая температура для более детерминированных ответов
                        max_tokens=base_max_tokens,  # Увеличено для поддержки reasoning-моделей
                        top_p=0.9,
                        presence_penalty=0.0,
                    )
                else:
                    # Fallback - если метод называется по-другому
                    logger.warning("LLM клиент не поддерживает generate_summary, пропускаем валидацию")
                    return True
                
                # Парсим ответ
                # Метод generate_summary должен вернуть только content, но на всякий случай проверяем
                response_clean = response.strip().upper()
                
                # Если ответ содержит JSON (что не должно происходить, но на всякий случай)
                if response_clean.startswith("{") or response_clean.startswith("["):
                    try:
                        import json
                        parsed = json.loads(response)
                        # Пытаемся извлечь content из JSON
                        if isinstance(parsed, dict):
                            if "choices" in parsed and len(parsed["choices"]) > 0:
                                message = parsed["choices"][0].get("message", {})
                                response_clean = message.get("content", "").strip().upper()
                            elif "content" in parsed:
                                response_clean = parsed["content"].strip().upper()
                    except (json.JSONDecodeError, KeyError, AttributeError):
                        logger.warning(f"Не удалось распарсить JSON-ответ от LLM для '{normalized_value}': {response[:200]}")
                
                if "ДА" in response_clean or "YES" in response_clean:
                    logger.debug(f"LLM подтвердил сущность: {entity_type}={normalized_value}")
                    return True
                elif "НЕТ" in response_clean or "NO" in response_clean:
                    logger.debug(f"LLM отклонил сущность: {entity_type}={normalized_value}")
                    return False
                else:
                    # Если ответ неоднозначный, логируем и отклоняем (консервативный подход)
                    logger.warning(
                        f"Неоднозначный ответ LLM для '{normalized_value}' (тип: {entity_type}): '{response[:100]}'. "
                        f"Отклоняем добавление для безопасности."
                    )
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при валидации сущности '{normalized_value}' через LLM: {e}")
            raise RuntimeError(
                f"Ошибка валидации сущности '{normalized_value}' (тип: {entity_type}) через LLM: {e}. "
                "Проверьте конфигурацию LLM клиента."
            ) from e

    async def _validate_entities_batch_async(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Батч-валидация нескольких сущностей одним запросом к LLM
        
        Args:
            candidates: Список словарей с ключами:
                - entity_type: тип сущности
                - normalized_value: нормализованное значение
                - original_value: оригинальное значение
                - chat_name: название чата (опционально)
        
        Returns:
            Словарь {normalized_value: bool} с результатами валидации
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            # Если LLM недоступен, разрешаем все
            return {c["normalized_value"]: True for c in candidates}
        
        if not candidates:
            return {}
        
        # Группируем кандидатов по типу сущности для формирования промпта
        entity_type_names = {
            "persons": "имя человека",
            "organizations": "название организации",
            "locations": "название места/локации",
            "crypto_tokens": "криптовалютный токен",
            "telegram_channels": "Telegram канал",
            "telegram_bots": "Telegram бот",
            "crypto_addresses": "криптовалютный адрес",
            "domains": "доменное имя",
        }
        
        # Создаем промпт для батч-валидации
        items_list = []
        for i, candidate in enumerate(candidates):
            entity_type = candidate["entity_type"]
            type_name = entity_type_names.get(entity_type, entity_type)
            items_list.append(
                f"{i+1}. Тип: {type_name}, "
                f"Слово: \"{candidate['original_value']}\" "
                f"(нормализованное: \"{candidate['normalized_value']}\")"
            )
        
        items_text = "\n".join(items_list)
        
        # Собираем контекст никнеймов из всех чатов
        chat_names = {c.get("chat_name") for c in candidates if c.get("chat_name")}
        username_context = ""
        if chat_names:
            all_usernames = set()
            for chat_name in chat_names:
                usernames_in_chat = list(self.username_to_names.get(chat_name, {}).keys())
                all_usernames.update(usernames_in_chat[:10])
            if all_usernames:
                username_context = f"\n\nВАЖНО: В чатах есть пользователи с никнеймами: {', '.join(list(all_usernames)[:20])}. Если слово является никнеймом пользователя из чата, а не каналом/ботом - отклони."
        
        prompt = f"""Ты эксперт по анализу текста. Определи для каждого слова/фразы, является ли оно действительно указанным типом сущности.

Список для проверки:
{items_text}{username_context}

Правила:
1. Для каждого типа сущности: это должно быть действительно сущность этого типа, а не обычное слово, глагол, прилагательное или другое слово общего назначения.
2. Исключи мат, грубые слова, времена года, дни недели, обычные глаголы и прилагательные.
3. Если это опечатка или случайное сокращение - отклони. Официальные сокращения стран, городов и других географических объектов (например, "США", "СПб") - принимай.
4. Если это никнейм пользователя из чата (не канал/бот) - отклони.
5. Для локаций: учитывай склонения и падежи русских названий мест.
6. Для имен: учитывай уменьшительные, ласкательные и сокращенные формы имен, а также склонения.

Ответь ТОЛЬКО в формате JSON массива, где каждый элемент - объект с ключами:
- "index": номер из списка (1, 2, 3...)
- "valid": true или false

Пример ответа:
[
  {{"index": 1, "valid": true}},
  {{"index": 2, "valid": false}},
  {{"index": 3, "valid": true}}
]

Не добавляй никаких объяснений, только JSON массив."""

        try:
            async with llm_client:
                if hasattr(llm_client, 'generate_summary'):
                    base_max_tokens = 100
                    if hasattr(llm_client, '_is_reasoning_model') and hasattr(llm_client, 'llm_model_name'):
                        if llm_client._is_reasoning_model(llm_client.llm_model_name or ""):
                            base_max_tokens = 4096
                    
                    # Увеличиваем max_tokens пропорционально размеру батча
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.1,
                        max_tokens=base_max_tokens * len(candidates),
                        top_p=0.9,
                        presence_penalty=0.0,
                    )
                else:
                    logger.warning("LLM клиент не поддерживает generate_summary, пропускаем валидацию")
                    return {c["normalized_value"]: True for c in candidates}
                
                # Парсим JSON ответ
                try:
                    # Извлекаем JSON из ответа (может быть обернут в markdown код блоки)
                    response_clean = response.strip()
                    if "```json" in response_clean:
                        response_clean = response_clean.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_clean:
                        response_clean = response_clean.split("```")[1].split("```")[0].strip()
                    
                    # Пытаемся найти JSON массив в ответе
                    import json
                    # Ищем начало массива
                    start_idx = response_clean.find('[')
                    end_idx = response_clean.rfind(']')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        response_clean = response_clean[start_idx:end_idx+1]
                    
                    results = json.loads(response_clean)
                    
                    # Создаем словарь результатов
                    validation_results = {}
                    for result in results:
                        index = result.get("index", 0) - 1  # Индекс в массиве (0-based)
                        if 0 <= index < len(candidates):
                            normalized_value = candidates[index]["normalized_value"]
                            validation_results[normalized_value] = result.get("valid", False)
                    
                    # Если не все результаты получены, разрешаем остальные (консервативный подход)
                    for candidate in candidates:
                        if candidate["normalized_value"] not in validation_results:
                            logger.warning(f"Не получен результат валидации для '{candidate['normalized_value']}'. Разрешаем.")
                            validation_results[candidate["normalized_value"]] = True
                    
                    return validation_results
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON ответа от LLM: {e}. Ответ: {response[:500]}")
                    # При ошибке парсинга разрешаем все (консервативный подход)
                    return {c["normalized_value"]: True for c in candidates}
                    
        except Exception as e:
            logger.error(f"Ошибка при батч-валидации сущностей через LLM: {e}")
            # При ошибке разрешаем все
            return {c["normalized_value"]: True for c in candidates}

    async def _flush_validation_queue_async(self) -> None:
        """Обработка накопленной очереди валидации батчами"""
        if not self._validation_queue:
            return
        
        queue_size = len(self._validation_queue)
        logger.info(f"Обработка очереди валидации: {queue_size} кандидатов")
        
        # Обрабатываем батчами
        for i in range(0, len(self._validation_queue), self.batch_validation_size):
            batch = self._validation_queue[i:i + self.batch_validation_size]
            batch_num = (i // self.batch_validation_size) + 1
            total_batches = (len(self._validation_queue) + self.batch_validation_size - 1) // self.batch_validation_size
            
            logger.debug(f"Валидация батча {batch_num}/{total_batches} ({len(batch)} сущностей)")
            
            try:
                results = await self._validate_entities_batch_async(batch)
                
                # Применяем результаты
                for candidate in batch:
                    normalized_value = candidate["normalized_value"]
                    entity_type = candidate["entity_type"]
                    is_valid = results.get(normalized_value, True)
                    total_count = self.entity_counts[entity_type].get(normalized_value, 0)
                    
                    if is_valid:
                        self.learned_dictionaries[entity_type].add(normalized_value)
                        logger.info(
                            f"Добавлена новая сущность в словарь: {entity_type}={normalized_value} "
                            f"(встречается {total_count} раз)"
                        )
                        
                        # Генерируем описание, если включено
                        if self.enable_description_generation:
                            try:
                                # Собираем все контексты для более полного описания
                                all_contexts = self.collect_entity_contexts(entity_type, normalized_value)
                                await self.generate_entity_description(
                                    entity_type,
                                    candidate.get("original_value", normalized_value),
                                    all_contexts=all_contexts if all_contexts else None
                                )
                            except Exception as e:
                                logger.debug(f"Не удалось сгенерировать описание для {normalized_value}: {e}")
                    else:
                        logger.debug(
                            f"Сущность отклонена LLM: {entity_type}={normalized_value} "
                            f"(встречается {total_count} раз, но не прошла валидацию)"
                        )
            except Exception as e:
                logger.error(f"Ошибка при обработке батча валидации: {e}")
                # При ошибке разрешаем все сущности в батче (консервативный подход)
                for candidate in batch:
                    normalized_value = candidate["normalized_value"]
                    entity_type = candidate["entity_type"]
                    total_count = self.entity_counts[entity_type].get(normalized_value, 0)
                    self.learned_dictionaries[entity_type].add(normalized_value)
                    logger.warning(
                        f"Сущность добавлена из-за ошибки валидации: {entity_type}={normalized_value} "
                        f"(встречается {total_count} раз)"
                    )
        
        # Очищаем очередь
        processed_count = len(self._validation_queue)
        self._validation_queue.clear()
        logger.info(f"Очередь валидации обработана: {processed_count} кандидатов")
    
    def flush_validation_queue(self) -> None:
        """Синхронная обертка для обработки очереди валидации"""
        if not self._validation_queue:
            return
        
        try:
            # Пытаемся получить существующий event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Если loop уже запущен, создаем новый в отдельном потоке
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(self._flush_validation_queue_async())
                        )
                        future.result(timeout=30.0)  # Таймаут 30 секунд
                else:
                    # Если loop не запущен, используем его
                    loop.run_until_complete(self._flush_validation_queue_async())
            except RuntimeError:
                # Если нет event loop, создаем новый
                asyncio.run(self._flush_validation_queue_async())
        except concurrent.futures.TimeoutError:
            logger.warning("Таймаут при обработке очереди валидации. Продолжаем с сохранением.")
        except Exception as e:
            logger.warning(f"Ошибка при обработке очереди валидации: {e}. Продолжаем с сохранением.")

    def _validate_entity_with_llm(
        self, entity_type: str, normalized_value: str, original_value: str, chat_name: Optional[str] = None
    ) -> bool:
        """
        Синхронная обертка для асинхронной валидации через LLM
        
        Args:
            entity_type: Тип сущности
            normalized_value: Нормализованное значение
            original_value: Оригинальное значение
            chat_name: Название чата (опционально, для проверки никнеймов)
            
        Returns:
            True если сущность валидна, False иначе
        """
        # Определяем таймаут в зависимости от типа модели
        llm_client = self._get_llm_client()
        timeout = 5.0  # По умолчанию 5 секунд
        if llm_client and hasattr(llm_client, '_is_reasoning_model') and hasattr(llm_client, 'llm_model_name'):
            if llm_client._is_reasoning_model(llm_client.llm_model_name or ""):
                timeout = 30.0  # Для reasoning-моделей увеличиваем таймаут до 30 секунд
        
        try:
            # Пытаемся получить существующий event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Если loop уже запущен, создаем новый в отдельном потоке
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._validate_entity_with_llm_async(entity_type, normalized_value, original_value, chat_name)
                        )
                        try:
                            return future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            logger.warning(
                                f"Таймаут валидации сущности {entity_type}={normalized_value} "
                                f"(таймаут {timeout}с). Отклоняем добавление."
                            )
                            return False
                else:
                    # Если loop не запущен, используем его
                    return loop.run_until_complete(
                        self._validate_entity_with_llm_async(entity_type, normalized_value, original_value, chat_name)
                    )
            except RuntimeError:
                # Если нет event loop, создаем новый
                return asyncio.run(
                    self._validate_entity_with_llm_async(entity_type, normalized_value, original_value, chat_name)
                )
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"Таймаут валидации сущности {entity_type}={normalized_value} "
                f"(таймаут {timeout}с). Отклоняем добавление."
            )
            return False
        except Exception as e:
            # Логируем полную информацию об ошибке для диагностики
            import traceback
            error_details = str(e) if str(e) else type(e).__name__
            logger.warning(
                f"Ошибка при синхронной валидации сущности {entity_type}={normalized_value}: {error_details}. "
                f"Отклоняем добавление для безопасности."
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Трассировка ошибки валидации:\n{traceback.format_exc()}")
            # При ошибке отклоняем добавление (консервативный подход)
            return False

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
        self.entity_descriptions.clear()
        logger.info("Все словари и счетчики сброшены")

    def get_entity_description(self, entity_type: str, value: str) -> Optional[str]:
        """
        Получение описания сущности

        Args:
            entity_type: Тип сущности
            value: Значение сущности

        Returns:
            Описание сущности или None, если описание отсутствует
        """
        if entity_type not in ENTITY_TYPES:
            return None

        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return None

        return self.entity_descriptions.get(entity_type, {}).get(normalized_value)

    def update_entity_description(self, entity_type: str, value: str, description: str) -> None:
        """
        Обновление описания сущности

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            description: Новое описание
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return

        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return

        self.entity_descriptions[entity_type][normalized_value] = description.strip()
        logger.debug(f"Обновлено описание для {entity_type}={normalized_value}")

    def _collect_entity_context(
        self, entity_type: str, normalized_value: str, original_value: str
    ) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """
        Сбор контекста для генерации описания сущности

        Args:
            entity_type: Тип сущности
            normalized_value: Нормализованное значение
            original_value: Оригинальное значение

        Returns:
            Кортеж (context_messages, mention_stats):
            - context_messages: Список сообщений, где упоминается сущность (если доступен граф)
            - mention_stats: Статистика упоминаний (частота, чаты, временные метки)
        """
        context_messages = None
        mention_stats = None

        # Собираем статистику упоминаний
        total_count = self.entity_counts[entity_type].get(normalized_value, 0)
        chat_counts = {}
        for chat_name, chat_data in self.chat_entity_counts.items():
            chat_count = chat_data.get(entity_type, {}).get(normalized_value, 0)
            if chat_count > 0:
                chat_counts[chat_name] = chat_count

        mention_stats = {
            "total_count": total_count,
            "chat_counts": chat_counts,
            "chats": list(chat_counts.keys()),
        }

        # Собираем контекст из графа, если доступен
        if self.graph:
            try:
                # Ищем узлы DocChunk, которые упоминают эту сущность
                from ..memory.graph_types import NodeType
                
                # Ищем через FTS поиск по сущностям
                search_results = self.graph.search_text(
                    query=normalized_value,
                    top_k=10,
                    node_types=[NodeType.DOC_CHUNK],
                )
                
                context_messages = []
                for result in search_results[:5]:  # Берем первые 5 результатов
                    node_id = result.get("node_id")
                    if node_id and node_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[node_id]
                        content = node_data.get("content", "")
                        if content and len(content) > 20:  # Минимальная длина сообщения
                            context_messages.append(content[:300])  # Ограничиваем длину
            except Exception as e:
                logger.debug(f"Ошибка при сборе контекста из графа для {normalized_value}: {e}")

        return context_messages, mention_stats

    async def generate_entity_description(
        self,
        entity_type: str,
        value: str,
        context_messages: Optional[List[str]] = None,
        mention_stats: Optional[Dict[str, Any]] = None,
        all_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Генерация описания сущности через LLM на основе всех контекстов

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            context_messages: Список сообщений с упоминаниями сущности (опционально, для обратной совместимости)
            mention_stats: Статистика упоминаний (опционально)
            all_contexts: Полный список контекстов с метаданными (приоритетнее context_messages)

        Returns:
            Сгенерированное описание или None при ошибке
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return None

        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return None

        # Если описание уже существует, не генерируем заново
        existing_description = self.entity_descriptions.get(entity_type, {}).get(normalized_value)
        if existing_description:
            logger.debug(f"Описание для {entity_type}={normalized_value} уже существует")
            return existing_description

        llm_client = self._get_llm_client()
        if not llm_client:
            logger.debug("LLM клиент недоступен, пропускаем генерацию описания")
            return None

        # Собираем контекст, если не передан
        if all_contexts is None:
            if context_messages is None or mention_stats is None:
                context_messages, mention_stats = self._collect_entity_context(
                    entity_type, normalized_value, value
                )
            # Преобразуем context_messages в формат all_contexts для единообразия
            all_contexts = [{"content": msg} for msg in (context_messages or [])]
        else:
            # Если передан all_contexts, извлекаем статистику из него
            if mention_stats is None:
                total_count = self.entity_counts[entity_type].get(normalized_value, 0)
                chat_counts = {}
                for ctx in all_contexts:
                    chat = ctx.get("chat", "unknown")
                    chat_counts[chat] = chat_counts.get(chat, 0) + 1
                mention_stats = {
                    "total_count": total_count,
                    "chat_counts": chat_counts,
                    "chats": list(chat_counts.keys()),
                }

        # Формируем промпт
        entity_type_names = {
            "persons": "персона",
            "organizations": "организация",
            "locations": "локация/место",
            "crypto_tokens": "криптовалютный токен",
            "telegram_channels": "Telegram канал",
            "telegram_bots": "Telegram бот",
            "crypto_addresses": "криптовалютный адрес",
            "domains": "доменное имя",
        }

        type_name = entity_type_names.get(entity_type, entity_type)

        # Формируем контекст для промпта из all_contexts
        context_text = ""
        if all_contexts:
            # Группируем по чатам для лучшего понимания
            contexts_by_chat = {}
            for ctx in all_contexts[:20]:  # Берем первые 20 контекстов
                chat = ctx.get("chat", "unknown")
                if chat not in contexts_by_chat:
                    contexts_by_chat[chat] = []
                content = ctx.get("content", "")
                if content:
                    contexts_by_chat[chat].append(content[:200])  # Ограничиваем длину
            
            # Формируем текст контекста
            context_parts = []
            for chat, contents in list(contexts_by_chat.items())[:5]:  # Максимум 5 чатов
                context_parts.append(f"\nЧат '{chat}':")
                for content in contents[:3]:  # Максимум 3 сообщения на чат
                    context_parts.append(f"  - {content}")
            
            context_text = "\n".join(context_parts)
        
        stats_text = ""
        if mention_stats:
            total_count = mention_stats.get("total_count", 0)
            chats = mention_stats.get("chats", [])
            stats_text = f"\nСтатистика упоминаний:\n- Всего упоминаний: {total_count}\n- Чаты: {', '.join(chats[:5])}"

        prompt = f"""Ты эксперт по анализу текста. Создай краткое, но информативное описание сущности на основе всех контекстов упоминаний.

Тип сущности: {type_name}
Имя сущности: {value} (нормализованное: {normalized_value})
{stats_text}

Контексты упоминаний из разных чатов:
{context_text if context_text else "Контекст недоступен"}

Создай краткое описание (1-2 предложения, до 200 символов) на русском языке.
Описание должно:
- Быть информативным и помогать понять, что это за сущность
- Отражать роль/назначение сущности в контексте обсуждений
- Включать ключевые характеристики, если они упоминаются

Только описание, без дополнительных комментариев."""

        try:
            async with llm_client:
                if hasattr(llm_client, 'generate_summary'):
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=300,
                        top_p=0.9,
                        presence_penalty=0.0,
                    )
                    
                    # Очищаем ответ от лишних символов
                    description = response.strip()
                    # Убираем кавычки, если есть
                    if description.startswith('"') and description.endswith('"'):
                        description = description[1:-1]
                    if description.startswith("'") and description.endswith("'"):
                        description = description[1:-1]
                    
                    # Ограничиваем длину
                    max_length = 200
                    if len(description) > max_length:
                        description = description[:max_length].rsplit(' ', 1)[0] + "..."
                    
                    if description:
                        self.update_entity_description(entity_type, normalized_value, description)
                        logger.info(f"Сгенерировано описание для {entity_type}={normalized_value}: {description[:50]}...")
                        return description
                else:
                    logger.warning("LLM клиент не поддерживает generate_summary, пропускаем генерацию описания")
                    return None
        except Exception as e:
            logger.warning(f"Ошибка при генерации описания для {entity_type}={normalized_value}: {e}")
            return None

        return None

    def collect_entity_contexts(
        self, entity_type: str, normalized_value: str
    ) -> List[Dict[str, Any]]:
        """
        Сбор всех контекстов упоминания сущности из графа
        
        Args:
            entity_type: Тип сущности
            normalized_value: Нормализованное значение сущности
            
        Returns:
            Список контекстов с метаданными (чат, дата, автор, содержание)
        """
        contexts = []
        
        if not self.graph:
            logger.debug("Граф недоступен для сбора контекстов")
            return contexts
        
        try:
            from ..memory.graph_types import NodeType, EdgeType
            
            # Ищем EntityNode для этой сущности
            entity_id = f"entity-{normalized_value.replace(' ', '-')}"
            
            # Ищем через FTS поиск по сущностям в DocChunk узлах
            search_results, _ = self.graph.search_text(
                query=normalized_value,
                limit=100,  # Берем больше результатов для полной картины
            )
            
            # Также ищем через граф - находим DocChunk узлы, связанные с EntityNode
            if entity_id in self.graph.graph:
                # Получаем все узлы, которые упоминают эту сущность
                neighbors = self.graph.get_neighbors(
                    entity_id,
                    edge_type=EdgeType.MENTIONS,
                    direction="in",  # Входящие связи (DocChunk -> mentions -> Entity)
                )
                
                for neighbor_id, edge_data in neighbors:
                    if neighbor_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[neighbor_id]
                        node_type = node_data.get("type")
                        
                        if node_type == NodeType.DOC_CHUNK.value or node_type == NodeType.DOC_CHUNK:
                            content = node_data.get("content", "")
                            properties = node_data.get("properties", {})
                            
                            if content and len(content) > 10:
                                context = {
                                    "node_id": neighbor_id,
                                    "content": content,
                                    "source": properties.get("source", ""),
                                    "chat": properties.get("chat") or properties.get("source", ""),
                                    "author": properties.get("author", ""),
                                    "timestamp": properties.get("timestamp") or properties.get("date_utc", ""),
                                    "tags": properties.get("tags", []),
                                }
                                contexts.append(context)
            
            # Добавляем контексты из FTS поиска
            seen_node_ids = {ctx["node_id"] for ctx in contexts}
            for result in search_results:
                node_id = result.get("node_id")
                if node_id and node_id not in seen_node_ids:
                    if node_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[node_id]
                        node_type = node_data.get("type")
                        
                        if node_type == NodeType.DOC_CHUNK.value or node_type == NodeType.DOC_CHUNK:
                            content = result.get("content", "") or node_data.get("content", "")
                            properties = node_data.get("properties", {})
                            
                            if content and len(content) > 10:
                                context = {
                                    "node_id": node_id,
                                    "content": content[:500],  # Ограничиваем длину
                                    "source": result.get("source", "") or properties.get("source", ""),
                                    "chat": properties.get("chat") or properties.get("source", ""),
                                    "author": properties.get("author", ""),
                                    "timestamp": properties.get("timestamp") or properties.get("date_utc", ""),
                                    "tags": properties.get("tags", []),
                                    "score": result.get("score", 0.0),
                                }
                                contexts.append(context)
                                seen_node_ids.add(node_id)
            
            # Группируем по чатам и сортируем по времени
            contexts_by_chat = {}
            for ctx in contexts:
                chat = ctx.get("chat", "unknown")
                if chat not in contexts_by_chat:
                    contexts_by_chat[chat] = []
                contexts_by_chat[chat].append(ctx)
            
            # Сортируем контексты по времени (если доступно)
            for chat in contexts_by_chat:
                contexts_by_chat[chat].sort(
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )
            
            logger.debug(f"Собрано {len(contexts)} контекстов для {entity_type}={normalized_value} из {len(contexts_by_chat)} чатов")
            
        except Exception as e:
            logger.warning(f"Ошибка при сборе контекстов для {normalized_value}: {e}")
        
        return contexts

    def build_entity_profile(
        self, entity_type: str, value: str
    ) -> Dict[str, Any]:
        """
        Формирование полного профиля сущности на основе всех упоминаний
        
        Args:
            entity_type: Тип сущности
            value: Значение сущности
            
        Returns:
            Словарь с полным профилем сущности:
            - entity_type, value, normalized_value
            - description: полное описание
            - aliases: альтернативные названия
            - mention_count: общее количество упоминаний
            - chats: список чатов, где упоминается
            - chat_counts: количество упоминаний по чатам
            - first_seen, last_seen: временные метки
            - contexts: список контекстов упоминаний
            - related_entities: связанные сущности через граф
            - importance: важность сущности (на основе частоты)
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return {}
        
        normalized_value = self._normalize_entity_value(value)
        if not normalized_value:
            return {}
        
        # Собираем базовую статистику
        total_count = self.entity_counts[entity_type].get(normalized_value, 0)
        chat_counts = {}
        all_chats = set()
        
        for chat_name, chat_data in self.chat_entity_counts.items():
            chat_count = chat_data.get(entity_type, {}).get(normalized_value, 0)
            if chat_count > 0:
                chat_counts[chat_name] = chat_count
                all_chats.add(chat_name)
        
        # Собираем контексты из графа
        contexts = self.collect_entity_contexts(entity_type, normalized_value)
        
        # Определяем временные метки
        timestamps = []
        for ctx in contexts:
            ts = ctx.get("timestamp")
            if ts:
                try:
                    from datetime import datetime
                    if isinstance(ts, str):
                        # Пробуем распарсить ISO формат
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        timestamps.append(dt)
                except Exception:
                    pass
        
        first_seen = min(timestamps).isoformat() if timestamps else None
        last_seen = max(timestamps).isoformat() if timestamps else None
        
        # Собираем связанные сущности через граф
        related_entities = []
        if self.graph:
            try:
                from ..memory.graph_types import EdgeType
                entity_id = f"entity-{normalized_value.replace(' ', '-')}"
                
                if entity_id in self.graph.graph:
                    # Находим соседние EntityNode через различные типы связей
                    neighbors = self.graph.get_neighbors(entity_id, direction="both")
                    
                    for neighbor_id, edge_data in neighbors:
                        if neighbor_id in self.graph.graph:
                            neighbor_data = self.graph.graph.nodes[neighbor_id]
                            neighbor_type = neighbor_data.get("type")
                            
                            # Собираем только EntityNode
                            if neighbor_type == "Entity" or (isinstance(neighbor_type, str) and "Entity" in neighbor_type):
                                related_entities.append({
                                    "entity_id": neighbor_id,
                                    "label": neighbor_data.get("label", ""),
                                    "entity_type": neighbor_data.get("entity_type", ""),
                                    "edge_type": edge_data.get("type", ""),
                                    "weight": edge_data.get("weight", 0.0),
                                })
            except Exception as e:
                logger.debug(f"Ошибка при сборе связанных сущностей: {e}")
        
        # Получаем описание (если уже сгенерировано)
        description = self.get_entity_description(entity_type, value)
        
        # Формируем алиасы (из username_to_names и name_to_usernames)
        aliases = [normalized_value]
        for chat_name in all_chats:
            # Проверяем связи имен
            usernames = self.name_to_usernames.get(chat_name, {}).get(value, set())
            names = self.username_to_names.get(chat_name, {}).get(normalized_value, set())
            aliases.extend(list(usernames))
            aliases.extend(list(names))
        
        # Убираем дубликаты и нормализуем
        aliases = list(set([a.lower().strip() for a in aliases if a]))
        
        # Вычисляем важность на основе частоты упоминаний
        # Нормализуем от 0.0 до 1.0 на основе максимальной частоты для этого типа
        max_count = max(self.entity_counts[entity_type].values()) if self.entity_counts[entity_type] else 1
        importance = min(1.0, (total_count / max_count) * 0.8 + 0.2)  # Минимум 0.2, максимум 1.0
        
        profile = {
            "entity_type": entity_type,
            "value": value,
            "normalized_value": normalized_value,
            "description": description,
            "aliases": aliases,
            "mention_count": total_count,
            "chats": sorted(list(all_chats)),
            "chat_counts": chat_counts,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "contexts": contexts[:20],  # Ограничиваем количество контекстов
            "context_count": len(contexts),
            "related_entities": related_entities[:10],  # Ограничиваем количество связанных сущностей
            "importance": importance,
        }
        
        logger.debug(f"Построен профиль для {entity_type}={normalized_value}: {total_count} упоминаний, {len(contexts)} контекстов")
        
        return profile


# Глобальный экземпляр для использования в других модулях
_global_entity_dictionary: Optional[EntityDictionary] = None


def get_entity_dictionary(
    enable_llm_validation: bool = True,
    enable_description_generation: bool = True,
    graph: Optional[Any] = None,
) -> EntityDictionary:
    """
    Получение глобального экземпляра словаря сущностей
    
    Args:
        enable_llm_validation: Включить валидацию через LLM (по умолчанию True)
        enable_description_generation: Включить генерацию описаний (по умолчанию True)
        graph: Граф памяти для доступа к контексту (опционально)
    """
    global _global_entity_dictionary
    if _global_entity_dictionary is None:
        _global_entity_dictionary = EntityDictionary(
            enable_llm_validation=enable_llm_validation,
            enable_description_generation=enable_description_generation,
            graph=graph,
        )
    else:
        # Обновляем флаги, если словарь уже создан
        _global_entity_dictionary.enable_llm_validation = enable_llm_validation
        _global_entity_dictionary.enable_description_generation = enable_description_generation
        if graph is not None:
            _global_entity_dictionary.graph = graph
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
