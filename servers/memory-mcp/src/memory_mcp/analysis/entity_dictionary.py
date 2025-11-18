#!/usr/bin/env python3
"""
Модуль для автоматического обучения словарей сущностей
Отслеживает частоту появления терминов и автоматически добавляет их в словари
Использует LLM для валидации сущностей перед добавлением в словарь
"""

import asyncio
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
    """Класс для автоматического обучения словарей сущностей
    Использует LLM для валидации сущностей перед добавлением в словарь
    """

    def __init__(
        self, 
        storage_path: Path = Path("config/entity_dictionaries"),
        enable_llm_validation: bool = True,
        llm_client: Optional[Any] = None,  # LMStudioEmbeddingClient или OllamaEmbeddingClient
    ):
        """
        Инициализация словаря сущностей

        Args:
            storage_path: Путь к директории для хранения словарей
            enable_llm_validation: Включить валидацию через LLM
            llm_client: Клиент для LLM (опционально, создается автоматически если не указан)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Счетчики частоты появления сущностей
        self.entity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Счетчики по чатам для контекстного анализа
        self.chat_entity_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        
        # Связи имен и никнеймов внутри чатов: {chat_name: {username: [display_names]}}
        self.username_to_names: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        
        # Обратные связи: {chat_name: {display_name: [usernames]}}
        self.name_to_usernames: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        
        # Загруженные словари
        self.learned_dictionaries: Dict[str, Set[str]] = {
            entity_type: set() for entity_type in ENTITY_TYPES
        }
        
        # LLM для валидации
        self.enable_llm_validation = enable_llm_validation
        self._llm_client = llm_client
        self._llm_client_initialized = False
        
        # Загружаем существующие словари
        self.load_dictionaries()

    def link_username_to_name(self, chat_name: str, username: str, display_name: str) -> None:
        """
        Связывание никнейма с именем пользователя внутри чата
        
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
        """
        Проверка, является ли значение никнеймом пользователя из чата
        
        Args:
            chat_name: Название чата
            value: Проверяемое значение
            
        Returns:
            True если это никнейм пользователя из чата
        """
        normalized_value = value.lower().strip().replace('@', '')
        return normalized_value in self.username_to_names.get(chat_name, {})

    def track_entity(self, entity_type: str, value: str, chat_name: str, author_username: Optional[str] = None, author_display_name: Optional[str] = None) -> bool:
        """
        Отслеживание появления сущности

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            chat_name: Название чата
            author_username: Никнейм автора сообщения (опционально, для связывания)
            author_display_name: Отображаемое имя автора (опционально, для связывания)

        Returns:
            True если сущность добавлена в словарь, False иначе
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return False

        if not value or not value.strip():
            return False

        # Связываем имя автора с никнеймом, если доступны
        if author_username and author_display_name:
            self.link_username_to_name(chat_name, author_username, author_display_name)

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
            # Валидация через LLM перед добавлением в словарь
            if self.enable_llm_validation:
                is_valid = self._validate_entity_with_llm(entity_type, normalized_value, value, chat_name)
                if not is_valid:
                    logger.debug(
                        f"Сущность отклонена LLM: {entity_type}={normalized_value} "
                        f"(встречается {total_count} раз, но не прошла валидацию)"
                    )
                    return False
            
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
                response_clean = response.strip().upper()
                if "ДА" in response_clean or "YES" in response_clean:
                    return True
                elif "НЕТ" in response_clean or "NO" in response_clean:
                    return False
                else:
                    # Если ответ неоднозначный, логируем и разрешаем (консервативный подход)
                    logger.debug(f"Неоднозначный ответ LLM для '{normalized_value}': '{response}'. Разрешаем добавление.")
                    return True
                    
        except Exception as e:
            logger.error(f"Ошибка при валидации сущности '{normalized_value}' через LLM: {e}")
            raise RuntimeError(
                f"Ошибка валидации сущности '{normalized_value}' (тип: {entity_type}) через LLM: {e}. "
                "Проверьте конфигурацию LLM клиента."
            ) from e

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
                        return future.result(timeout=5.0)  # Таймаут 5 секунд
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
        except Exception as e:
            logger.warning(f"Ошибка при синхронной валидации сущности: {e}. Разрешаем добавление.")
            return True  # При ошибке разрешаем добавление

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


def get_entity_dictionary(enable_llm_validation: bool = True) -> EntityDictionary:
    """
    Получение глобального экземпляра словаря сущностей
    
    Args:
        enable_llm_validation: Включить валидацию через LLM (по умолчанию True)
    """
    global _global_entity_dictionary
    if _global_entity_dictionary is None:
        _global_entity_dictionary = EntityDictionary(enable_llm_validation=enable_llm_validation)
    else:
        # Обновляем флаг валидации, если словарь уже создан
        _global_entity_dictionary.enable_llm_validation = enable_llm_validation
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
