#!/usr/bin/env python3
"""
Модуль для извлечения сущностей из сообщений (E1)
Извлекает: mentions, tickers/coins, urls/domains, dates/datetimes, numbers, files
Расширенная версия с автоматическим обучением словарей и новыми типами сущностей
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Импорты для расширенного извлечения сущностей
try:
    from natasha import (
        Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
        NewsNERTagger, Doc
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Natasha не установлена. Расширенное извлечение сущностей недоступно.")

from .entity_dictionary import get_entity_dictionary

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Класс для извлечения сущностей из текста сообщений"""

    # Паттерны для распознавания сущностей
    MENTION_PATTERN = r"@(\w+)"
    TICKER_PATTERN = r"\b([A-Z]{2,6}(?:USDT|USD|BTC|ETH)?)\b|\$([A-Z]{2,6})"
    URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+'
    DATE_PATTERN = (
        r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
    )
    DATETIME_PATTERN = r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b"
    NUMBER_PATTERN = r"\b(\d+(?:[.,]\d+)?)\s*([%$€₽¥£]|USD|EUR|RUB|TON|BTC|ETH)?\b"
    FILE_PATTERN = (
        r"\b[\w\-]+\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|zip|rar|mp4|mp3|wav)\b"
    )
    
    # Расширенные паттерны для новых типов сущностей
    CRYPTO_ADDRESS_PATTERNS = {
        "bitcoin": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",  # Bitcoin адреса
        "ethereum": r"\b0x[a-fA-F0-9]{40}\b",  # Ethereum адреса
        "ton": r"\bUQ[a-zA-Z0-9_-]{47}\b",  # TON адреса
    }
    
    TELEGRAM_PATTERNS = {
        "channel": r"@(\w+)",  # Каналы
        "bot": r"@(\w+bot)\b",  # Боты
        "group": r"t\.me/(\w+)",  # Группы через t.me
    }

    # Список известных криптовалют и тикеров
    KNOWN_TICKERS = {
        "BTC",
        "ETH",
        "TON",
        "USDT",
        "USDC",
        "BNB",
        "XRP",
        "ADA",
        "SOL",
        "DOT",
        "MATIC",
        "AVAX",
        "LINK",
        "UNI",
        "ATOM",
        "FTM",
        "NEAR",
        "ALGO",
        "VET",
        "ICP",
        "SAND",
        "MANA",
        "AXS",
        "GALA",
        "ENJ",
        "CHZ",
        "THETA",
        "FIL",
        "AAVE",
        "MKR",
        "SNX",
        "COMP",
        "YFI",
        "SUSHI",
        "CRV",
        "BAL",
        "1INCH",
    }

    def __init__(
        self, 
        enable_learning: bool = True, 
        enable_natasha: bool = True,
        enable_llm_validation: bool = True,
    ):
        """
        Инициализация экстрактора
        
        Args:
            enable_learning: Включить автоматическое обучение словарей
            enable_natasha: Включить извлечение сущностей через Natasha
        """
        # Базовые регулярные выражения
        self.mention_regex = re.compile(self.MENTION_PATTERN, re.IGNORECASE)
        self.ticker_regex = re.compile(self.TICKER_PATTERN)
        self.url_regex = re.compile(self.URL_PATTERN)
        self.date_regex = re.compile(self.DATE_PATTERN)
        self.datetime_regex = re.compile(self.DATETIME_PATTERN, re.IGNORECASE)
        self.number_regex = re.compile(self.NUMBER_PATTERN, re.IGNORECASE)
        self.file_regex = re.compile(self.FILE_PATTERN, re.IGNORECASE)
        
        # Расширенные регулярные выражения
        self.crypto_address_regexes = {
            crypto_type: re.compile(pattern, re.IGNORECASE)
            for crypto_type, pattern in self.CRYPTO_ADDRESS_PATTERNS.items()
        }
        
        self.telegram_regexes = {
            tg_type: re.compile(pattern, re.IGNORECASE)
            for tg_type, pattern in self.TELEGRAM_PATTERNS.items()
        }
        
        # Настройки
        self.enable_learning = enable_learning
        self.enable_natasha = enable_natasha and NATASHA_AVAILABLE
        
        # Словарь сущностей для автоматического обучения
        self.enable_llm_validation = enable_llm_validation
        if enable_learning:
            # Передаем флаг валидации в словарь
            entity_dict = get_entity_dictionary()
            if hasattr(entity_dict, 'enable_llm_validation'):
                entity_dict.enable_llm_validation = enable_llm_validation
            self.entity_dictionary = entity_dict
        else:
            self.entity_dictionary = None
        
        # Natasha компоненты для извлечения именованных сущностей
        self.natasha_components = None
        if self.enable_natasha:
            try:
                emb = NewsEmbedding()
                self.natasha_components = {
                    'segmenter': Segmenter(),
                    'morph_vocab': MorphVocab(),
                    'emb': emb,
                    'morph_tagger': NewsMorphTagger(emb),
                    'ner_tagger': NewsNERTagger(emb)
                }
                logger.info("Natasha компоненты инициализированы")
            except Exception as e:
                logger.warning(f"Ошибка инициализации Natasha: {e}")
                self.enable_natasha = False

    def extract_entities(self, text: str, chat_name: str = "") -> Dict[str, List[str]]:
        """
        Извлечение всех сущностей из текста

        Args:
            text: Текст для анализа
            chat_name: Название чата для обучения словарей

        Returns:
            Словарь с категориями сущностей
        """
        entities = {
            "mentions": [],
            "tickers": [],
            "urls": [],
            "domains": [],
            "dates": [],
            "times": [],
            "numbers": [],
            "files": [],
            # Новые типы сущностей
            "persons": [],
            "organizations": [],
            "locations": [],
            "crypto_addresses": [],
            "telegram_channels": [],
            "telegram_bots": [],
        }

        if not text:
            return entities

        # Базовые сущности
        entities["mentions"] = self._extract_mentions(text)
        entities["tickers"] = self._extract_tickers(text)
        
        urls = self._extract_urls(text)
        entities["urls"] = urls
        entities["domains"] = self._extract_domains(urls)
        
        entities["dates"] = self._extract_dates(text)
        entities["times"] = self._extract_times(text)
        entities["numbers"] = self._extract_numbers(text)
        entities["files"] = self._extract_files(text)

        # Новые типы сущностей
        entities["persons"] = self._extract_persons(text)
        entities["organizations"] = self._extract_organizations(text)
        entities["locations"] = self._extract_locations(text)
        entities["crypto_addresses"] = self._extract_crypto_addresses(text)
        
        telegram_entities = self._extract_telegram_entities(text)
        entities["telegram_channels"] = telegram_entities.get("channels", [])
        entities["telegram_bots"] = telegram_entities.get("bots", [])

        # Автоматическое обучение словарей
        if self.enable_learning and self.entity_dictionary and chat_name:
            self._update_learned_dictionaries(text, chat_name, entities)

        return entities

    def _extract_mentions(self, text: str) -> List[str]:
        """Извлечение упоминаний (@username)"""
        mentions = self.mention_regex.findall(text)
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_mentions = []
        for mention in mentions:
            mention_lower = mention.lower()
            if mention_lower not in seen:
                seen.add(mention_lower)
                unique_mentions.append(mention)
        return unique_mentions

    def _extract_tickers(self, text: str) -> List[str]:
        """Извлечение тикеров криптовалют"""
        matches = self.ticker_regex.findall(text)
        tickers = []

        for match in matches:
            # match это кортеж из двух групп: (группа1, группа2)
            ticker = match[0] if match[0] else match[1]
            if ticker:
                # Проверяем, является ли это известным тикером
                base_ticker = (
                    ticker.replace("USDT", "")
                    .replace("USD", "")
                    .replace("BTC", "")
                    .replace("ETH", "")
                )
                if base_ticker in self.KNOWN_TICKERS or ticker in self.KNOWN_TICKERS:
                    tickers.append(ticker)

        # Убираем дубликаты
        return list(dict.fromkeys(tickers))

    def _extract_urls(self, text: str) -> List[str]:
        """Извлечение URL"""
        urls = self.url_regex.findall(text)
        return list(dict.fromkeys(urls))  # Убираем дубликаты

    def _extract_domains(self, urls: List[str]) -> List[str]:
        """Извлечение доменов из URL"""
        domains = []
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                if domain and domain not in domains:
                    domains.append(domain)
            except Exception as e:
                logger.debug(f"Ошибка парсинга URL {url}: {e}")
                continue
        return domains

    def _extract_dates(self, text: str) -> List[str]:
        """Извлечение дат"""
        dates = self.date_regex.findall(text)
        # Нормализуем формат дат
        normalized_dates = []
        for date_str in dates:
            try:
                # Пробуем различные форматы
                for fmt in [
                    "%d.%m.%Y",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y.%m.%d",
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                ]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        normalized_dates.append(dt.strftime("%Y-%m-%d"))
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.debug(f"Ошибка парсинга даты {date_str}: {e}")
                continue

        return list(dict.fromkeys(normalized_dates))

    def _extract_times(self, text: str) -> List[str]:
        """Извлечение времени"""
        times = self.datetime_regex.findall(text)
        return list(dict.fromkeys(times))

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение чисел с контекстом ($, %, количество)"""
        matches = self.number_regex.findall(text)
        numbers = []

        for match in matches:
            value_str = match[0]
            unit = match[1] if len(match) > 1 else ""

            try:
                # Заменяем запятую на точку для парсинга
                value = float(value_str.replace(",", "."))

                number_info = {
                    "value": value,
                    "original": value_str,
                    "unit": unit if unit else None,
                }
                numbers.append(number_info)
            except ValueError:
                continue

        return numbers

    def _extract_files(self, text: str) -> List[str]:
        """Извлечение имён файлов"""
        self.file_regex.findall(text)
        # files содержит расширения, нужно получить полные имена
        all_files = []
        for match in self.file_regex.finditer(text):
            all_files.append(match.group(0))
        return list(dict.fromkeys(all_files))

    def _extract_persons(self, text: str) -> List[str]:
        """Извлечение имен людей через Natasha с фильтрацией глаголов и обычных слов"""
        if not self.enable_natasha or not self.natasha_components:
            return []

        # Стоп-слова, которые не должны попадать в словарь имен
        # Включаем как нижний регистр (для проверки), так и с заглавной буквы
        stop_words = {
            'бля', 'блять', 'блядь', 'хуй', 'пизда', 'ебан', 'ебанутый',
            'весна', 'лето', 'осень', 'зима', 'день', 'ночь', 'утро', 'вечер',
            'походила', 'позвоню', 'сказал', 'сказала', 'говорил', 'говорила',
            'делал', 'делала', 'ходил', 'ходила', 'пришел', 'пришла',
            'саш', 'аллой', 'снежаны',  # Примеры из логов - явно не имена
            # С заглавной буквы тоже
            'Бля', 'Блять', 'Блядь', 'Хуй', 'Пизда', 'Ебан', 'Ебанутый',
            'Весна', 'Лето', 'Осень', 'Зима', 'День', 'Ночь', 'Утро', 'Вечер',
            'Походила', 'Позвоню', 'Сказал', 'Сказала', 'Говорил', 'Говорила',
            'Делал', 'Делала', 'Ходил', 'Ходила', 'Пришел', 'Пришла',
            'Саш', 'Аллой', 'Снежаны',
        }

        try:
            doc = Doc(text)
            doc.segment(self.natasha_components['segmenter'])
            doc.tag_morph(self.natasha_components['morph_tagger'])
            doc.tag_ner(self.natasha_components['ner_tagger'])

            persons = []
            for span in doc.spans:
                if span.type == 'PER':  # Person
                    person_name = span.text.strip()
                    
                    # Фильтруем слишком короткие имена
                    if len(person_name) <= 2:
                        continue
                    
                    # Фильтруем стоп-слова (проверяем и в нижнем регистре, и в оригинальном)
                    person_lower = person_name.lower()
                    if person_lower in stop_words or person_name in stop_words:
                        continue
                    
                    # Проверяем, что это не глагол - проверяем морфологический разбор
                    # Имена обычно имеют признаки NOUN или PROPN, а не VERB
                    is_verb = False
                    for token in span.tokens:
                        if token.pos == 'VERB' or 'VERB' in str(token.feats):
                            is_verb = True
                            break
                    
                    if is_verb:
                        continue
                    
                    # Проверяем, что имя начинается с заглавной буквы (имена обычно пишутся с заглавной)
                    # Это важная проверка для фильтрации обычных слов
                    if not person_name[0].isupper():
                        continue
                    
                    # Проверяем, что имя содержит только буквы (без цифр и специальных символов)
                    if not person_name.replace(' ', '').replace('-', '').isalpha():
                        continue
                    
                    persons.append(person_name)

            return list(dict.fromkeys(persons))  # Убираем дубликаты
        except Exception as e:
            logger.debug(f"Ошибка извлечения имен людей: {e}")
            return []

    def _extract_organizations(self, text: str) -> List[str]:
        """Извлечение организаций через Natasha"""
        if not self.enable_natasha or not self.natasha_components:
            return []

        try:
            doc = Doc(text)
            doc.segment(self.natasha_components['segmenter'])
            doc.tag_morph(self.natasha_components['morph_tagger'])
            doc.tag_ner(self.natasha_components['ner_tagger'])

            organizations = []
            for span in doc.spans:
                if span.type == 'ORG':  # Organization
                    org_name = span.text.strip()
                    if len(org_name) > 3:  # Фильтруем слишком короткие названия
                        organizations.append(org_name)

            return list(dict.fromkeys(organizations))  # Убираем дубликаты
        except Exception as e:
            logger.debug(f"Ошибка извлечения организаций: {e}")
            return []

    def _extract_locations(self, text: str) -> List[str]:
        """Извлечение мест через Natasha"""
        if not self.enable_natasha or not self.natasha_components:
            return []

        try:
            doc = Doc(text)
            doc.segment(self.natasha_components['segmenter'])
            doc.tag_morph(self.natasha_components['morph_tagger'])
            doc.tag_ner(self.natasha_components['ner_tagger'])

            locations = []
            for span in doc.spans:
                if span.type == 'LOC':  # Location
                    location_name = span.text.strip()
                    if len(location_name) > 2:  # Фильтруем слишком короткие названия
                        locations.append(location_name)

            return list(dict.fromkeys(locations))  # Убираем дубликаты
        except Exception as e:
            logger.debug(f"Ошибка извлечения мест: {e}")
            return []

    def _extract_crypto_addresses(self, text: str) -> List[Dict[str, str]]:
        """Извлечение криптовалютных адресов"""
        addresses = []
        
        for crypto_type, regex in self.crypto_address_regexes.items():
            matches = regex.findall(text)
            for match in matches:
                addresses.append({
                    "type": crypto_type,
                    "address": match,
                    "network": crypto_type.upper()
                })

        return addresses

    def _extract_telegram_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлечение Telegram-специфичных сущностей"""
        result = {"channels": [], "bots": [], "groups": []}
        
        # Извлечение каналов
        channel_matches = self.telegram_regexes["channel"].findall(text)
        result["channels"] = list(dict.fromkeys(channel_matches))
        
        # Извлечение ботов
        bot_matches = self.telegram_regexes["bot"].findall(text)
        result["bots"] = list(dict.fromkeys(bot_matches))
        
        # Извлечение групп через t.me
        group_matches = self.telegram_regexes["group"].findall(text)
        result["groups"] = list(dict.fromkeys(group_matches))
        
        return result

    def _update_learned_dictionaries(self, text: str, chat_name: str, entities: Dict[str, List[str]]) -> None:
        """Обновление словарей на основе извлеченных сущностей"""
        if not self.entity_dictionary:
            return

        # Отслеживаем различные типы сущностей
        entity_mappings = {
            "crypto_tokens": entities.get("tickers", []),
            "persons": entities.get("persons", []),
            "organizations": entities.get("organizations", []),
            "locations": entities.get("locations", []),
            "telegram_channels": entities.get("telegram_channels", []),
            "telegram_bots": entities.get("telegram_bots", []),
            "domains": entities.get("domains", []),
        }

        # Отслеживаем криптовалютные адреса
        crypto_addresses = entities.get("crypto_addresses", [])
        for addr_info in crypto_addresses:
            self.entity_dictionary.track_entity("crypto_addresses", addr_info["address"], chat_name)

        # Отслеживаем остальные сущности
        for entity_type, entity_list in entity_mappings.items():
            for entity_value in entity_list:
                self.entity_dictionary.track_entity(entity_type, entity_value, chat_name)

    def extract_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Извлечение сущностей из списка сообщений

        Args:
            messages: Список сообщений

        Returns:
            Агрегированный словарь сущностей с подсчётом частоты
        """
        aggregated = {
            "mentions": {},
            "tickers": {},
            "urls": {},
            "domains": {},
            "dates": {},
            "numbers": [],
            "files": {},
            "links": {},  # Добавляем отдельную категорию для всех ссылок
            # Новые типы сущностей
            "persons": {},
            "organizations": {},
            "locations": {},
            "crypto_addresses": {},
            "telegram_channels": {},
            "telegram_bots": {},
        }

        for msg in messages:
            text = msg.get("text", "")
            chat_name = msg.get("chat", "")

            # Извлекаем сущности из текста
            if text:
                entities = self.extract_entities(text, chat_name)

                # Агрегируем mentions
                for mention in entities["mentions"]:
                    aggregated["mentions"][mention] = (
                        aggregated["mentions"].get(mention, 0) + 1
                    )

                # Агрегируем tickers
                for ticker in entities["tickers"]:
                    aggregated["tickers"][ticker] = (
                        aggregated["tickers"].get(ticker, 0) + 1
                    )

                # Агрегируем URLs из текста
                for url in entities["urls"]:
                    aggregated["urls"][url] = aggregated["urls"].get(url, 0) + 1
                    aggregated["links"][url] = aggregated["links"].get(url, 0) + 1

                # Агрегируем домены
                for domain in entities["domains"]:
                    aggregated["domains"][domain] = (
                        aggregated["domains"].get(domain, 0) + 1
                    )

                # Агрегируем даты
                for date in entities["dates"]:
                    aggregated["dates"][date] = aggregated["dates"].get(date, 0) + 1

                # Собираем все числа
                aggregated["numbers"].extend(entities["numbers"])

                # Агрегируем файлы из текста
                for file in entities["files"]:
                    aggregated["files"][file] = aggregated["files"].get(file, 0) + 1

                # Агрегируем новые типы сущностей
                for person in entities["persons"]:
                    aggregated["persons"][person] = aggregated["persons"].get(person, 0) + 1

                for org in entities["organizations"]:
                    aggregated["organizations"][org] = aggregated["organizations"].get(org, 0) + 1

                for location in entities["locations"]:
                    aggregated["locations"][location] = aggregated["locations"].get(location, 0) + 1

                for addr_info in entities["crypto_addresses"]:
                    addr_key = f"{addr_info['type']}:{addr_info['address']}"
                    aggregated["crypto_addresses"][addr_key] = aggregated["crypto_addresses"].get(addr_key, 0) + 1

                for channel in entities["telegram_channels"]:
                    aggregated["telegram_channels"][channel] = aggregated["telegram_channels"].get(channel, 0) + 1

                for bot in entities["telegram_bots"]:
                    aggregated["telegram_bots"][bot] = aggregated["telegram_bots"].get(bot, 0) + 1

            # === НОВОЕ: Извлекаем сущности из attachments ===
            attachments = msg.get("attachments", [])
            if attachments:
                for attachment in attachments:
                    att_type = attachment.get("type", "")

                    # Извлекаем URL из attachments
                    if att_type == "url":
                        url = attachment.get("href", "")
                        if url:
                            aggregated["urls"][url] = aggregated["urls"].get(url, 0) + 1
                            aggregated["links"][url] = (
                                aggregated["links"].get(url, 0) + 1
                            )

                            # Извлекаем домен
                            try:
                                parsed = urlparse(url)
                                domain = parsed.netloc
                                if domain:
                                    aggregated["domains"][domain] = (
                                        aggregated["domains"].get(domain, 0) + 1
                                    )
                            except Exception as e:
                                logger.debug(
                                    f"Ошибка парсинга URL из attachment {url}: {e}"
                                )

                    # Извлекаем имена файлов из вложений
                    elif att_type in ["photo", "document", "video", "audio", "voice"]:
                        file_name = attachment.get("file", "")
                        if file_name:
                            # Извлекаем только имя файла без пути
                            import os

                            base_name = os.path.basename(file_name)
                            aggregated["files"][base_name] = (
                                aggregated["files"].get(base_name, 0) + 1
                            )

        # Сортируем по частоте и берём топ-N
        result = {
            "mentions": self._get_top_items(aggregated["mentions"], 20),
            "tickers": self._get_top_items(aggregated["tickers"], 15),
            "urls": self._get_top_items(aggregated["urls"], 20),
            "domains": self._get_top_items(aggregated["domains"], 10),
            "dates": self._get_top_items(aggregated["dates"], 10),
            "numbers": aggregated["numbers"][:50],  # Первые 50 чисел
            "files": self._get_top_items(aggregated["files"], 20),
            "links": self._get_top_items(
                aggregated["links"], 30
            ),  # Все ссылки (текст + attachments)
            # Новые типы сущностей
            "persons": self._get_top_items(aggregated["persons"], 15),
            "organizations": self._get_top_items(aggregated["organizations"], 10),
            "locations": self._get_top_items(aggregated["locations"], 10),
            "crypto_addresses": self._get_top_items(aggregated["crypto_addresses"], 10),
            "telegram_channels": self._get_top_items(aggregated["telegram_channels"], 10),
            "telegram_bots": self._get_top_items(aggregated["telegram_bots"], 10),
        }

        return result

    def _get_top_items(
        self, items_dict: Dict[str, int], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Получение топ-N элементов по частоте"""
        sorted_items = sorted(items_dict.items(), key=lambda x: x[1], reverse=True)
        return [{"value": item, "count": count} for item, count in sorted_items[:top_n]]


# Удобная функция для быстрого использования
def extract_entities(text: str, chat_name: str = "") -> Dict[str, List[str]]:
    """Быстрое извлечение сущностей из текста"""
    extractor = EntityExtractor()
    return extractor.extract_entities(text, chat_name)


if __name__ == "__main__":
    # Тест модуля
    test_text = """
    @alice Привет! Смотри, TON сегодня вырос на 5%.
    Вот ссылка: https://coinmarketcap.com/currencies/toncoin/
    Встречаемся 15.10.2025 в 14:30. Бюджет: $1000 или 50 TON.
    Отправлю файл report.pdf позже.
    Bitcoin адрес: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
    Ethereum адрес: 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6
    """

    extractor = EntityExtractor()
    entities = extractor.extract_entities(test_text, "test_chat")

    print("Извлечённые сущности:")
    for category, values in entities.items():
        if values:
            print(f"  {category}: {values}")
    
    # Показываем статистику словарей
    if extractor.entity_dictionary:
        print("\nСтатистика словарей:")
        stats = extractor.entity_dictionary.get_entity_stats()
        for entity_type, stat in stats.items():
            print(f"  {entity_type}: {stat}")
