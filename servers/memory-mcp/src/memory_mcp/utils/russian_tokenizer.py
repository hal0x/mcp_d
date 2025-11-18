#!/usr/bin/env python3
"""
Улучшенная токенизация для русского языка
Поддерживает морфологический анализ и нормализацию слов
"""

import logging
import re
from functools import lru_cache
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Попытка импорта natasha, если не установлен - используем fallback
try:
    from natasha import MorphVocab

    MORPH_AVAILABLE = True
except ImportError:
    MORPH_AVAILABLE = False
    logger.warning("natasha не установлен, используется упрощенная токенизация")

# Паттерн для извлечения слов (включая русские буквы и специальные токены)
RUSSIAN_TOKEN_PATTERN = re.compile(
    r"(?:MONEY|BILLION|MILLION|THOUSAND|PERCENTAGE)_[A-Z]+|(?:AMOUNT|VALUE)_[a-z0-9.]+|[а-яё]+|[a-z]+|\d+",
    re.IGNORECASE | re.UNICODE,
)

# Минимальная длина токена
MIN_TOKEN_LENGTH = 3

# Стоп-слова для русского языка
RUSSIAN_STOP_WORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "тогда",
    "кто",
    "этот",
    "того",
    "потому",
    "этого",
    "какой",
    "совсем",
    "ним",
    "здесь",
    "этом",
    "один",
    "почти",
    "мой",
    "тем",
    "чтобы",
    "нее",
    "сейчас",
    "были",
    "куда",
    "зачем",
    "всех",
    "никогда",
    "можно",
    "при",
    "наконец",
    "два",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
}

# Стоп-слова для английского языка
ENGLISH_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
}

# Объединенные стоп-слова
STOP_WORDS = RUSSIAN_STOP_WORDS | ENGLISH_STOP_WORDS


class RussianTokenizer:
    """Улучшенный токенизатор для русского языка"""

    def __init__(self, use_morphology: bool = True):
        self.use_morphology = use_morphology and MORPH_AVAILABLE
        self.morph_vocab = None

        if self.use_morphology:
            try:
                self.morph_vocab = MorphVocab()
                logger.info("Инициализирован морфологический словарь natasha")
            except Exception as e:
                logger.warning(f"Ошибка инициализации natasha: {e}")
                self.use_morphology = False

    @lru_cache(maxsize=10000)
    def normalize_word(self, word: str) -> str:
        """Нормализация слова с использованием морфологии"""
        if not self.use_morphology or not self.morph_vocab:
            return word.lower()

        try:
            # Получаем лексему (все формы слова) через natasha
            # Первый элемент лексемы - это нормальная форма
            lexeme = self.morph_vocab.get_lexeme(word.lower())
            if lexeme and len(lexeme) > 0:
                normalized = lexeme[0]
                if normalized:
                    return normalized
        except Exception as e:
            logger.debug(f"Ошибка нормализации слова '{word}': {e}")

        return word.lower()

    def is_stop_word(self, word: str) -> bool:
        """Проверка, является ли слово стоп-словом"""
        return word.lower() in STOP_WORDS

    def process_numbers(self, text: str) -> str:
        """Улучшенная обработка чисел и денежных сумм с разделением на отдельные токены"""
        result = text

        # Обработка денежных сумм с разделением на тип и значение
        # $120,000 -> MONEY_USD AMOUNT_120000
        def replace_money_usd(match):
            amount = match.group(0)[1:].replace(",", "")  # Убираем $ и запятые
            return f"MONEY_USD AMOUNT_{amount}"

        result = re.sub(r"\$[\d,]+(?:\.\d{2})?", replace_money_usd, result)

        # €50,000 -> MONEY_EUR AMOUNT_50000
        def replace_money_eur(match):
            amount = match.group(0)[1:].replace(",", "")  # Убираем € и запятые
            return f"MONEY_EUR AMOUNT_{amount}"

        result = re.sub(r"€[\d,]+(?:\.\d{2})?", replace_money_eur, result)

        # ₽1,000,000 -> MONEY_RUB AMOUNT_1000000
        def replace_money_rub(match):
            amount = match.group(0)[1:].replace(",", "")  # Убираем ₽ и запятые
            return f"MONEY_RUB AMOUNT_{amount}"

        result = re.sub(r"₽[\d,]+(?:\.\d{2})?", replace_money_rub, result)

        def normalize_amount_value(raw: str) -> str:
            cleaned = raw.replace("\u00a0", "").replace(" ", "")
            if "." not in cleaned and cleaned.count(",") == 1:
                return cleaned.replace(",", ".")
            return cleaned.replace(",", "")

        currency_word_patterns = [
            ("USD", r"(?P<amount>[\d\s,.]+)\s*(?:usd|доллар(?:ов)?|бакс(?:ов)?)"),
            ("EUR", r"(?P<amount>[\d\s,.]+)\s*(?:eur|евро)"),
            ("RUB", r"(?P<amount>[\d\s,.]+)\s*(?:руб(?:ля|лей)?|р(?:уб)?\.?|rub)"),
        ]

        for currency_code, pattern in currency_word_patterns:
            def replace_currency(match, code=currency_code):
                amount = normalize_amount_value(match.group("amount"))
                return f"MONEY_{code} AMOUNT_{amount}"

            result = re.sub(pattern, replace_currency, result, flags=re.IGNORECASE)

        # Обработка больших чисел с разделением на тип и значение
        # 1.5 млрд -> BILLION VALUE_1.5
        def replace_billion(match):
            number = match.group(0).split()[0].replace(",", "")
            return f"BILLION VALUE_{number}"

        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:млрд|миллиард)",
            replace_billion,
            result,
            flags=re.IGNORECASE,
        )

        # 500 млн -> MILLION VALUE_500
        def replace_million(match):
            number = match.group(0).split()[0].replace(",", "")
            return f"MILLION VALUE_{number}"

        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:млн|миллион)",
            replace_million,
            result,
            flags=re.IGNORECASE,
        )

        # 2,500 тыс -> THOUSAND VALUE_2500
        def replace_thousand(match):
            number = match.group(0).split()[0].replace(",", "")
            return f"THOUSAND VALUE_{number}"

        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:тыс|тысяч)",
            replace_thousand,
            result,
            flags=re.IGNORECASE,
        )

        # Обработка процентов с разделением на тип и значение
        # 15% -> PERCENTAGE VALUE_15
        def replace_percentage(match):
            number = match.group(0).replace("%", "").replace(",", "")
            return f"PERCENTAGE VALUE_{number}"

        result = re.sub(r"[\d,]+(?:\.\d+)?\s*%", replace_percentage, result)

        return result

    def extract_tokens(self, text: str) -> List[str]:
        """Извлечение токенов из текста с улучшенной обработкой чисел"""
        if not text:
            return []

        # Обрабатываем числа
        processed_text = self.process_numbers(text)

        # Извлекаем слова с помощью регулярного выражения
        words = RUSSIAN_TOKEN_PATTERN.findall(processed_text.lower())

        # Фильтруем и нормализуем токены
        tokens = []
        for word in words:
            if len(word) >= MIN_TOKEN_LENGTH and not self.is_stop_word(word):
                normalized = self.normalize_word(word)
                if normalized and len(normalized) >= MIN_TOKEN_LENGTH:
                    tokens.append(normalized)

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Основной метод токенизации"""
        return self.extract_tokens(text)

    @lru_cache(maxsize=10000)
    def get_word_variants(self, word: str) -> Set[str]:
        """Получение вариантов слова (исходное слово + нормальная форма)
        
        Возвращает множество вариантов слова для расширения поисковых запросов.
        Включает исходное слово и его нормальную форму (лемму).
        """
        word_lower = word.lower()
        variants = {word_lower}

        if not self.use_morphology or not self.morph_vocab:
            return variants

        try:
            # Получаем лексему (все формы слова) через natasha
            # Первый элемент лексемы - это нормальная форма
            lexeme = self.morph_vocab.get_lexeme(word_lower)
            if lexeme and len(lexeme) > 0:
                normalized = lexeme[0]
                if normalized and normalized != word_lower:
                    variants.add(normalized)
        except Exception as e:
            logger.debug(f"Ошибка получения вариантов для '{word}': {e}")

        return variants


# Глобальный экземпляр токенизатора
_tokenizer_instance: Optional[RussianTokenizer] = None


def get_tokenizer() -> RussianTokenizer:
    """Получение глобального экземпляра токенизатора"""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = RussianTokenizer()
    return _tokenizer_instance


def tokenize_text(text: str) -> List[str]:
    """Удобная функция для токенизации текста"""
    return get_tokenizer().tokenize(text)


def normalize_word(word: str) -> str:
    """Удобная функция для нормализации слова"""
    return get_tokenizer().normalize_word(word)


def get_word_variants(word: str) -> Set[str]:
    """Удобная функция для получения вариантов слова"""
    return get_tokenizer().get_word_variants(word)


# Функции для обратной совместимости
def _tokenize_legacy(text: str) -> List[str]:
    """Старая функция токенизации для обратной совместимости"""
    if not text:
        return []
    return [
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def _tokenize_enhanced(text: str) -> List[str]:
    """Улучшенная функция токенизации"""
    return tokenize_text(text)


# Экспортируем улучшенную функцию как основную
_tokenize = _tokenize_enhanced
