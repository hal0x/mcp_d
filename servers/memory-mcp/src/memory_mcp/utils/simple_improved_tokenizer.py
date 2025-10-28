#!/usr/bin/env python3
"""
Простая и эффективная версия улучшенного токенизатора с обработкой чисел
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

# Паттерн для извлечения слов (включая русские буквы)
RUSSIAN_TOKEN_PATTERN = re.compile(r"[а-яё]+|[a-z]+|\d+", re.IGNORECASE | re.UNICODE)

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


class SimpleImprovedTokenizer:
    """Простой улучшенный токенизатор с обработкой чисел"""

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
            # Получаем нормальную форму слова через natasha
            normalized = self.morph_vocab[word.lower()]
            if normalized:
                return normalized
        except Exception as e:
            logger.debug(f"Ошибка нормализации слова '{word}': {e}")

        return word.lower()

    def is_stop_word(self, word: str) -> bool:
        """Проверка, является ли слово стоп-словом"""
        return word.lower() in STOP_WORDS

    def process_numbers(self, text: str) -> str:
        """Простая обработка чисел и денежных сумм"""
        result = text

        # Обработка денежных сумм
        # $120,000 -> MONEY_USD
        result = re.sub(r"\$[\d,]+(?:\.\d{2})?", "MONEY_USD", result)
        # €50,000 -> MONEY_EUR
        result = re.sub(r"€[\d,]+(?:\.\d{2})?", "MONEY_EUR", result)
        # ₽1,000,000 -> MONEY_RUB
        result = re.sub(r"₽[\d,]+(?:\.\d{2})?", "MONEY_RUB", result)

        # Обработка больших чисел
        # 1.5 млрд -> BILLION
        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:млрд|миллиард)",
            "BILLION",
            result,
            flags=re.IGNORECASE,
        )
        # 500 млн -> MILLION
        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:млн|миллион)",
            "MILLION",
            result,
            flags=re.IGNORECASE,
        )
        # 2,500 тыс -> THOUSAND
        result = re.sub(
            r"[\d,]+(?:\.\d+)?\s*(?:тыс|тысяч)", "THOUSAND", result, flags=re.IGNORECASE
        )

        # Обработка процентов
        # 15% -> PERCENTAGE
        result = re.sub(r"[\d,]+(?:\.\d+)?\s*%", "PERCENTAGE", result)

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

    def get_word_variants(self, word: str) -> Set[str]:
        """Получение вариантов слова (основа + окончания)"""
        variants = {word.lower()}

        if not self.use_morphology or not self.morph_vocab:
            return variants

        try:
            # Добавляем нормальную форму
            normalized = self.morph_vocab[word.lower()]
            if normalized:
                variants.add(normalized)
        except Exception as e:
            logger.debug(f"Ошибка получения вариантов для '{word}': {e}")

        return variants


# Глобальный экземпляр улучшенного токенизатора
_simple_improved_tokenizer_instance: Optional[SimpleImprovedTokenizer] = None


def get_simple_improved_tokenizer() -> SimpleImprovedTokenizer:
    """Получение глобального экземпляра улучшенного токенизатора"""
    global _simple_improved_tokenizer_instance
    if _simple_improved_tokenizer_instance is None:
        _simple_improved_tokenizer_instance = SimpleImprovedTokenizer()
    return _simple_improved_tokenizer_instance


def tokenize_text_simple_improved(text: str) -> List[str]:
    """Удобная функция для улучшенной токенизации текста"""
    return get_simple_improved_tokenizer().tokenize(text)


def normalize_word_simple_improved(word: str) -> str:
    """Удобная функция для нормализации слова"""
    return get_simple_improved_tokenizer().normalize_word(word)


def get_word_variants_simple_improved(word: str) -> Set[str]:
    """Удобная функция для получения вариантов слова"""
    return get_simple_improved_tokenizer().get_word_variants(word)
