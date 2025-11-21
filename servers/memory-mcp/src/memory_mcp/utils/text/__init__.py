"""Утилиты для работы с текстом."""

from .russian_tokenizer import (
    STOP_WORDS,
    get_tokenizer,
    get_word_variants,
    tokenize_text,
)
from .similarity import (
    calculate_rank_based_similarity,
    calculate_similarity_from_distance,
    normalize_similarity_scores,
)

__all__ = [
    "tokenize_text",
    "get_tokenizer",
    "get_word_variants",
    "STOP_WORDS",
    "normalize_similarity_scores",
    "calculate_similarity_from_distance",
    "calculate_rank_based_similarity",
]


