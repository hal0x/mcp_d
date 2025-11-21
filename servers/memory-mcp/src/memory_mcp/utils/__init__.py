"""Утилиты для работы с данными, текстом, системой и обработкой."""

from .data import (
    MessageExtractor,
    deduplicate_messages,
    load_json_file,
    load_jsonl_file,
)
from .processing import (
    compress_messages_context,
    format_datetime_display,
    optimize_context_dict,
    parse_datetime_utc,
    truncate_embedding_text,
    truncate_text,
    truncate_text_from_end,
)
from .system import (
    StateManager,
    find_project_root,
    is_valid_url,
    slugify,
)
from .text import (
    STOP_WORDS,
    calculate_rank_based_similarity,
    calculate_similarity_from_distance,
    get_tokenizer,
    get_word_variants,
    normalize_similarity_scores,
    tokenize_text,
)

__all__ = [
    # Text
    "tokenize_text",
    "get_tokenizer",
    "get_word_variants",
    "STOP_WORDS",
    "normalize_similarity_scores",
    "calculate_similarity_from_distance",
    "calculate_rank_based_similarity",
    # Data
    "MessageExtractor",
    "load_json_file",
    "load_jsonl_file",
    "deduplicate_messages",
    # System
    "slugify",
    "find_project_root",
    "StateManager",
    "is_valid_url",
    # Processing
    "parse_datetime_utc",
    "format_datetime_display",
    "truncate_text",
    "truncate_text_from_end",
    "optimize_context_dict",
    "compress_messages_context",
    "truncate_embedding_text",
]


