"""Batch management utilities for the quality analyzer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    max_context_tokens: int = 131072  # Для gpt-oss-20b
    system_prompt_reserve: float = 0.2
    max_batch_size: int = 10
    max_query_tokens: int = 6000


class BatchManager:
    """Calculates optimal batch sizes for query processing."""

    def __init__(self, config: BatchConfig):
        self.config = config

    def estimate_query_tokens(self, query_text: str) -> int:
        # Approximate conversion: 4 characters ~ 1 token
        return max(50, len(query_text) // 4)

    def suggest_batch_size(self, queries: Sequence[str]) -> int:
        if not queries:
            return 1

        avg_tokens = sum(self.estimate_query_tokens(q) for q in queries) / len(queries)
        available_tokens = int(
            self.config.max_context_tokens * (1 - self.config.system_prompt_reserve)
        )
        batch_size = max(1, int(available_tokens // max(avg_tokens, 1)))
        return min(batch_size, self.config.max_batch_size)

    def split(self, queries: Sequence[dict]) -> Iterable[list[dict]]:
        batch_size = self.suggest_batch_size([q.get("query", "") for q in queries])
        logger.debug(
            "Используется размер батча %d для %d запросов", batch_size, len(queries)
        )
        for i in range(0, len(queries), batch_size):
            yield list(queries[i : i + batch_size])

    def normalize_queries(self, queries: Sequence[dict]) -> None:
        max_chars = self.config.max_query_tokens * 4
        for query in queries:
            text = query.get("query", "")
            if len(text) > max_chars:
                logger.warning(
                    "Запрос обрезан до %d символов для соблюдения лимита", max_chars
                )
                query["query"] = text[:max_chars] + "..."
