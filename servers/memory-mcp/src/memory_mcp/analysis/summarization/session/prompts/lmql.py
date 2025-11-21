#!/usr/bin/env python3
"""
Генерация структурированных саммаризаций через LMQL для session_summarizer
"""

import logging
from typing import Any, Dict, Optional

from ....core.lmql_adapter import LMQLAdapter

logger = logging.getLogger(__name__)


async def generate_summary_with_lmql(
    lmql_adapter: Optional[LMQLAdapter],
    prompt: str,
    chat_mode: str,
    language: str,
) -> Optional[Dict[str, Any]]:
    """
    Генерация структурированной саммаризации через LMQL.

    Args:
        lmql_adapter: LMQL адаптер
        prompt: Промпт для саммаризации
        chat_mode: Режим чата (channel/group)
        language: Язык вывода

    Returns:
        Словарь со структурой саммаризации или None при ошибке
    """
    if not lmql_adapter:
        return None

    try:
        if chat_mode == "channel":
            json_schema = """{
    "context": "[CONTEXT]",
    "key_points": [KEY_POINTS],
    "important_items": [IMPORTANT_ITEMS],
    "risks": [RISKS]
}"""
            constraints = """
    len(TOKENS(CONTEXT)) >= 20 and
    len(KEY_POINTS) <= 5 and
    len(IMPORTANT_ITEMS) >= 0 and
    len(RISKS) >= 0 and
    all(isinstance(kp, str) for kp in KEY_POINTS) and
    all(isinstance(ii, str) for ii in IMPORTANT_ITEMS) and
    all(isinstance(r, str) for r in RISKS)
"""
        else:
            json_schema = """{
    "context": "[CONTEXT]",
    "discussion": [DISCUSSION],
    "decisions": [DECISIONS],
    "risks": [RISKS]
}"""
            constraints = """
    len(TOKENS(CONTEXT)) >= 20 and
    len(DISCUSSION) <= 6 and
    len(DECISIONS) >= 0 and
    len(RISKS) >= 0 and
    all(isinstance(d, str) for d in DISCUSSION) and
    all(isinstance(dec, str) for dec in DECISIONS) and
    all(isinstance(r, str) for r in RISKS)
"""

        response_data = await lmql_adapter.execute_json_query(
            prompt=prompt,
            json_schema=json_schema,
            constraints=constraints,
            temperature=0.3,
            max_tokens=30000,
        )

        structure = {
            "context": response_data.get("context", ""),
            "key_points": response_data.get("key_points", []),
            "important_items": response_data.get("important_items", []),
            "discussion": response_data.get("discussion", []),
            "decisions": response_data.get("decisions", []),
            "risks": response_data.get("risks", []),
        }

        return structure

    except Exception as e:
        logger.error(f"Ошибка при генерации саммаризации через LMQL: {e}")
        return None

