#!/usr/bin/env python3
"""
Парсинг структурированных саммаризаций для session_summarizer
"""

from typing import Any, Dict


def parse_summary_structure(summary_text: str) -> Dict[str, Any]:
    """
    Парсинг структурированной саммаризации

    Args:
        summary_text: Текст саммаризации от LLM

    Returns:
        Словарь с разделами
    """
    structure = {
        "context": "",
        "key_points": [],
        "important_items": [],
        "discussion": [],
        "decisions": [],
        "risks": [],
    }

    lines = summary_text.split("\n")
    current_section = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Определяем секцию
        if "## Контекст" in line or "## Context" in line:
            if current_section and current_text:
                # Сохраняем предыдущую секцию правильно
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "context"
            current_text = []
        elif "## Ключевые тезисы" in line or "## Key points" in line:
            if current_section and current_text:
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "key_points"
            current_text = []
        elif "## Что важно" in line or "## What matters" in line:
            if current_section and current_text:
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "important_items"
            current_text = []
        elif "## Ход дискуссии" in line or "## Discussion" in line:
            if current_section and current_text:
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "discussion"
            current_text = []
        elif (
            "## Решения" in line
            or "## Next steps" in line
            or "## Decisions" in line
        ):
            if current_section and current_text:
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "decisions"
            current_text = []
        elif (
            "## Риски" in line or "## Risks" in line or "## Open questions" in line
        ):
            if current_section and current_text:
                if current_section == "context":
                    structure[current_section] = "\n".join(current_text)
                else:
                    structure[current_section] = current_text
            current_section = "risks"
            current_text = []
        elif line.startswith("-") or line.startswith("*") or line.startswith("- ["):
            # Буллет-пойнт
            if current_section in [
                "key_points",
                "important_items",
                "discussion",
                "decisions",
                "risks",
            ]:
                current_text.append(line)
        elif current_section == "context":
            # Для контекста собираем весь текст
            current_text.append(line)

    # Сохраняем последнюю секцию
    if current_section and current_text:
        if current_section == "context":
            structure[current_section] = "\n".join(current_text)
        else:
            structure[current_section] = current_text

    return structure

