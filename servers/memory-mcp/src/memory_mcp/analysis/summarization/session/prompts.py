#!/usr/bin/env python3
"""
Модуль для создания и парсинга промптов для session_summarizer
Объединяет функциональность из prompts/builder.py, prompts/completeness.py,
prompts/fallback.py, prompts/lmql.py, prompts/parser.py
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ....core.adapters.lmql_adapter import LMQLAdapter
from .utils import detect_chat_mode, format_message_bullet, truncate_text

logger = logging.getLogger(__name__)


# ============================================================================
# Из prompts/builder.py
# ============================================================================

def create_summarization_prompt(
    conversation_text: str,
    chat: str,
    language: str,
    session: Dict[str, Any],
    chat_mode: str,
    previous_context: Dict[str, Any],
    extended_context: Optional[Dict[str, Any]] = None,
    instruction_manager=None,
    incremental_context_manager=None,
) -> str:
    """
    Создание промпта для саммаризации с контекстом предыдущих сессий

    Args:
        conversation_text: Текст разговора
        chat: Название чата
        language: Язык вывода
        session: Сессия с метаданными
        chat_mode: Режим чата (group/channel)
        previous_context: Контекст предыдущих сессий
        extended_context: Расширенный контекст для малых сессий
        instruction_manager: Менеджер инструкций
        incremental_context_manager: Менеджер инкрементального контекста

    Returns:
        Промпт для LLM
    """
    lang_instruction = "на русском языке" if language == "ru" else "in English"

    # Формируем контекстную часть промпта с ограничением размера
    context_section = ""
    if previous_context["previous_sessions_count"] > 0:
        context_section = f"""
## Контекст предыдущих сессий
Это сессия #{previous_context['previous_sessions_count'] + 1} в чате "{chat}".

Предыдущие сессии:
"""
        # Ограничиваем количество предыдущих сессий для предотвращения переполнения
        max_timeline_items = (
            12  # Увеличиваем до 12 предыдущих сессий для максимального контекста
        )
        timeline_items = previous_context["session_timeline"][:max_timeline_items]

        for timeline_item in timeline_items:
            context_section += f"- {timeline_item['session_id']}: {timeline_item['context_summary']}\n"

        # Если есть еще сессии, добавляем краткое упоминание
        if len(previous_context["session_timeline"]) > max_timeline_items:
            remaining_count = (
                len(previous_context["session_timeline"]) - max_timeline_items
            )
            context_section += f"- ... и еще {remaining_count} сессий\n"

        if previous_context["recent_context"]:
            # Ограничиваем размер недавнего контекста
            recent_context = previous_context["recent_context"]
            if (
                len(recent_context) > 100000
            ):  # Увеличиваем до 100000 символов (~25000 токенов) для эффективного использования большого контекста
                recent_context = recent_context[:100000] + "..."
            context_section += f"\nНедавний контекст: {recent_context}\n"

        if previous_context["ongoing_decisions"]:
            context_section += "\nТекущие решения из предыдущих сессий:\n"
            # Ограничиваем количество решений
            max_decisions = (
                12  # Увеличиваем до 12 решений для максимального контекста
            )
            for decision in previous_context["ongoing_decisions"][:max_decisions]:
                context_section += f"- {decision}\n"
            if len(previous_context["ongoing_decisions"]) > max_decisions:
                context_section += f"- ... и еще {len(previous_context['ongoing_decisions']) - max_decisions} решений\n"

        if previous_context.get("plans_and_tasks"):
            context_section += "\nПланы и задачи из предыдущих сессий:\n"
            max_plans = 15  # Максимум 15 планов и задач
            for item in previous_context["plans_and_tasks"][:max_plans]:
                context_section += f"- {item}\n"
            if len(previous_context["plans_and_tasks"]) > max_plans:
                context_section += f"- ... и еще {len(previous_context['plans_and_tasks']) - max_plans} пунктов\n"

        if previous_context.get("active_discussions"):
            context_section += "\nАктивные обсуждения из предыдущих сессий:\n"
            max_discussions = 10  # Максимум 10 обсуждений
            for discussion in previous_context["active_discussions"][:max_discussions]:
                context_section += f"- {discussion}\n"
            if len(previous_context["active_discussions"]) > max_discussions:
                context_section += f"- ... и еще {len(previous_context['active_discussions']) - max_discussions} обсуждений\n"

        if previous_context.get("active_risks"):
            context_section += "\nПроблемы и открытые вопросы из предыдущих сессий:\n"
            # Ограничиваем количество рисков
            max_risks = 12  # Увеличиваем до 12 рисков для максимального контекста
            for risk in previous_context["active_risks"][:max_risks]:
                context_section += f"- {risk}\n"
            if len(previous_context["active_risks"]) > max_risks:
                context_section += f"- ... и еще {len(previous_context['active_risks']) - max_risks} проблем\n"
        
        # Для обратной совместимости оставляем open_risks
        if previous_context.get("open_risks") and not previous_context.get("active_risks"):
            context_section += "\nОткрытые риски из предыдущих сессий:\n"
            # Ограничиваем количество рисков
            max_risks = 12  # Увеличиваем до 12 рисков для максимального контекста
            for risk in previous_context["open_risks"][:max_risks]:
                context_section += f"- {risk}\n"
            if len(previous_context["open_risks"]) > max_risks:
                context_section += f"- ... и еще {len(previous_context['open_risks']) - max_risks} рисков\n"

        context_section += "\n"

    # Добавляем накопительный контекст чата с ограничением размера
    chat_context_section = ""
    if previous_context.get("chat_context"):
        chat_context = previous_context["chat_context"]
        # Ограничиваем размер контекста чата
        if len(chat_context) > 50000:  # Увеличено до 50000 символов (~12500 токенов) для эффективного использования большого контекста
            chat_context = chat_context[:50000] + "..."
        chat_context_section = f"""
## Образ чата
{chat_context}

"""

    # Добавляем расширенный контекст для малых сессий
    extended_context_section = ""
    if extended_context and extended_context.get("previous_messages_count", 0) > 0:
        if incremental_context_manager:
            extended_context_text = (
                incremental_context_manager.format_context_for_prompt(
                    extended_context, max_context_length=100000  # Увеличено для эффективного использования большого контекста
                )
            )
            extended_context_section = f"""
## Расширенный контекст (для малой сессии)
{extended_context_text}

"""

    if chat_mode == "channel":
        mode_instructions = """
Создай структурированную выжимку для канала (один или несколько авторов публикуют материалы, дискуссии мало):

## Контекст
[1-2 предложения: тема публикаций, цель, период]

## Ключевые тезисы
- [Тезис 1]
- [Тезис 2]
- [Тезис 3]
[до 5 пунктов]

## Что важно
- [Факты/цифры/ссылки]

## Риски / Вопросы
- [если есть]
"""
    else:
        mode_instructions = """
Создай структурированную саммаризацию группового обсуждения:

## Контекст
[1-3 предложения о предпосылках и цели беседы]

## Ход дискуссии
- [Буллет 1]
- [Буллет 2]
- [Буллет 3]
[до 6 пунктов]

## Решения / Next steps
- [ ] [Действие] — **owner:** @[username] — **due:** [дата/время] — pri: [P1/P2/P3]

## Риски / Открытые вопросы
- [Риск/вопрос]
"""

    custom_instruction = ""
    if instruction_manager:
        custom_instruction = instruction_manager.get_instruction(chat, chat_mode)
    custom_section = ""
    if custom_instruction:
        custom_section = (
            f"\nДополнительная инструкция:\n{custom_instruction.strip()}\n"
        )

    prompt = f"""{chat_context_section}{extended_context_section}{context_section}Проанализируй следующий разговор из Telegram чата "{chat}".

Разговор:
{conversation_text}

Создай структурированную саммаризацию {lang_instruction}. Учти тип коммуникации: {('канал' if chat_mode=='channel' else 'групповой чат')}.

{mode_instructions}
{custom_section}

Важно:
- Будь конкретным и точным
- Отмечай риски и открытые вопросы
- ОБЯЗАТЕЛЬНО учитывай контекст предыдущих сессий при анализе - связывай текущие события с предыдущими
- Если в контексте упоминаются важные события (болезни, планы, решения), обязательно отрази их влияние на текущую сессию
- ОБРАЩАЙ ВНИМАНИЕ на повторяющиеся сообщения ("повторено Nx"): это может значить высокую важность или spam/вариации
- При малом количестве сообщений используй расширенный контекст для лучшего понимания
- Связывай текущие действия с предыдущими решениями и рисками

Саммаризация:"""

    # Проверяем размер промпта и логируем предупреждение если он слишком большой
    estimated_tokens = len(prompt) // 4
    if (
        estimated_tokens > 30000
    ):  # Предупреждение при превышении 30k токенов (близко к лимиту 32k)
        logger.warning(
            f"⚠️  Промпт для саммаризации сессии {session.get('session_id', 'unknown')} "
            f"очень длинный: ~{estimated_tokens} токенов. "
            f"LLM клиент автоматически разобьет его на части при необходимости."
        )

    return prompt


# ============================================================================
# Из prompts/completeness.py
# ============================================================================

def ensure_summary_completeness(
    messages: List[Dict[str, Any]],
    chat: str,
    structure: Dict[str, Any],
    strict_mode: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """
    Проверяет и при необходимости дополняет саммаризацию эвристическими данными.

    Args:
        messages: Исходные сообщения сессии
        chat: Название чата
        structure: Структура, полученная от LLM
        strict_mode: Если True, выбрасывает исключение вместо fallback

    Returns:
        (дополненная структура, признак использования fallback)
    """
    # context может быть строкой или списком
    context_raw = structure.get("context") or ""
    if isinstance(context_raw, list):
        context_text = "\n".join(context_raw).strip()
    else:
        context_text = context_raw.strip()

    key_points = structure.get("key_points") or []
    important_items = structure.get("important_items") or []
    discussion = structure.get("discussion") or []
    decisions = structure.get("decisions") or []
    risks = structure.get("risks") or []

    chat_mode = detect_chat_mode(messages)

    needs_context = len(context_text) < 40
    needs_discussion = len(discussion) < 2
    needs_decisions = len(decisions) == 0
    needs_risks = len(risks) == 0

    if chat_mode == "channel":
        needs_key_points = len(key_points) == 0
        needs_important_items = len(important_items) == 0
    else:
        needs_key_points = False
        needs_important_items = False

    if not any(
        [
            needs_context,
            needs_key_points,
            needs_important_items,
            needs_discussion,
            needs_decisions,
            needs_risks,
        ]
    ):
        return structure, False

    if strict_mode:
        raise RuntimeError(
            f"LLM вернул пустую или неполную структуру саммаризации для чата '{chat}'. "
            "Проверьте конфигурацию LLM клиента."
        )

    fallback = build_fallback_structure(messages, chat, chat_mode)
    patched = dict(structure)

    if needs_context:
        patched["context"] = fallback["context"]
    if needs_key_points:
        patched["key_points"] = fallback.get("key_points", [])
    if needs_important_items:
        patched["important_items"] = fallback.get("important_items", [])
    if needs_discussion:
        patched["discussion"] = fallback["discussion"]
    if needs_decisions:
        patched["decisions"] = fallback["decisions"]
    if needs_risks:
        patched["risks"] = fallback["risks"]

    return patched, True


# ============================================================================
# Из prompts/fallback.py
# ============================================================================

def build_fallback_structure(
    messages: List[Dict[str, Any]], chat: str, chat_mode: str
) -> Dict[str, Any]:
    """
    Формирует эвристическую саммаризацию, если LLM дал пустой результат.
    """
    valid_messages = [m for m in messages if (m.get("text") or "").strip()]

    if not valid_messages:
        if chat_mode == "channel":
            return {
                "context": f"Автосаммаризация (fallback) для канала {chat}: сообщений не найдено.",
                "key_points": ["- Данных для анализа не обнаружено"],
                "important_items": ["- Данных для анализа не обнаружено"],
                "discussion": ["- Данных для анализа не обнаружено"],
                "decisions": [],
                "risks": [
                    "- Автопроверка: риски не обнаружены; требуется ручная проверка."
                ],
            }
        else:
            return {
                "context": f"Автосаммаризация (fallback) для чата {chat}: сообщений не найдено.",
                "discussion": ["- Данных для анализа не обнаружено"],
                "decisions": [
                    "- [ ] Автопроверка: действия не обнаружены; требуется ручная проверка."
                ],
                "risks": [
                    "- Автопроверка: риски не обнаружены; требуется ручная проверка."
                ],
            }

    from .utils import (
        collect_participants,
        select_key_messages,
        select_messages_with_keywords,
    )

    participants = collect_participants(valid_messages)
    first_text = truncate_text(valid_messages[0].get("text", ""), 220)
    last_text = truncate_text(valid_messages[-1].get("text", ""), 220)
    context_lines = [
        f'Автосаммаризация (fallback) по чату "{chat}". Сообщений: {len(valid_messages)}.',
    ]
    if participants:
        context_lines.append(f"Активные участники: {', '.join(participants)}")
    if first_text:
        context_lines.append(f"Старт обсуждения: {first_text}")
    if last_text and last_text != first_text:
        context_lines.append(f"Финал обсуждения: {last_text}")

    discussion_limit = 5 if chat_mode == "channel" else 4
    discussion_msgs = select_key_messages(
        valid_messages, limit=discussion_limit
    )
    if len(discussion_msgs) < 2:
        discussion_msgs = valid_messages[: min(2, len(valid_messages))]
    discussion_lines = [
        format_message_bullet(msg, prefix="- ") for msg in discussion_msgs
    ]

    decision_msgs = select_messages_with_keywords(
        valid_messages,
        keywords=[
            "нужно",
            "надо",
            "должн",
            "давайте",
            "решим",
            "todo",
            "should",
            "must",
            "plan",
        ],
        limit=3,
    )
    if chat_mode == "channel":
        # В каналах решения обычно неуместны
        decision_lines = (
            []
            if decision_msgs == []
            else [
                format_message_bullet(m, prefix="- ") for m in decision_msgs
            ]
        )
    elif decision_msgs:
        decision_lines = [
            format_message_bullet(msg, prefix="- [ ] ")
            for msg in decision_msgs
        ]
    else:
        decision_lines = [
            "- [ ] Автопроверка: явных действий не зафиксировано; требуется ручная проверка."
        ]

    risk_msgs = select_messages_with_keywords(
        valid_messages,
        keywords=["риск", "проблем", "опас", "сомн", "issue", "блок", "concern"],
        limit=3,
    )
    if risk_msgs:
        risk_lines = [
            format_message_bullet(msg, prefix="- ") for msg in risk_msgs
        ]
    else:
        risk_lines = [
            "- Автопроверка: явных рисков в сообщениях не найдено; проверить вручную."
        ]

    if chat_mode == "channel":
        # Ключевые тезисы из первых сообщений
        key_points = []
        for msg in valid_messages[:3]:
            text = truncate_text(msg.get("text", ""), 100)
            if text:
                key_points.append(f"- {text}")

        # Важные моменты: ссылки, даты, имена, ключевые слова
        important_items = []
        important_keywords = [
            "важно",
            "критично",
            "срочно",
            "внимание",
            "attention",
            "important",
            "critical",
            "urgent",
            "required",
            "must",
            "should",
            "update",
            "upgrade",
            "vote",
            "voting",
            "action required",
            "mandatory",
            "scheduled",
            "deadline",
            "breaking",
            "announcement",
        ]

        for msg in valid_messages:
            text = msg.get("text", "")
            text_lower = text.lower()

            if any(keyword in text_lower for keyword in important_keywords):
                important_items.append(f"- {truncate_text(text, 80)}")
            # Сообщения с датами и временем (часто важные объявления)
            elif any(
                pattern in text
                for pattern in ["UTC", "GMT", "at ", "on ", "2024", "2025"]
            ):
                important_items.append(f"- {truncate_text(text, 80)}")

            if len(important_items) >= 3:
                break

        if not important_items:
            for msg in valid_messages[:2]:
                text = truncate_text(msg.get("text", ""), 80)
                if text:
                    important_items.append(f"- {text}")

        if not important_items:
            important_items = [
                "- Автопроверка: важные моменты не выделены; требуется ручная проверка."
            ]

        return {
            "context": "\n".join(context_lines),
            "key_points": key_points[:5],
            "important_items": important_items[:5],
            "discussion": discussion_lines,
            "decisions": decision_lines,
            "risks": risk_lines,
        }
    else:
        return {
            "context": "\n".join(context_lines),
            "discussion": discussion_lines,
            "decisions": decision_lines,
            "risks": risk_lines,
        }


# ============================================================================
# Из prompts/lmql.py
# ============================================================================

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


# ============================================================================
# Из prompts/parser.py
# ============================================================================

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

