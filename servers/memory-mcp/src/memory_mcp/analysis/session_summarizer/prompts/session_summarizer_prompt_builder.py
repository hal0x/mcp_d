#!/usr/bin/env python3
"""
Построение промптов для session_summarizer
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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

