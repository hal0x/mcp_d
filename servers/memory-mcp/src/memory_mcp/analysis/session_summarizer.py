#!/usr/bin/env python3
"""
Модуль для структурной саммаризации сессий
Согласно спецификации TelegramDumpManager_Spec.md
"""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..core.lmstudio_client import LMStudioEmbeddingClient
from ..utils.naming import slugify
from .context_manager import ContextManager
from .entity_extraction import EntityExtractor
from .incremental_context_manager import IncrementalContextManager
from .instruction_manager import InstructionManager
from .quality_evaluator import IterativeRefiner, QualityEvaluator
from .session_segmentation import SessionSegmenter

logger = logging.getLogger(__name__)

SESSION_SUMMARY_VERSION = "1.0.0"


# --- Доменные эвристики ---

CRYPTO_TICKERS = {
    "BTC",
    "ETH",
    "TON",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "TRX",
    "MATIC",
    "DOT",
    "AVAX",
    "LTC",
    "USDT",
    "USDC",
}

CRYPTO_TERMS = {
    "sec",
    "lawsuit",
    "exploit",
    "airdrop",
    "staking",
    "chain halt",
    "mainnet",
    "testnet",
    "governance",
    "token",
    "ledger",
    "defi",
    "dex",
}

CRYPTO_EXCHANGES = {
    "binance",
    "okx",
    "okex",
    "bybit",
    "coinbase",
    "kraken",
    "kucoin",
    "bitfinex",
    "huobi",
    "gate.io",
    "mexc",
    "bitget",
}

SCI_TECH_TERMS = {
    "arxiv",
    "preprint",
    "paper",
    "dataset",
    "benchmark",
    "sota",
    "research",
    "doi",
    "publication",
    "peer review",
    "open source",
}

SCI_TECH_PATTERNS = [
    re.compile(r"arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE),
]

GEOPOLITICS_PATTERNS = [
    (re.compile(r"росси", re.IGNORECASE), "Russia"),
    (re.compile(r"украин", re.IGNORECASE), "Ukraine"),
    (re.compile(r"кита", re.IGNORECASE), "China"),
    (re.compile(r"сша|usa|u\.s\.", re.IGNORECASE), "USA"),
    (
        re.compile(r"евросоюз|европейский союз|european union|\bEU\b", re.IGNORECASE),
        "EU",
    ),
    (re.compile(r"\bUN\b|организац.. объединенных нац", re.IGNORECASE), "UN"),
    (re.compile(r"\bNATO\b", re.IGNORECASE), "NATO"),
    (re.compile(r"санкц", re.IGNORECASE), "Sanctions"),
    (re.compile(r"министерств|ministry", re.IGNORECASE), "Ministry"),
    (
        re.compile(r"пресс(-| )?служб|press-service", re.IGNORECASE),
        "Official statement",
    ),
    (re.compile(r"правительств", re.IGNORECASE), "Government"),
    (re.compile(r"парламент", re.IGNORECASE), "Parliament"),
]


class SessionSummarizer:
    """Класс для структурной саммаризации сессий"""

    def __init__(
        self,
        embedding_client: Optional[LMStudioEmbeddingClient] = None,
        summaries_dir: Path = Path("artifacts/reports"),
        instruction_manager: Optional[InstructionManager] = None,
        enable_quality_check: bool = True,
        enable_iterative_refinement: bool = True,
        min_quality_score: float = 80.0,
    ):
        """
        Инициализация саммаризатора

        Args:
            embedding_client: Клиент для генерации эмбеддингов и текста (LM Studio, если None, создаётся новый)
            summaries_dir: Директория с саммаризациями для контекста
            instruction_manager: Менеджер специальных инструкций
            enable_quality_check: Включить проверку качества
            enable_iterative_refinement: Включить итеративное улучшение
            min_quality_score: Минимальный приемлемый балл качества
        """
        self.embedding_client = embedding_client or LMStudioEmbeddingClient()
        self.entity_extractor = EntityExtractor()
        self.session_segmenter = SessionSegmenter()
        self.context_manager = ContextManager(summaries_dir)
        self.incremental_context_manager = IncrementalContextManager()
        self.instruction_manager = instruction_manager or InstructionManager()

        # Система оценки качества
        self.enable_quality_check = enable_quality_check
        self.enable_iterative_refinement = enable_iterative_refinement
        self.quality_evaluator = QualityEvaluator(min_quality_score=min_quality_score)
        # Используем тот же порог для улучшения (или чуть ниже для гарантии достижимости)
        self.iterative_refiner = IterativeRefiner(
            self,
            max_iterations=5,
            target_score=max(
                85.0, min_quality_score - 2.0
            ),  # Целевой балл чуть ниже min_quality для margin
        )

    async def summarize_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Саммаризация одной сессии

        Args:
            session: Сессия для саммаризации

        Returns:
            Словарь с саммаризацией и метаданными
        """
        messages = session["messages"]
        session_id = session["session_id"]
        chat = session.get("chat", "Unknown")
        dominant_language = session.get("dominant_language", "ru")

        logger.info(f"Саммаризация сессии {session_id} ({len(messages)} сообщений)")

        # Определяем тип коммуникации (группа/канал)
        chat_mode = self._detect_chat_mode(messages)

        # Получаем контекст из предыдущих сессий
        previous_context = self.context_manager.get_previous_context(chat, session_id)

        # Получаем расширенный контекст для малых сессий
        extended_context = None
        if len(messages) < 20:  # Для сессий с менее чем 20 сообщениями
            extended_context = (
                self.incremental_context_manager.get_extended_context_for_session(
                    chat, messages, context_window_hours=24, max_previous_messages=50
                )
            )
            logger.info(
                f"Используем расширенный контекст для малой сессии {session_id}: "
                f"{extended_context['previous_messages_count']} предыдущих сообщений, "
                f"{extended_context['previous_sessions_count']} предыдущих сессий"
            )

        # Извлекаем сущности из сессии
        entities = self.entity_extractor.extract_from_messages(messages)

        # Подготавливаем текст разговора
        conversation_text = self._prepare_conversation_text(messages)

        # Создаём промпт для саммаризации с контекстом
        prompt = self._create_summarization_prompt(
            conversation_text,
            chat,
            dominant_language,
            session,
            chat_mode,
            previous_context,
            extended_context,
        )

        # Генерируем саммаризацию через LLM
        async with self.embedding_client:
            summary_text = await self.embedding_client.generate_summary(
                prompt=prompt,
                temperature=0.3,
                max_tokens=8000,
                top_p=0.93,
                presence_penalty=0.05,
            )

        # Парсим структурированную саммаризацию и дополняем пропуски при необходимости
        summary_structure = self._parse_summary_structure(summary_text)
        summary_structure, fallback_used = self._ensure_summary_completeness(
            messages, chat, summary_structure
        )

        if fallback_used:
            logger.debug(
                f"Саммаризация {session_id} дополнена эвристиками (это нормально)"
            )

        # Извлекаем Action Items после возможного дополнения саммаризации
        action_items = await self._extract_action_items(messages, summary_structure)

        # Формируем итоговую саммаризацию
        legacy_summary = {
            "session_id": session_id,
            "chat": chat,
            "message_count": len(messages),
            "participants": session.get("participants", []),
            "time_range_bkk": session.get("time_range_bkk", ""),
            "start_time_utc": session.get("start_time_utc", ""),
            "end_time_utc": session.get("end_time_utc", ""),
            "dominant_language": dominant_language,
            "entities": entities,
            "chat_mode": chat_mode,
            "context": summary_structure.get("context", ""),
            "key_points": summary_structure.get("key_points", []),
            "important_items": summary_structure.get("important_items", []),
            "discussion": summary_structure.get("discussion", []),
            "decisions_next": action_items,
            "risks_open": summary_structure.get("risks", []),
            "links_artifacts": self._format_links_artifacts(entities),
            "raw_summary": summary_text,
            "fallback_used": fallback_used,
        }

        metrics = None

        # === ПРОВЕРКА КАЧЕСТВА И ИТЕРАТИВНОЕ УЛУЧШЕНИЕ ===
        if self.enable_quality_check:
            # Оцениваем качество саммаризации
            metrics = self.quality_evaluator.evaluate(
                legacy_summary, expected_language=dominant_language
            )

            logger.info(f"Качество саммаризации {session_id}: {metrics.score:.1f}/100")
            logger.info(
                f"  Контекст: {metrics.context_length} символов, Дискуссия: {metrics.discussion_count} пунктов"
            )
            logger.info(
                f"  Решения: {metrics.decisions_count}, Риски: {metrics.risks_count}, Ссылки: {metrics.links_count}"
            )

            # Добавляем метрики в саммаризацию
            legacy_summary["quality_metrics"] = {
                "score": metrics.score,
                "has_context": metrics.has_context,
                "context_length": metrics.context_length,
                "discussion_count": metrics.discussion_count,
                "decisions_count": metrics.decisions_count,
                "risks_count": metrics.risks_count,
                "links_count": metrics.links_count,
                "language_matches": metrics.language_matches,
                "issues": metrics.issues,
            }

            # Если качество неприемлемо, зафиксируем это — структурные проходы выполнятся позже
            if not self.quality_evaluator.is_acceptable(metrics):
                logger.warning(
                    f"Качество саммаризации {session_id} ниже порога ({metrics.score:.1f}). Будет применён структурный IterativeRefiner."
                )
            else:
                logger.info("Качество саммаризации приемлемо")

        canonical_summary, aux_data = self._build_canonical_summary(
            session=session,
            messages=messages,
            legacy_summary=legacy_summary,
            quality_metrics=metrics,
            chat_mode=chat_mode,
            profile=self._map_profile(chat_mode, messages),
            entities=entities,
        )

        canonical_summary["raw_summary"] = summary_text
        canonical_summary["fallback_used"] = fallback_used
        canonical_summary["_legacy"] = legacy_summary

        if (
            self.enable_quality_check
            and self.enable_iterative_refinement
            and canonical_summary.get("quality", {}).get("score", 0.0)
            < self.iterative_refiner.target_score
        ):
            canonical_summary = await self.iterative_refiner.refine(
                canonical_summary,
                aux_data,
                session,
            )

        return canonical_summary

    def _map_profile(self, chat_mode: str, messages: List[Dict[str, Any]]) -> str:
        unique_authors = set()
        for msg in messages:
            author = msg.get("from") or {}
            username = (
                author.get("username") or author.get("display") or author.get("id")
            )
            if username:
                unique_authors.add(str(username))
        if len(unique_authors) <= 1 or chat_mode == "channel":
            return "broadcast"
        return "group-project"

    def _message_key(
        self, msg: Dict[str, Any], fallback_index: Optional[int] = None
    ) -> str:
        """Генерирует стабильный ключ сообщения для привязки служебных данных."""

        candidate = msg.get("id")
        if candidate is None:
            candidate = msg.get("message_id") or msg.get("msg_id")
        if candidate is None and fallback_index is not None:
            candidate = f"auto_{fallback_index:05d}"
        if candidate is None:
            text = self._extract_message_text(msg)
            candidate = f"auto_{abs(hash((msg.get('date_utc'), text))) % 10**8:08d}"
        return str(candidate)

    def _extract_message_text(self, msg: Dict[str, Any]) -> str:
        """Возвращает текст сообщения с учётом форматированных структур Telegram."""

        text = msg.get("text", "")
        if isinstance(text, list):
            parts: List[str] = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            text = "".join(parts)
        return text if isinstance(text, str) else str(text)

    def _detect_domain_addons(
        self, keyed_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Определяет доменные аддоны и дополнительные метки для сообщений."""

        addons: set[str] = set()
        asset_tags: set[str] = set()
        geo_tags: set[str] = set()
        per_message: Dict[str, Dict[str, Any]] = {}

        for item in keyed_messages:
            key = item["key"]
            text = item["text"]
            lowered = text.lower()
            uppered = text.upper()

            entry: Dict[str, Any] = {}

            # --- Crypto ---
            ticker_hits = {
                ticker
                for ticker in CRYPTO_TICKERS
                if re.search(rf"\b{re.escape(ticker)}\b", uppered)
            }
            exchange_hits = {
                exch.upper() for exch in CRYPTO_EXCHANGES if exch in lowered
            }
            keyword_hit = any(term in lowered for term in CRYPTO_TERMS)

            if ticker_hits or exchange_hits or keyword_hit:
                addons.add("crypto")
                combined_tags = sorted(ticker_hits | exchange_hits)
                if combined_tags:
                    entry["asset_tags"] = combined_tags
                    asset_tags.update(combined_tags)

            # --- Sci-tech ---
            sci_term_hit = any(term in lowered for term in SCI_TECH_TERMS)
            sci_pattern_hit = any(pattern.search(text) for pattern in SCI_TECH_PATTERNS)
            if sci_term_hit or sci_pattern_hit:
                addons.add("sci-tech")
                entry["sci_markers"] = True

            # --- Geopolitics ---
            geo_hits = {
                label for pattern, label in GEOPOLITICS_PATTERNS if pattern.search(text)
            }
            if geo_hits:
                addons.add("geopolitics")
                geo_sorted = sorted(geo_hits)
                entry["geo_entities"] = geo_sorted
                geo_tags.update(geo_sorted)

            if entry:
                per_message[key] = entry

        return {
            "addons": addons,
            "asset_tags": sorted(asset_tags),
            "geo_entities": sorted(geo_tags),
            "by_key": per_message,
        }

    def _prepare_conversation_text(
        self,
        messages: List[Dict[str, Any]],
        max_messages: int = 50,
        max_chars: int = 16000,
    ) -> str:
        """
        Подготовка текста разговора для саммаризации

        Args:
            messages: Список сообщений
            max_messages: Максимальное количество сообщений для включения

        Returns:
            Форматированный текст разговора
        """
        # Берём первые и последние сообщения, если слишком много
        selected_messages = messages
        if len(messages) > max_messages:
            # Берём первые 40% и последние 60%
            first_count = int(max_messages * 0.4)
            last_count = max_messages - first_count
            selected_messages = messages[:first_count] + messages[-last_count:]

        text_parts = []
        for _i, msg in enumerate(selected_messages):
            # Извлекаем информацию о сообщении
            text = msg.get("text", "").strip()
            if not text:
                continue

            # Извлекаем автора
            from_user = msg.get("from", {})
            if isinstance(from_user, dict):
                author = from_user.get("username") or from_user.get(
                    "display", "Unknown"
                )
            else:
                author = str(from_user) if from_user else "Unknown"

            # Извлекаем время
            date_str = msg.get("date_utc") or msg.get("date", "")
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                if date_str:
                    from ..utils.datetime_utils import parse_datetime_utc

                    dt = parse_datetime_utc(date_str, use_zoneinfo=True)
                    if dt:
                        time_str = dt.astimezone(ZoneInfo("Asia/Bangkok")).strftime("%H:%M")
                    else:
                        time_str = "??:??"
                else:
                    time_str = "??:??"
            except Exception:
                time_str = "??:??"

            # Ограничиваем длину текста
            if len(text) > 300:
                text = text[:300] + "..."

            # Добавляем информацию о дублях, если есть
            duplicate_info = ""
            if msg.get("_duplicate_marker"):
                dup_count = msg.get("_duplicate_count", 0)
                variants = msg.get("_duplicate_variants", [])

                if dup_count > 0:
                    duplicate_info = f" [повторено {dup_count + 1}x"
                    # Показываем вариации если они есть и отличаются
                    if variants and len(variants) > 0:
                        # Проверяем, есть ли значимые различия
                        base_normalized = self._normalize_text_for_display(text)
                        has_variations = any(
                            self._normalize_text_for_display(v) != base_normalized
                            for v in variants
                        )
                        if has_variations:
                            duplicate_info += ", есть вариации"
                    duplicate_info += "]"

            text_parts.append(f"[{time_str}] {author}: {text}{duplicate_info}")

        result = "\n".join(text_parts)

        # Ограничиваем общую длину текста для контроля размера промпта
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (текст обрезан)"

        return result

    def _normalize_text_for_display(self, text: str) -> str:
        """
        Нормализация текста для проверки вариаций

        Args:
            text: Исходный текст

        Returns:
            Нормализованный текст
        """
        import re

        if not text:
            return ""
        # Убираем пунктуацию и пробелы, приводим к нижнему регистру
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _create_summarization_prompt(
        self,
        conversation_text: str,
        chat: str,
        language: str,
        session: Dict[str, Any],
        chat_mode: str,
        previous_context: Dict[str, Any],
        extended_context: Optional[Dict[str, Any]] = None,
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
                    len(recent_context) > 8000
                ):  # Увеличиваем до 8000 символов (~2000 токенов)
                    recent_context = recent_context[:8000] + "..."
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

            if previous_context["open_risks"]:
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
            if len(chat_context) > 4000:  # Увеличиваем до 4000 символов (~1000 токенов)
                chat_context = chat_context[:4000] + "..."
            chat_context_section = f"""
## Образ чата
{chat_context}

"""

        # Добавляем расширенный контекст для малых сессий
        extended_context_section = ""
        if extended_context and extended_context.get("previous_messages_count", 0) > 0:
            extended_context_text = (
                self.incremental_context_manager.format_context_for_prompt(
                    extended_context, max_context_length=8000
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

        custom_instruction = self.instruction_manager.get_instruction(chat, chat_mode)
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
                f"OllamaClient автоматически разобьет его на части при необходимости."
            )

        return prompt

    def _parse_summary_structure(self, summary_text: str) -> Dict[str, Any]:
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

    def _ensure_summary_completeness(
        self, messages: List[Dict[str, Any]], chat: str, structure: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Проверяет и при необходимости дополняет саммаризацию эвристическими данными.

        Args:
            messages: Исходные сообщения сессии
            chat: Название чата
            structure: Структура, полученная от LLM

        Returns:
            (дополненная структура, признак использования fallback)
        """
        # Безопасное извлечение context (может быть строкой или списком)
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

        # Определяем тип чата
        chat_mode = self._detect_chat_mode(messages)

        needs_context = len(context_text) < 40
        needs_discussion = len(discussion) < 2
        needs_decisions = len(decisions) == 0
        needs_risks = len(risks) == 0

        # Для каналов всегда заполняем ключевые тезисы и важные моменты
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

        fallback = self._build_fallback_structure(messages, chat, chat_mode)
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

    def _build_fallback_structure(
        self, messages: List[Dict[str, Any]], chat: str, chat_mode: str
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

        participants = self._collect_participants(valid_messages)
        first_text = self._truncate_text(valid_messages[0].get("text", ""), 220)
        last_text = self._truncate_text(valid_messages[-1].get("text", ""), 220)
        context_lines = [
            f'Автосаммаризация (fallback) по чату "{chat}". Сообщений: {len(valid_messages)}.',
        ]
        if participants:
            context_lines.append(f"Активные участники: {', '.join(participants)}")
        if first_text:
            context_lines.append(f"Старт обсуждения: {first_text}")
        if last_text and last_text != first_text:
            context_lines.append(f"Финал обсуждения: {last_text}")

        # Для канала больше упор на ключевые публикации, для группы — на реплики
        discussion_limit = 5 if chat_mode == "channel" else 4
        discussion_msgs = self._select_key_messages(
            valid_messages, limit=discussion_limit
        )
        if len(discussion_msgs) < 2:
            discussion_msgs = valid_messages[: min(2, len(valid_messages))]
        discussion_lines = [
            self._format_message_bullet(msg, prefix="- ") for msg in discussion_msgs
        ]

        decision_msgs = self._select_messages_with_keywords(
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
            # В каналах решения обычно неуместны — делаем пустой список или заметку
            decision_lines = (
                []
                if decision_msgs == []
                else [
                    self._format_message_bullet(m, prefix="- ") for m in decision_msgs
                ]
            )
        elif decision_msgs:
            decision_lines = [
                self._format_message_bullet(msg, prefix="- [ ] ")
                for msg in decision_msgs
            ]
        else:
            decision_lines = [
                "- [ ] Автопроверка: явных действий не зафиксировано; требуется ручная проверка."
            ]

        risk_msgs = self._select_messages_with_keywords(
            valid_messages,
            keywords=["риск", "проблем", "опас", "сомн", "issue", "блок", "concern"],
            limit=3,
        )
        if risk_msgs:
            risk_lines = [
                self._format_message_bullet(msg, prefix="- ") for msg in risk_msgs
            ]
        else:
            risk_lines = [
                "- Автопроверка: явных рисков в сообщениях не найдено; проверить вручную."
            ]

        # Для каналов добавляем ключевые тезисы и важные моменты
        if chat_mode == "channel":
            # Извлекаем ключевые тезисы из первых сообщений
            key_points = []
            for msg in valid_messages[:3]:
                text = self._truncate_text(msg.get("text", ""), 100)
                if text:
                    key_points.append(f"- {text}")

            # Извлекаем важные моменты (ссылки, даты, имена, ключевые слова)
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

                # Проверяем ключевые слова
                if any(keyword in text_lower for keyword in important_keywords):
                    important_items.append(f"- {self._truncate_text(text, 80)}")
                # Также добавляем сообщения с датами и временем (часто важные объявления)
                elif any(
                    pattern in text
                    for pattern in ["UTC", "GMT", "at ", "on ", "2024", "2025"]
                ):
                    important_items.append(f"- {self._truncate_text(text, 80)}")

                if len(important_items) >= 3:
                    break

            # Если ничего не найдено, берем первые сообщения как важные
            if not important_items:
                for msg in valid_messages[:2]:
                    text = self._truncate_text(msg.get("text", ""), 80)
                    if text:
                        important_items.append(f"- {text}")

            # Если все еще пусто, добавляем заметку
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

    def _detect_chat_mode(self, messages: List[Dict[str, Any]]) -> str:
        """
        Определяет тип чата: 'channel' или 'group'.

        Логика:
        - Если у большинства сообщений нет поля 'from' или оно None → канал
        - Если есть много разных авторов → группа
        - Если один автор доминирует → канал
        """
        if not messages:
            return "group"

        # Подсчитываем сообщения без автора (характерно для каналов)
        messages_without_author = 0
        authors = []

        for m in messages:
            fr = m.get("from")
            if fr is None or not fr:
                messages_without_author += 1
            else:
                name = (
                    fr.get("username") or fr.get("display") or fr.get("id") or "unknown"
                )
                authors.append(str(name))

        total_messages = len(messages)
        messages_with_author = total_messages - messages_without_author

        # Если больше 70% сообщений без автора → канал
        if messages_without_author / total_messages > 0.7:
            return "channel"

        # Если сообщений с авторами мало, но они есть
        if messages_with_author < 5:
            return "group"

        # Анализируем авторов
        total = len([a for a in authors if a != "unknown"])
        if total == 0:
            return "group"

        cnt = Counter(a for a in authors if a != "unknown")
        top, top_count = cnt.most_common(1)[0]
        top_share = top_count / total
        unique = len(cnt)

        # Канал, если автор один/почти один, и сообщений достаточно
        if (top_share >= 0.85 and unique <= 3 and total >= 5) or unique == 1:
            return "channel"

        return "group"

    def _collect_participants(self, messages: List[Dict[str, Any]]) -> List[str]:
        participants = Counter()
        for msg in messages:
            author = msg.get("from") or {}
            display = author.get("display") or author.get("username")
            if not display:
                user_id = author.get("id") if isinstance(author, dict) else None
                display = f"user_{user_id}" if user_id else "unknown"
            participants[display] += 1
        top = [name for name, _ in participants.most_common(5)]
        return top

    def _select_key_messages(
        self, messages: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for msg in messages:
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            score = len(text)
            for reaction in msg.get("reactions", []) or []:
                score += reaction.get("count", 0) * 5
            scored.append((score, msg))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_msgs = [msg for _, msg in scored[:limit]]
        top_msgs.sort(key=lambda m: (m.get("date_utc") or m.get("date") or ""))
        return top_msgs

    def _select_messages_with_keywords(
        self, messages: List[Dict[str, Any]], keywords: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        matched = []
        keywords_lower = [kw.lower() for kw in keywords]

        for msg in messages:
            text = (msg.get("text") or "").lower()
            if not text:
                continue
            if any(kw in text for kw in keywords_lower):
                matched.append(msg)

        matched.sort(key=lambda m: (m.get("date_utc") or m.get("date") or ""))
        return matched[:limit]

    def _format_message_bullet(self, msg: Dict[str, Any], prefix: str = "- ") -> str:
        time_str = self._format_message_time(msg)
        author = self._format_author_name(msg)
        text = self._truncate_text((msg.get("text") or "").replace("\n", " "), 220)
        return f"{prefix}[{time_str}] {author}: {text}"

    def _format_author_name(self, msg: Dict[str, Any]) -> str:
        author = msg.get("from") or {}
        return author.get("display") or author.get("username") or "автор"

    def _format_message_time(self, msg: Dict[str, Any]) -> str:
        """Форматирует время сообщения для отображения."""
        from ..utils.datetime_utils import format_datetime_display

        date = msg.get("date_utc") or msg.get("date")
        return format_datetime_display(date, format_type="time", fallback="??:??")

    def _truncate_text(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text.strip()
        return text[: max_len - 3].rstrip() + "..."

    async def _extract_action_items(
        self, messages: List[Dict[str, Any]], summary_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Извлечение Action Items из сессии

        Args:
            messages: Список сообщений
            summary_structure: Структура саммаризации

        Returns:
            Список Action Items с confidence >= 0.7
        """
        action_items = []

        # Берём решения из саммаризации
        decisions = summary_structure.get("decisions", [])

        for decision in decisions:
            # Парсим структуру решения
            item = self._parse_action_item(decision)
            if item and item.get("confidence", 0) >= 0.7:
                action_items.append(item)

        # Также ищем маркеры действий в последних сообщениях
        last_messages = messages[-10:] if len(messages) > 10 else messages
        for msg in last_messages:
            text = msg.get("text", "").lower()
            if any(
                marker in text
                for marker in [
                    "решили",
                    "next",
                    "todo",
                    "нужно",
                    "надо",
                    "следующий шаг",
                ]
            ):
                # Извлекаем как потенциальное действие
                item = {
                    "text": msg.get("text", ""),
                    "confidence": 0.75,
                    "owner": None,
                    "due": None,
                    "priority": "P2",
                }
                action_items.append(item)

        return action_items[:5]  # Максимум 5 действий

    def _parse_action_item(self, decision_text: str) -> Optional[Dict[str, Any]]:
        """
        Парсинг Action Item из текста решения

        Args:
            decision_text: Текст решения

        Returns:
            Словарь с данными Action Item или None
        """
        import re

        # Извлекаем компоненты
        item = {
            "text": decision_text,
            "confidence": 0.8,  # По умолчанию высокая уверенность из саммаризации
            "owner": None,
            "due": None,
            "priority": "P2",
        }

        # Ищем владельца (owner)
        owner_match = re.search(r"owner:\s*@?(\w+)", decision_text, re.IGNORECASE)
        if owner_match:
            item["owner"] = "@" + owner_match.group(1)

        # Ищем срок (due)
        due_match = re.search(
            r"due:\s*([0-9\-:T ]+(?:BKK|UTC)?)", decision_text, re.IGNORECASE
        )
        if due_match:
            item["due"] = due_match.group(1).strip()

        # Ищем приоритет
        pri_match = re.search(r"pri:\s*(P[123])", decision_text, re.IGNORECASE)
        if pri_match:
            item["priority"] = pri_match.group(1).upper()

        return item

    def _build_canonical_summary(
        self,
        session: Dict[str, Any],
        messages: List[Dict[str, Any]],
        legacy_summary: Dict[str, Any],
        quality_metrics,
        chat_mode: str,
        profile: str,
        entities: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Формирует каноническую структуру саммаризации (v1)."""

        keyed_messages = []
        for idx, msg in enumerate(messages):
            text_value = self._extract_message_text(msg)
            if not text_value or not text_value.strip():
                continue
            key = self._message_key(msg, fallback_index=idx)
            keyed_messages.append(
                {"index": idx, "key": key, "message": msg, "text": text_value}
            )

        valid_messages = [item["message"] for item in keyed_messages]
        message_map = [
            self._build_message_envelope(
                item["message"], fallback_index=item["index"], msg_key=item["key"]
            )
            for item in keyed_messages
        ]

        chat_name = legacy_summary.get("chat", session.get("chat"))
        chat_id = session.get("chat_id") or slugify(chat_name)
        time_span = legacy_summary.get("time_range_bkk") or self._derive_time_span(
            messages
        )
        messages_total = len(messages)
        participants = legacy_summary.get("participants", [])
        dominant_language = legacy_summary.get("dominant_language") or session.get(
            "dominant_language"
        )

        addons_info = self._detect_domain_addons(keyed_messages)

        topics = self._generate_topics(profile, legacy_summary, message_map, time_span)
        claims = self._generate_claims(
            profile, topics, message_map, entities, addons_info
        )
        discussion = self._generate_discussion(profile, message_map)
        actions = self._generate_actions(
            topics, legacy_summary.get("decisions_next", []), valid_messages
        )
        risks = self._generate_risks(
            topics, legacy_summary.get("risks_open", []), valid_messages
        )
        uncertainties = []
        entities_flat = self._flatten_entities(entities)
        attachments = self._build_attachments(legacy_summary.get("links_artifacts", []))
        rationale = self._derive_rationale(profile, actions, risks)

        aux_data: Dict[str, Any] = {
            "profile": profile,
            "messages_total": messages_total,
            "messages": messages,
            "message_map": message_map,
            "addons_info": addons_info,
            "legacy_summary": legacy_summary,
            "legacy_metrics": quality_metrics,
            "legacy_decisions": legacy_summary.get("decisions_next", []),
            "legacy_risks": legacy_summary.get("risks_open", []),
            "entities_flat": entities_flat,
            "time_span_default": time_span,
        }

        message_index_map: Dict[str, int] = {}
        for idx, item in enumerate(message_map):
            identifiers = []
            if item.get("id") is not None:
                identifiers.append(str(item.get("id")))
            if item.get("key") is not None:
                identifiers.append(str(item.get("key")))
            for identifier in identifiers:
                if identifier:
                    message_index_map.setdefault(identifier, idx)
        aux_data["message_index_map"] = message_index_map

        small_session_info = None
        if self._is_small_session(messages_total, topics):
            (
                topics,
                claims,
                discussion,
                actions,
                risks,
                rationale,
                small_session_info,
            ) = self._apply_small_session_policy(
                topics=topics,
                claims=claims,
                discussion=discussion,
                actions=actions,
                risks=risks,
                rationale=rationale,
                message_map=message_map,
                legacy_summary=legacy_summary,
                fallback_span=time_span,
            )
        if small_session_info:
            aux_data["small_session_info"] = small_session_info

        meta = {
            "chat_name": chat_name,
            "profile": profile,
            "time_span": time_span,
            "messages_total": messages_total,
            "participants": participants,
            "dominant_language": dominant_language,
            "chat_mode": chat_mode,
            "start_time_utc": session.get("start_time_utc", ""),
            "end_time_utc": session.get("end_time_utc", ""),
            "addons": sorted(addons_info.get("addons", [])),
        }
        if small_session_info:
            meta["policy_flags"] = small_session_info.get("policy_flags", [])

        summary = {
            "version": SESSION_SUMMARY_VERSION,
            "chat_id": chat_id,
            "session_id": legacy_summary["session_id"],
            "meta": meta,
            "topics": topics,
            "claims": claims,
            "discussion": discussion,
            "actions": actions,
            "risks": risks,
            "uncertainties": uncertainties,
            "entities": entities_flat,
            "attachments": attachments,
            "rationale": rationale,
        }

        quality_context = self._refresh_quality(summary, aux_data)
        aux_data.update(quality_context)

        return summary, aux_data

    def _refresh_quality(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Пересчитывает метрики и качество для канонической саммаризации."""

        topics = summary.get("topics", [])
        claims = summary.get("claims", [])
        actions = summary.get("actions", [])
        risks = summary.get("risks", [])
        discussion = summary.get("discussion", [])
        messages = aux_data.get("messages", [])
        messages_total = aux_data.get("messages_total", len(messages))
        message_map = aux_data.get("message_map", [])
        profile = summary.get("meta", {}).get("profile") or aux_data.get("profile")
        legacy_summary = aux_data.get("legacy_summary") or {}
        legacy_metrics = aux_data.get("legacy_metrics")
        small_session_info = aux_data.get("small_session_info")

        coverage, claims_coverage = self._calculate_coverage(
            topics, claims, messages_total
        )
        thread_count = self._count_threads(messages)
        schema_eval = self._evaluate_schema_requirements(
            profile=profile,
            topics=topics,
            claims=claims,
            actions=actions,
            risks=risks,
            discussion=discussion,
            messages=messages,
            thread_count=thread_count,
        )
        schema_eval["claims_coverage_ok"] = (
            claims_coverage >= schema_eval["claims_coverage_target"]
        )
        if (
            not schema_eval["claims_coverage_ok"]
            and "claims_coverage_below_threshold" not in schema_eval["issues"]
        ):
            schema_eval["issues"].append("claims_coverage_below_threshold")

        dup_rate = self._estimate_duplicate_rate(message_map)
        quality_score, len_penalty = self._compute_quality_score(
            profile=profile,
            coverage=coverage,
            claims_coverage=claims_coverage,
            schema_eval=schema_eval,
            dup_rate=dup_rate,
            claims=claims,
        )
        if small_session_info:
            quality_score = min(
                quality_score, small_session_info.get("score_cap", quality_score)
            )

        confidence = round(quality_score / 100.0, 3)
        quality_status = self._derive_quality_status(
            score=quality_score,
            schema_eval=schema_eval,
            legacy_summary=legacy_summary,
            legacy_metrics=legacy_metrics,
        )
        if small_session_info:
            quality_status = "needs_review"

        meta = summary.setdefault("meta", {})
        meta["confidence"] = confidence
        addons_info = aux_data.get("addons_info", {})
        meta["addons"] = sorted(addons_info.get("addons", []))
        if small_session_info:
            meta.setdefault("policy_flags", small_session_info.get("policy_flags", []))

        existing_details = summary.get("quality", {}).get("details", {})
        legacy_metrics_details = legacy_summary.get("quality_metrics", {})
        quality_details = {
            "legacy_metrics": legacy_metrics_details,
            "schema_issues": schema_eval["issues"],
            "blocking_issues": schema_eval["blocking_issues"],
            "dup_rate": dup_rate,
            "len_penalty": len_penalty,
            "claims_threshold": {
                "claims_coverage_target": schema_eval["claims_coverage_target"],
                "claims_coverage_ok": schema_eval["claims_coverage_ok"],
            },
        }
        if legacy_metrics:
            quality_details["legacy_score"] = legacy_metrics.score
        if small_session_info:
            quality_details["policy_flags"] = small_session_info.get("policy_flags", [])
        if "refinement_history" in existing_details:
            quality_details["refinement_history"] = existing_details[
                "refinement_history"
            ]

        quality = {
            "score": round(quality_score, 2),
            "status": quality_status,
            "kpi": {
                "coverage": coverage,
                "claims_coverage": claims_coverage,
                "topics": len(topics),
                "actions": len(actions),
                "risks": len(risks),
                "threads": thread_count,
            },
            "flags": {
                "topics_ok": schema_eval["topics_ok"],
                "claims_ok": schema_eval["claims_ok"],
                "actions_ok": schema_eval["actions_ok"],
                "risks_ok": schema_eval["risks_ok"],
                "threads_ok": schema_eval["threads_ok"],
                "structure_ok": schema_eval["structure_ok"],
                "claims_coverage_ok": schema_eval["claims_coverage_ok"],
                "discussion_ok": schema_eval["discussion_ok"],
            },
            "details": quality_details,
        }

        summary["quality"] = quality

        return {
            "thread_count": thread_count,
            "schema_eval": schema_eval,
            "coverage": coverage,
            "claims_coverage": claims_coverage,
        }

    def _run_structural_pass(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        session: Dict[str, Any],
        iteration: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        profile = summary.get("meta", {}).get("profile") or aux_data.get("profile")
        changed = False

        if iteration == 1:
            changed = self._refine_pass_expand_evidence(summary, aux_data, profile)
        elif iteration == 2:
            changed = self._refine_pass_claims(summary, aux_data, profile)
        elif iteration == 3:
            changed = self._refine_pass_topics(summary, aux_data)
        elif iteration == 4:
            changed = self._refine_pass_profile_rules(summary, aux_data, profile)
        elif iteration == 5:
            changed = self._refine_pass_threads(summary, aux_data, profile)
        elif iteration == 6:
            # Дополнительная итерация: улучшение структуры дискуссии
            changed = self._refine_pass_discussion_structure(summary, aux_data, profile)
        elif iteration == 7:
            # Дополнительная итерация: улучшение качества контекста
            changed = self._refine_pass_context_quality(summary, aux_data, profile)

        previous_score = summary.get("quality", {}).get("score", 0.0)
        quality_context = self._refresh_quality(summary, aux_data)
        aux_data.update(quality_context)
        if summary.get("quality", {}).get("score", 0.0) != previous_score:
            changed = True

        return summary, {"changed": changed}

    def _refine_pass_expand_evidence(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        message_map = aux_data.get("message_map", [])
        if not message_map:
            return False

        per_topic_limit = 60 if profile == "group-project" else 40
        fallback_span = aux_data.get(
            "time_span_default", summary.get("meta", {}).get("time_span", "")
        )

        changed = False
        for topic in summary.get("topics", []):
            current_ids = [mid for mid in topic.get("message_ids", []) if mid]
            if not current_ids:
                continue
            target = min(per_topic_limit, len(message_map))
            desired = max(len(current_ids) * 2, min(len(current_ids) + 2, target))
            desired = min(desired, target)
            expanded_ids = self._expand_message_ids(current_ids, aux_data, desired)
            if expanded_ids != current_ids:
                topic["message_ids"] = expanded_ids
                segment = self._collect_segment_by_ids(expanded_ids, aux_data)
                topic["time_span"] = self._topic_time_span(segment, fallback_span)
                changed = True

        return changed

    def _refine_pass_claims(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        message_map = aux_data.get("message_map", [])
        if not message_map:
            return False

        claims = summary.get("claims", [])
        existing_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for claim in claims:
            existing_by_topic.setdefault(claim.get("topic_title"), []).append(claim)

        modality = "media" if profile == "broadcast" else "internal"
        source = modality
        entities_pool = aux_data.get("entities_flat") or summary.get("entities", [])
        addons_info = aux_data.get("addons_info", {})
        global_asset_tags = addons_info.get("asset_tags", [])
        global_geo_tags = addons_info.get("geo_entities", [])
        addons_set = addons_info.get("addons", set())
        per_message_addons = addons_info.get("by_key", {})

        additions = 0
        for topic in summary.get("topics", []):
            title = topic.get("title")
            if not title:
                continue
            if existing_by_topic.get(title):
                continue

            for message_id in topic.get("message_ids", []):
                envelope = self._lookup_message_envelope(message_id, aux_data)
                if not envelope or not envelope.get("text"):
                    continue
                summary_text = self._build_claim_summary(
                    envelope.get("text", ""), topic.get("summary", "")
                )
                summary_text = summary_text[:160]
                claim = {
                    "ts": envelope.get("ts_bkk", ""),
                    "source": source,
                    "modality": modality,
                    "credibility": "medium",
                    "entities": list(entities_pool[:4]),
                    "summary": summary_text,
                    "msg_id": self._message_identifier(envelope) or "",
                    "topic_title": title,
                }
                self._apply_addon_metadata_to_claim(
                    claim,
                    per_message_addons.get(envelope.get("key")),
                    addons_set,
                    global_asset_tags,
                    global_geo_tags,
                )
                claims.append(claim)
                additions += 1
                break

        if not additions:
            return False

        credibility_order = {"high": 0, "medium": 1, "low": 2}
        claims.sort(
            key=lambda item: (
                credibility_order.get(item.get("credibility", "medium"), 1),
                item.get("ts", ""),
            )
        )
        summary["claims"] = claims[:20]
        return True

    def _refine_pass_topics(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
    ) -> bool:
        topics = summary.get("topics", [])
        if len(topics) <= 1:
            return False

        changed = False
        fallback_span = aux_data.get(
            "time_span_default",
            summary.get("meta", {}).get("time_span", ""),
        )

        i = 0
        while i < len(topics):
            if len(topics) <= 1:
                break
            topic = topics[i]
            message_ids = topic.get("message_ids", [])
            if len(message_ids) >= 3:
                i += 1
                continue
            target_idx = i - 1 if i > 0 else i + 1
            if target_idx < 0 or target_idx >= len(topics):
                i += 1
                continue
            receiver = topics[target_idx]
            combined_ids = list(
                dict.fromkeys(receiver.get("message_ids", []) + message_ids)
            )
            receiver["message_ids"] = combined_ids
            combined_summary = (
                f"{receiver.get('summary', '')} {topic.get('summary', '')}".strip()
            )
            receiver["summary"] = self._normalize_summary(combined_summary)
            receiver["title"] = self._build_topic_title(receiver["summary"])
            segment = self._collect_segment_by_ids(combined_ids, aux_data)
            receiver["time_span"] = self._topic_time_span(segment, fallback_span)
            topics.pop(i)
            changed = True
            continue
        while len(topics) > 6:
            smallest_idx = min(
                range(len(topics)),
                key=lambda idx: len(topics[idx].get("message_ids", [])),
            )
            receiver_idx = smallest_idx - 1 if smallest_idx > 0 else smallest_idx + 1
            if receiver_idx < 0 or receiver_idx >= len(topics):
                break

            if smallest_idx < receiver_idx:
                donor = topics.pop(smallest_idx)
                receiver_idx -= 1
                receiver = topics[receiver_idx]
            else:
                receiver = topics[receiver_idx]
                donor = topics.pop(smallest_idx)

            combined_ids = list(
                dict.fromkeys(
                    receiver.get("message_ids", []) + donor.get("message_ids", [])
                )
            )
            receiver["message_ids"] = combined_ids
            combined_summary = (
                f"{receiver.get('summary', '')} {donor.get('summary', '')}".strip()
            )
            receiver["summary"] = self._normalize_summary(combined_summary)
            receiver["title"] = self._build_topic_title(receiver["summary"])
            segment = self._collect_segment_by_ids(combined_ids, aux_data)
            receiver["time_span"] = self._topic_time_span(segment, fallback_span)
            changed = True

        return changed

    def _refine_pass_profile_rules(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        changed = False

        if profile == "broadcast":
            claims_cov = aux_data.get("claims_coverage", 0.0)
            if claims_cov < 0.25:
                changed |= self._refine_pass_claims(summary, aux_data, profile)
            summary["rationale"] = self._derive_rationale(
                profile, summary.get("actions", []), summary.get("risks", [])
            )
            return changed

        # group-project specific adjustments
        actions = summary.setdefault("actions", [])
        if len(actions) < 2:
            for decision in aux_data.get("legacy_decisions", []):
                action = self._create_action_from_decision(decision, summary, aux_data)
                if action:
                    actions.append(action)
                    changed = True
                if len(actions) >= 2:
                    break

        risks = summary.setdefault("risks", [])
        if len(risks) < 1:
            for risk_entry in aux_data.get("legacy_risks", []):
                risk = self._create_risk_entry(risk_entry, summary, aux_data)
                if risk:
                    risks.append(risk)
                    changed = True
                    break

        summary["rationale"] = self._derive_rationale(profile, actions, risks)
        return changed

    def _refine_pass_threads(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        messages = aux_data.get("messages", [])
        if not any(msg.get("reply_to") for msg in messages):
            return False

        discussion = summary.setdefault("discussion", [])
        existing_ids = {item.get("msg_id") for item in discussion if item.get("msg_id")}

        for envelope in aux_data.get("message_map", []):
            raw = envelope.get("raw", {})
            if not raw.get("reply_to"):
                continue
            identifier = self._message_identifier(envelope)
            if identifier in existing_ids:
                continue
            quote = self._truncate_text(envelope.get("text", ""), 220)
            if not quote:
                continue
            discussion.append(
                {
                    "ts": envelope.get("ts_bkk", ""),
                    "author": envelope.get("author", ""),
                    "msg_id": identifier or "",
                    "quote": quote,
                }
            )
            return True

        return False

    def _refine_pass_discussion_structure(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        """Дополнительная итерация: улучшение структуры дискуссии"""
        discussion = summary.get("discussion", [])
        if not discussion:
            return False

        changed = False

        # Улучшаем качество цитат - делаем их более информативными
        for item in discussion:
            quote = item.get("quote", "")
            if len(quote) < 50:  # Слишком короткие цитаты
                # Попробуем найти более длинную цитату из того же сообщения
                msg_id = item.get("msg_id", "")
                if msg_id:
                    message_map = aux_data.get("message_map", [])
                    for envelope in message_map:
                        if self._message_identifier(envelope) == msg_id:
                            full_text = envelope.get("text", "")
                            if len(full_text) > len(quote) and len(full_text) <= 300:
                                item["quote"] = self._truncate_text(full_text, 250)
                                changed = True
                            break

        # Добавляем недостающие элементы дискуссии если их мало
        if len(discussion) < 3 and profile == "group-project":
            messages = aux_data.get("messages", [])
            existing_ids = {
                item.get("msg_id") for item in discussion if item.get("msg_id")
            }

            # Ищем дополнительные сообщения для дискуссии
            for msg in messages[-10:]:  # Последние 10 сообщений
                msg_id = self._message_key(msg)
                if msg_id not in existing_ids and len(msg.get("text", "")) > 30:
                    quote = self._truncate_text(msg.get("text", ""), 200)
                    if quote:
                        discussion.append(
                            {
                                "ts": msg.get("date_utc", ""),
                                "author": msg.get("from", {}).get("username", ""),
                                "msg_id": msg_id,
                                "quote": quote,
                            }
                        )
                        changed = True
                        if len(discussion) >= 5:  # Ограничиваем количество
                            break

        return changed

    def _refine_pass_context_quality(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        profile: str,
    ) -> bool:
        """Дополнительная итерация: улучшение качества контекста"""
        context = summary.get("context", "")
        if len(context) < 100:  # Слишком короткий контекст
            # Попробуем создать более информативный контекст
            messages = aux_data.get("messages", [])
            if messages:
                # Берем первые несколько сообщений для контекста
                context_parts = []
                for msg in messages[:3]:
                    text = msg.get("text", "")
                    if text and len(text) > 20:
                        context_parts.append(text[:100])

                if context_parts:
                    new_context = " ".join(context_parts)[:300]
                    if len(new_context) > len(context):
                        summary["context"] = new_context
                        return True

        # Улучшаем качество топиков
        topics = summary.get("topics", [])
        changed = False

        for topic in topics:
            title = topic.get("title", "")
            if len(title) < 10:  # Слишком короткий заголовок
                # Попробуем создать более описательный заголовок
                message_ids = topic.get("message_ids", [])
                if message_ids:
                    message_map = aux_data.get("message_map", [])
                    topic_texts = []
                    for envelope in message_map:
                        if self._message_identifier(envelope) in message_ids:
                            text = envelope.get("text", "")
                            if text:
                                topic_texts.append(text[:50])

                    if topic_texts:
                        # Создаем более информативный заголовок
                        combined_text = " ".join(topic_texts[:3])
                        if len(combined_text) > len(title):
                            topic["title"] = self._truncate_text(combined_text, 80)
                            changed = True

        return changed

    def _expand_message_ids(
        self,
        seed_ids: List[str],
        aux_data: Dict[str, Any],
        target_count: int,
    ) -> List[str]:
        message_map = aux_data.get("message_map", [])
        index_map = aux_data.get("message_index_map", {})
        if not message_map:
            return seed_ids

        indices = [
            index_map.get(mid) for mid in seed_ids if index_map.get(mid) is not None
        ]
        if not indices:
            return seed_ids

        left = min(indices)
        right = max(indices)
        while (right - left + 1) < target_count and (
            left > 0 or right < len(message_map) - 1
        ):
            if left > 0:
                left -= 1
            if (right - left + 1) >= target_count:
                break
            if right < len(message_map) - 1:
                right += 1

        expanded: List[str] = []
        for idx in range(left, right + 1):
            identifier = self._message_identifier(message_map[idx])
            if identifier:
                expanded.append(identifier)
            if len(expanded) >= target_count:
                break

        return expanded

    def _collect_segment_by_ids(
        self, message_ids: List[str], aux_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        message_map = aux_data.get("message_map", [])
        index_map = aux_data.get("message_index_map", {})
        segment: List[Dict[str, Any]] = []
        for mid in message_ids:
            idx = index_map.get(mid)
            if idx is not None and 0 <= idx < len(message_map):
                segment.append(message_map[idx])
        return segment

    def _message_identifier(self, envelope: Dict[str, Any]) -> Optional[str]:
        if envelope.get("id") is not None:
            return str(envelope.get("id"))
        if envelope.get("key") is not None:
            return str(envelope.get("key"))
        return None

    def _lookup_message_envelope(
        self, message_id: str, aux_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        index_map = aux_data.get("message_index_map", {})
        message_map = aux_data.get("message_map", [])
        key = str(message_id)
        idx = index_map.get(key)
        if idx is None or idx < 0 or idx >= len(message_map):
            return None
        return message_map[idx]

    def _create_action_from_decision(
        self,
        decision: Dict[str, Any],
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not decision:
            return None
        text = decision.get("text") if isinstance(decision, dict) else str(decision)
        if not text:
            return None
        action = {
            "text": text,
            "owner": decision.get("owner") or decision.get("assignee"),
            "due_raw": decision.get("due_raw") or decision.get("due"),
            "due": decision.get("due"),
            "priority": decision.get("priority") or "normal",
            "status": decision.get("status") or "open",
        }
        msg_id = self._find_message_id_for_text(text, aux_data.get("messages", []))
        if msg_id:
            action["msg_id"] = msg_id
        action["topic_title"] = self._guess_topic_title(summary.get("topics", []), text)
        return action

    def _create_risk_entry(
        self,
        risk_entry: Any,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not risk_entry:
            return None
        if isinstance(risk_entry, dict):
            text = risk_entry.get("text") or risk_entry.get("description", "")
            likelihood = risk_entry.get("likelihood") or "medium"
            impact = risk_entry.get("impact") or "medium"
            mitigation = risk_entry.get("mitigation")
        else:
            text = str(risk_entry)
            likelihood = "medium"
            impact = "medium"
            mitigation = None
        if not text:
            return None
        risk = {
            "text": text,
            "likelihood": likelihood,
            "impact": impact,
        }
        if mitigation:
            risk["mitigation"] = mitigation
        msg_id = self._find_message_id_for_text(text, aux_data.get("messages", []))
        if msg_id:
            risk["msg_id"] = msg_id
        risk["topic_title"] = self._guess_topic_title(summary.get("topics", []), text)
        return risk

    def _guess_topic_title(
        self, topics: List[Dict[str, Any]], text: Optional[str]
    ) -> str:
        if not topics:
            return ""
        if not text:
            return topics[0].get("title", "")
        lowered = text.lower()
        for topic in topics:
            summary_text = topic.get("summary", "").lower()
            if summary_text and lowered in summary_text or summary_text in lowered:
                return topic.get("title", "")
        return topics[0].get("title", "")

    def _build_message_envelope(
        self,
        msg: Dict[str, Any],
        fallback_index: Optional[int] = None,
        msg_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        original_ts = msg.get("date_utc") or msg.get("date") or ""
        from ..utils.datetime_utils import parse_datetime_utc

        try:
            dt_utc = parse_datetime_utc(original_ts, return_none_on_error=True, use_zoneinfo=True)
        except Exception:
            dt_utc = None

        dt_bkk = dt_utc.astimezone(ZoneInfo("Asia/Bangkok")) if dt_utc else None

        key = msg_key or self._message_key(msg, fallback_index)

        return {
            "id": msg.get("id"),
            "key": key,
            "ts_utc": dt_utc.isoformat() if dt_utc else "",
            "ts_bkk": dt_bkk.isoformat() if dt_bkk else "",
            "author": self._format_author_name(msg),
            "text": (msg.get("text") or "").strip(),
            "raw": msg,
        }

    def _derive_time_span(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        envelopes = [
            self._build_message_envelope(msg, fallback_index=idx)
            for idx, msg in enumerate(messages)
            if msg.get("text")
        ]
        if not envelopes:
            if messages:
                envelopes = [
                    self._build_message_envelope(messages[0], fallback_index=0),
                    self._build_message_envelope(
                        messages[-1], fallback_index=len(messages) - 1
                    ),
                ]
        start = envelopes[0].get("ts_bkk")
        end = envelopes[-1].get("ts_bkk")
        if not start or not end:
            return ""
        from ..utils.datetime_utils import parse_datetime_utc

        try:
            start_dt = parse_datetime_utc(start, return_none_on_error=True, use_zoneinfo=True)
            end_dt = parse_datetime_utc(end, return_none_on_error=True, use_zoneinfo=True)
            if start_dt and end_dt:
                return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')} BKK"
            return ""
        except Exception:
            return ""

    def _generate_topics(
        self,
        profile: str,
        legacy_summary: Dict[str, Any],
        message_map: List[Dict[str, Any]],
        fallback_span: str,
    ) -> List[Dict[str, Any]]:
        if profile == "broadcast":
            source = (
                legacy_summary.get("key_points")
                or legacy_summary.get("discussion")
                or []
            )
        else:
            source = (
                legacy_summary.get("discussion")
                or legacy_summary.get("key_points")
                or []
            )

        cleaned = []
        for item in source:
            normalized = self._clean_bullet(item)
            if normalized:
                cleaned.append(normalized)

        if len(cleaned) < 2:
            context_sentences = re.split(
                r"[\.?!]\s+", legacy_summary.get("context", "")
            )
            for sentence in context_sentences:
                sentence = self._strip_markdown(sentence).strip()
                if len(sentence.split()) >= 4 and sentence not in cleaned:
                    cleaned.append(sentence)
                if len(cleaned) >= 2:
                    break

        cleaned = cleaned[:6]
        if not cleaned:
            cleaned = [legacy_summary.get("context", "") or "Основная тема сессии"]

        segments = self._split_messages_for_topics(message_map, len(cleaned))
        topics = []

        for idx, text in enumerate(cleaned):
            segment = segments[idx] if idx < len(segments) else []
            message_ids = []
            for item in segment:
                message_id = item.get("id")
                if message_id is None:
                    message_id = item.get("key")
                if message_id is not None:
                    message_ids.append(str(message_id))
            span = self._topic_time_span(segment, fallback_span)
            title = self._build_topic_title(text)
            summary_text = self._normalize_summary(text)
            topics.append(
                {
                    "title": title,
                    "time_span": span,
                    "message_ids": message_ids,
                    "summary": summary_text,
                }
            )

        return topics

    def _is_small_session(
        self, messages_total: int, topics: List[Dict[str, Any]]
    ) -> bool:
        return messages_total < 5 or len(topics) < 2

    def _apply_small_session_policy(
        self,
        topics: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        discussion: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        rationale: str,
        message_map: List[Dict[str, Any]],
        legacy_summary: Dict[str, Any],
        fallback_span: str,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        Dict[str, Any],
    ]:
        policy_flags = ["small_session"]

        normalized_topics = topics[:1]
        if not normalized_topics:
            normalized_topics = [
                self._build_minimal_topic(message_map, legacy_summary, fallback_span)
            ]
        else:
            primary_topic = normalized_topics[0]
            if not primary_topic.get("message_ids") and message_map:
                anchor = message_map[0]
                msg_identifier = anchor.get("id") or anchor.get("key")
                if msg_identifier is not None:
                    primary_topic["message_ids"] = [str(msg_identifier)]
            if not primary_topic.get("time_span"):
                primary_topic["time_span"] = self._topic_time_span(
                    message_map[:1], fallback_span
                )

        primary_topic_title = normalized_topics[0].get("title", "Краткая сессия")

        normalized_claims: List[Dict[str, Any]] = []
        for claim in claims[:3]:
            claim_copy = dict(claim)
            claim_copy["topic_title"] = primary_topic_title
            normalized_claims.append(claim_copy)

        normalized_discussion = discussion[:3]

        rationale_override = "insufficient_evidence"

        policy_info = {
            "score_cap": 60.0,
            "policy_flags": policy_flags,
        }

        return (
            normalized_topics,
            normalized_claims,
            normalized_discussion,
            [],
            [],
            rationale_override,
            policy_info,
        )

    def _build_minimal_topic(
        self,
        message_map: List[Dict[str, Any]],
        legacy_summary: Dict[str, Any],
        fallback_span: str,
    ) -> Dict[str, Any]:
        if message_map:
            anchor = message_map[0]
            summary_candidate = anchor.get("text", "") or legacy_summary.get(
                "context", "Краткий обзор"
            )
            message_id = anchor.get("id") or anchor.get("key")
            message_ids = [str(message_id)] if message_id is not None else []
            segment = [anchor]
        else:
            summary_candidate = legacy_summary.get("context", "Краткий обзор сессии")
            message_ids = []
            segment = []

        summary_text = self._normalize_summary(summary_candidate or "Краткий обзор")
        title = self._build_topic_title(summary_text)
        time_span = self._topic_time_span(segment, fallback_span)

        return {
            "title": title,
            "time_span": time_span,
            "message_ids": message_ids,
            "summary": summary_text,
        }

    def _clean_bullet(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        cleaned = cleaned.replace("- [ ]", "").replace("- [x]", "").strip()
        cleaned = cleaned.replace("•", "").strip()
        cleaned = self._strip_markdown(cleaned)
        return cleaned

    def _split_messages_for_topics(
        self, message_map: List[Dict[str, Any]], topic_count: int
    ) -> List[List[Dict[str, Any]]]:
        if topic_count <= 0 or not message_map:
            return []
        if topic_count == 1:
            return [message_map]

        total = len(message_map)
        base = max(1, total // topic_count)
        segments: List[List[Dict[str, Any]]] = []
        start_idx = 0

        for i in range(topic_count):
            end_idx = start_idx + base
            if i == topic_count - 1:
                end_idx = total
            segment = message_map[start_idx:end_idx]
            if not segment and message_map:
                segment = [message_map[min(start_idx, total - 1)]]
            segments.append(segment)
            start_idx = end_idx

        return segments

    def _topic_time_span(
        self, segment: List[Dict[str, Any]], fallback_span: str
    ) -> str:
        if not segment:
            return fallback_span
        start = segment[0].get("ts_bkk")
        end = segment[-1].get("ts_bkk")
        from ..utils.datetime_utils import parse_datetime_utc

        try:
            start_dt = parse_datetime_utc(start, return_none_on_error=True, use_zoneinfo=True) if start else None
            end_dt = parse_datetime_utc(end, return_none_on_error=True, use_zoneinfo=True) if end else None
            if start_dt and end_dt:
                if start_dt.date() == end_dt.date():
                    return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')} BKK"
                return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%Y-%m-%d %H:%M')} BKK"
        except Exception:
            return fallback_span
        return fallback_span

    def _build_topic_title(self, summary_text: str) -> str:
        clean_text = self._strip_markdown(summary_text)
        words = [w for w in re.split(r"\s+", clean_text) if w]
        if not words:
            return "Основная тема"
        max_words = min(max(len(words), 5), 8)
        title_words = words[:max_words]
        title = " ".join(title_words)
        title = title.strip().rstrip(":;,.")
        return title[:80].capitalize()

    def _generate_claims(
        self,
        profile: str,
        topics: List[Dict[str, Any]],
        message_map: List[Dict[str, Any]],
        entities: Dict[str, Any],
        addons_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not topics:
            return []

        entity_candidates = self._flatten_entities(entities)[:4]
        claims: List[Dict[str, Any]] = []
        modality = "media" if profile == "broadcast" else "internal"
        source = modality

        per_message_addons = addons_info.get("by_key", {}) if addons_info else {}
        global_asset_tags = addons_info.get("asset_tags", []) if addons_info else []
        global_geo_tags = addons_info.get("geo_entities", []) if addons_info else []
        addons_set = addons_info.get("addons", set()) if addons_info else set()

        for topic in topics:
            linked_messages = [
                item
                for item in message_map
                if str(item.get("id")) in topic.get("message_ids", [])
            ]
            if not linked_messages:
                anchor = message_map[0] if message_map else None
                if not anchor:
                    continue
                summary_text = self._build_claim_summary(
                    anchor.get("text", ""), topic.get("summary", "")
                )
                summary_text = summary_text[:160]  # Ограничение для избежания штрафа
                claim = {
                    "ts": anchor.get("ts_bkk", ""),
                    "source": source,
                    "modality": modality,
                    "credibility": "medium",
                    "entities": entity_candidates,
                    "summary": summary_text,
                    "msg_id": str(anchor.get("id"))
                    if anchor.get("id") is not None
                    else "",
                    "topic_title": topic.get("title"),
                }
                self._apply_addon_metadata_to_claim(
                    claim,
                    per_message_addons.get(anchor.get("key", "")),
                    addons_set,
                    global_asset_tags,
                    global_geo_tags,
                )
                claims.append(claim)
                continue

            for item in linked_messages[:3]:
                summary_text = self._build_claim_summary(
                    item.get("text", ""), topic.get("summary", "")
                )
                if not summary_text:
                    continue
                summary_text = summary_text[:160]  # Ограничение для избежания штрафа
                claim = {
                    "ts": item.get("ts_bkk", ""),
                    "source": source,
                    "modality": modality,
                    "credibility": "medium",
                    "entities": entity_candidates,
                    "summary": summary_text,
                    "msg_id": str(item.get("id")) if item.get("id") is not None else "",
                    "topic_title": topic.get("title"),
                }
                self._apply_addon_metadata_to_claim(
                    claim,
                    per_message_addons.get(item.get("key", "")),
                    addons_set,
                    global_asset_tags,
                    global_geo_tags,
                )
                claims.append(claim)
                if len(claims) >= 20:
                    return claims

        credibility_order = {"high": 0, "medium": 1, "low": 2}
        sorted_claims = sorted(
            claims,
            key=lambda item: (
                credibility_order.get(item.get("credibility", "medium"), 1),
                item.get("ts", ""),
            ),
        )
        return sorted_claims[:20]

    def _apply_addon_metadata_to_claim(
        self,
        claim: Dict[str, Any],
        message_addon: Optional[Dict[str, Any]],
        active_addons: set,
        global_asset_tags: List[str],
        global_geo_tags: List[str],
    ) -> None:
        """Обогащает claim доменными метаданными, если они доступны."""

        if message_addon:
            asset_tags = message_addon.get("asset_tags")
            if asset_tags:
                claim["asset_tags"] = asset_tags

            geo_entities = message_addon.get("geo_entities")
            if geo_entities:
                claim["geo_scope"] = geo_entities

            if message_addon.get("sci_markers"):
                claim["field"] = "sci-tech"

        # Если у конкретного сообщения нет тегов, но есть глобальные — добавляем мягко
        if "asset_tags" not in claim and global_asset_tags:
            claim["asset_tags"] = global_asset_tags

        if "geo_scope" not in claim and global_geo_tags:
            claim["geo_scope"] = global_geo_tags

        if "field" not in claim and "sci-tech" in active_addons:
            claim["field"] = "sci-tech"

    def _generate_discussion(
        self,
        profile: str,
        message_map: List[Dict[str, Any]],
        limit_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not message_map:
            return []

        limit = limit_override or (6 if profile == "broadcast" else 8)
        min_items = 3 if profile == "broadcast" else 4
        key_messages = self._select_key_messages(
            [item["raw"] for item in message_map], limit=limit
        )

        timeline = []
        for idx, msg in enumerate(key_messages[:limit]):
            envelope = self._build_message_envelope(msg, fallback_index=idx)
            quote = self._truncate_text(envelope["text"].replace("\n", " "), 220)
            if not quote:
                continue
            msg_identifier = envelope.get("id")
            if msg_identifier is None:
                msg_identifier = envelope.get("key")
            timeline.append(
                {
                    "ts": envelope.get("ts_bkk", ""),
                    "author": envelope.get("author", ""),
                    "msg_id": str(msg_identifier) if msg_identifier is not None else "",
                    "quote": quote,
                }
            )

        if len(timeline) < min_items and message_map:
            filler = message_map[:min_items]
            for envelope in filler:
                quote = self._truncate_text(envelope["text"], 220)
                if not quote:
                    continue
                msg_identifier = envelope.get("id")
                if msg_identifier is None:
                    msg_identifier = envelope.get("key")
                timeline.append(
                    {
                        "ts": envelope.get("ts_bkk", ""),
                        "author": envelope.get("author", ""),
                        "msg_id": str(msg_identifier)
                        if msg_identifier is not None
                        else "",
                        "quote": quote,
                    }
                )

        # Deduplicate by msg_id
        seen = set()
        deduped = []
        for item in timeline:
            key = item.get("msg_id"), item.get("quote")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped[:limit]

    def _generate_actions(
        self,
        topics: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        actions = []
        topic_title = topics[0]["title"] if topics else ""
        for decision in decisions:
            text = decision.get("text", "").strip()
            if not text:
                continue
            msg_id = self._find_message_id_for_text(text, messages)
            status = "done" if text.lower().startswith("- [x]") else "open"
            priority_raw = (decision.get("priority") or "").upper()
            priority_map = {"P1": "high", "P2": "normal", "P3": "low"}
            priority = priority_map.get(
                priority_raw, priority_raw.lower() if priority_raw else "normal"
            )
            actions.append(
                {
                    "text": self._clean_bullet(text),
                    "owner": decision.get("owner"),
                    "due_raw": decision.get("due"),
                    "due": decision.get("due"),
                    "priority": priority,
                    "status": status,
                    "msg_id": msg_id,
                    "topic_title": topic_title,
                }
            )

        return actions

    def _generate_risks(
        self,
        topics: List[Dict[str, Any]],
        risks_text: List[str],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        risks = []
        topic_title = topics[0]["title"] if topics else ""
        for risk in risks_text:
            cleaned = self._clean_bullet(risk)
            if not cleaned:
                continue
            msg_id = self._find_message_id_for_text(cleaned, messages)
            risks.append(
                {
                    "text": cleaned,
                    "likelihood": None,
                    "impact": None,
                    "mitigation": None,
                    "msg_id": msg_id,
                    "topic_title": topic_title,
                }
            )
        return risks

    def _flatten_entities(self, entities: Dict[str, Any]) -> List[str]:
        if not entities:
            return []
        buckets = ["mentions", "tickers", "organizations", "people", "locations"]
        result = []
        for bucket in buckets:
            for item in entities.get(bucket, [])[:5]:
                value = item.get("value") if isinstance(item, dict) else str(item)
                if value and value not in result:
                    result.append(value)
        return result[:20]

    def _build_attachments(self, artifacts: List[str]) -> List[str]:
        attachments = []
        for artifact in artifacts or []:
            if not artifact:
                continue
            if artifact.startswith("http"):
                attachments.append(f"link:{artifact.split()[0]}")
            elif artifact.startswith("📎"):
                attachments.append(f"doc:{artifact.replace('📎', '').strip()}")
            else:
                attachments.append(f"link:{artifact}")
        return attachments[:20]

    def _derive_rationale(
        self, profile: str, actions: List[Dict[str, Any]], risks: List[Dict[str, Any]]
    ) -> str:
        if profile == "broadcast":
            if not actions:
                return "news_channel_no_actions"
            if not risks:
                return "no_risks_detected"
            return "author_opinion_no_tasks"
        else:
            if actions:
                return "project_session_with_actions"
            if not risks:
                return "no_risks_detected"
            return "threads_not_applicable"

    def _evaluate_schema_requirements(
        self,
        profile: str,
        topics: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        discussion: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        thread_count: int,
    ) -> Dict[str, Any]:
        issues: List[str] = []

        topics_count = len(topics)
        claims_count = len(claims)
        actions_count = len(actions)
        risks_count = len(risks)
        discussion_count = len(discussion)
        replies_present = any(msg.get("reply_to") for msg in messages)

        topics_ok = topics_count >= 2
        if not topics_ok:
            issues.append("topics_minimum_not_met")
        if topics_count > 6:
            issues.append("topics_exceed_maximum")

        claims_ok = claims_count >= 1
        if not claims_ok:
            issues.append("claims_minimum_not_met")

        discussion_min = 3 if profile == "broadcast" else 4
        discussion_ok = discussion_count >= discussion_min
        if not discussion_ok:
            issues.append("discussion_minimum_not_met")

        if profile == "group-project":
            actions_ok = actions_count >= 2
            if not actions_ok:
                issues.append("actions_minimum_not_met")
            risks_ok = risks_count >= 1
            if not risks_ok:
                issues.append("risks_minimum_not_met")
        else:
            actions_ok = actions_count >= 1
            risks_ok = risks_count >= 1

        threads_ok = (not replies_present) or thread_count >= 1
        if replies_present and not threads_ok:
            issues.append("threads_missing")

        structure_ok = (
            topics_ok
            and claims_ok
            and (profile != "group-project" or (actions_ok and risks_ok))
        )

        claims_coverage_target = 0.25 if profile == "broadcast" else 0.35
        # claims_coverage tracked externally; ok flag filled later when value available

        blocking_issues = [
            issue
            for issue in issues
            if issue
            in {
                "topics_minimum_not_met",
                "claims_minimum_not_met",
                "actions_minimum_not_met",
                "risks_minimum_not_met",
                "threads_missing",
            }
        ]

        return {
            "issues": issues,
            "blocking_issues": blocking_issues,
            "topics_ok": topics_ok,
            "claims_ok": claims_ok,
            "actions_ok": actions_ok,
            "risks_ok": risks_ok,
            "threads_ok": threads_ok,
            "structure_ok": structure_ok,
            "discussion_ok": discussion_ok,
            "claims_coverage_target": claims_coverage_target,
            "claims_coverage_ok": True,  # placeholder, recalculated by caller
        }

    def _estimate_duplicate_rate(self, message_map: List[Dict[str, Any]]) -> float:
        if not message_map:
            return 0.0

        normalized_texts = []
        for envelope in message_map:
            text = self._strip_markdown(envelope.get("text", "")).lower()
            if text:
                normalized_texts.append(text)

        if not normalized_texts:
            return 0.0

        unique_texts = set(normalized_texts)
        duplicates = len(normalized_texts) - len(unique_texts)
        total = len(normalized_texts)
        if total <= 0:
            return 0.0
        return round(max(0.0, duplicates / total), 3)

    def _compute_quality_score(
        self,
        profile: str,
        coverage: float,
        claims_coverage: float,
        schema_eval: Dict[str, Any],
        dup_rate: float,
        claims: List[Dict[str, Any]],
    ) -> Tuple[float, int]:
        coverage_component = min(1.0, max(0.0, coverage))
        claims_component = min(1.0, max(0.0, claims_coverage))

        if profile == "broadcast":
            weights = {
                "coverage": 0.35,
                "claims_coverage": 0.10,
                "actions_ok": 0.05,
                "risks_ok": 0.05,
                "threads_ok": 0.05,
                "dedup": 0.15,
                "structure_ok": 0.25,
            }
        else:
            weights = {
                "coverage": 0.20,
                "claims_coverage": 0.10,
                "actions_ok": 0.30,
                "risks_ok": 0.15,
                "threads_ok": 0.10,
                "dedup": 0.05,
                "structure_ok": 0.10,
            }

        actions_component = 1.0 if schema_eval["actions_ok"] else 0.0
        risks_component = 1.0 if schema_eval["risks_ok"] else 0.0
        threads_component = 1.0 if schema_eval["threads_ok"] else 0.0
        structure_component = 1.0 if schema_eval["structure_ok"] else 0.0
        dedup_component = 1.0 - min(1.0, max(0.0, dup_rate))

        weighted_sum = (
            coverage_component * weights["coverage"]
            + claims_component * weights["claims_coverage"]
            + actions_component * weights["actions_ok"]
            + risks_component * weights["risks_ok"]
            + threads_component * weights["threads_ok"]
            + dedup_component * weights["dedup"]
            + structure_component * weights["structure_ok"]
        )

        len_penalty = 0
        for claim in claims:
            length = len(claim.get("summary", "") or "")
            if length <= 160:
                continue
            over_soft = min(length, 220) - 160
            if over_soft > 0:
                len_penalty += ((over_soft - 1) // 30 + 1) * 1
            if length > 220:
                over_hard = min(length, 300) - 220
                if over_hard > 0:
                    len_penalty += ((over_hard - 1) // 30 + 1) * 2

        score = max(0.0, min(100.0, weighted_sum * 100 - len_penalty))
        return score, len_penalty

    def _calculate_coverage(
        self,
        topics: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        messages_total: int,
    ) -> Tuple[float, float]:
        topic_message_ids = set()
        for topic in topics:
            topic_message_ids.update(topic.get("message_ids", []))
        coverage = (
            round(len(topic_message_ids) / messages_total, 3) if messages_total else 0.0
        )

        claim_message_ids = {
            claim.get("msg_id") for claim in claims if claim.get("msg_id")
        }
        claims_coverage = (
            round(len(claim_message_ids) / len(topic_message_ids), 3)
            if topic_message_ids
            else 0.0
        )

        return coverage, claims_coverage

    def _count_threads(self, messages: List[Dict[str, Any]]) -> int:
        thread_roots = set()
        for msg in messages:
            reply = msg.get("reply_to")
            root_id: Optional[str] = None
            if isinstance(reply, dict):
                root_id = reply.get("msg_id") or reply.get("id")
            elif isinstance(reply, (str, int)):
                root_id = str(reply)

            if root_id:
                thread_roots.add(root_id)
        return len(thread_roots)

    def _derive_quality_status(
        self,
        score: float,
        schema_eval: Dict[str, Any],
        legacy_summary: Dict[str, Any],
        legacy_metrics,
    ) -> str:
        improved = legacy_summary.get("quality_metrics", {}).get("improved")

        if (
            score >= 85
            and schema_eval["structure_ok"]
            and not schema_eval["blocking_issues"]
        ):
            return "accepted"

        if improved:
            return "refined"

        return "needs_review"

    def _find_message_id_for_text(
        self, text: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        plain = self._strip_markdown(text).lower()
        if not plain:
            return None
        snippet = plain[:40]
        for idx, msg in enumerate(messages):
            body = self._strip_markdown(msg.get("text", "")).lower()
            if snippet and snippet in body:
                msg_id = msg.get("id")
                if msg_id is not None:
                    return str(msg_id)
                return self._message_key(msg, fallback_index=idx)
        return None

    def _strip_markdown(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
        cleaned = re.sub(r"[`*_#>|~]+", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _normalize_summary(self, text: str, max_chars: int = 200) -> str:
        normalized = self._strip_markdown(text)
        if len(normalized) <= max_chars:
            return normalized
        truncated = normalized[:max_chars].rsplit(" ", 1)[0]
        return truncated.rstrip(",:;") + "..."

    def _build_claim_summary(self, message_text: str, fallback: str) -> str:
        candidate = self._normalize_summary(message_text, max_chars=240)
        if len(candidate.split()) >= 5:
            return candidate
        return self._normalize_summary(fallback, max_chars=240)

    def _format_links_artifacts(self, entities: Dict[str, Any]) -> List[str]:
        """
        Форматирование ссылок и артефактов

        Args:
            entities: Извлечённые сущности

        Returns:
            Список ссылок и артефактов
        """
        artifacts: List[str] = []
        seen: set[str] = set()

        links = entities.get("links", []) or entities.get("urls", [])
        for link_info in links[:15]:
            raw = link_info.get("value", "")
            count = link_info.get("count", 1)
            normalized = self._normalize_attachment("link", raw, count=count)
            if normalized and normalized not in seen:
                artifacts.append(normalized)
                seen.add(normalized)

        files = entities.get("files", [])
        for file_info in files[:10]:
            raw = file_info.get("value", "")
            count = file_info.get("count", 1)
            normalized = self._normalize_attachment("doc", raw, count=count)
            if normalized and normalized not in seen:
                artifacts.append(normalized)
                seen.add(normalized)

        return artifacts

    def _normalize_attachment(
        self, kind: str, value: str, *, count: int = 1
    ) -> Optional[str]:
        if not value:
            return None
        clean_value = value.strip()
        clean_value = clean_value.strip('*_"')

        if kind == "link":
            clean_value = self._sanitize_url(clean_value)
            if not clean_value:
                return None
        else:
            clean_value = re.sub(r"\s+", " ", clean_value)

        suffix = f" ({count}x)" if count and count > 1 else ""
        return f"{kind}:{clean_value}{suffix}"

    def _sanitize_url(self, url: str) -> Optional[str]:
        from urllib.parse import urlparse

        candidate = url.strip()
        candidate = candidate.split()[0]
        candidate = candidate.rstrip(").,;")
        if candidate.startswith("["):
            candidate = candidate.strip("[]")
        if candidate.startswith("www."):
            candidate = f"https://{candidate}"

        parsed = urlparse(candidate)
        if not parsed.scheme:
            candidate = f"https://{candidate}"
            parsed = urlparse(candidate)
        if not parsed.netloc:
            return None
        normalized = parsed._replace(fragment="").geturl()
        return normalized


async def summarize_chat_sessions(
    messages: List[Dict[str, Any]],
    chat_name: str,
    embedding_client: Optional[LMStudioEmbeddingClient] = None,
    summaries_dir: Path = Path("artifacts/reports"),
    instruction_manager: Optional[InstructionManager] = None,
) -> List[Dict[str, Any]]:
    """
    Удобная функция для саммаризации всех сессий в чате

    Args:
        messages: Список сообщений
        chat_name: Название чата
        embedding_client: Клиент для генерации эмбеддингов и текста (LM Studio)
        summaries_dir: Директория с саммаризациями для контекста

    Returns:
        Список саммаризаций сессий
    """
    # Сегментируем на сессии
    segmenter = SessionSegmenter()
    sessions = segmenter.segment_messages(messages, chat_name)

    # Саммаризируем каждую сессию
    summarizer = SessionSummarizer(
        embedding_client,
        summaries_dir,
        instruction_manager=instruction_manager,
    )
    summaries = []

    for session in sessions:
        try:
            summary = await summarizer.summarize_session(session)
            summaries.append(summary)

            # Небольшая задержка между запросами
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Ошибка при саммаризации сессии {session['session_id']}: {e}")
            continue

    return summaries


if __name__ == "__main__":
    # Тест модуля
    from datetime import timedelta

    # Создаём тестовые сообщения
    base_time = datetime.now(ZoneInfo("UTC"))
    test_messages = [
        {
            "id": "1",
            "date_utc": base_time.isoformat(),
            "text": "Привет! Давайте обсудим TON",
            "from": {"username": "alice"},
            "language": "ru",
        },
        {
            "id": "2",
            "date_utc": (base_time + timedelta(minutes=5)).isoformat(),
            "text": "Да, нужно принять решение о бюджете",
            "from": {"username": "bob"},
            "language": "ru",
        },
    ]

    async def test():
        summaries = await summarize_chat_sessions(test_messages, "TestChat")
        print(f"Создано саммаризаций: {len(summaries)}")
        for summary in summaries:
            print(f"  {summary['session_id']}: {summary.get('context', 'N/A')[:100]}")

    asyncio.run(test())
