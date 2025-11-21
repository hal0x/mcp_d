#!/usr/bin/env python3
"""
Модуль для структурной саммаризации сессий
Согласно спецификации TelegramDumpManager_Spec.md
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

from ....config import get_settings
from ....core.adapters.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory
from ....core.adapters.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env
from ....utils.system.naming import slugify
from ...context.context_manager import ContextManager
from ...entities.entity_extraction import EntityExtractor
from ...context.incremental_context_manager import IncrementalContextManager
from ...utils.instruction_manager import InstructionManager
from ..quality_evaluator import IterativeRefiner, QualityEvaluator
from .core.batch import summarize_batch_sessions
from .core.canonical import build_canonical_summary as build_canonical_summary_internal
from .prompts import (
    create_summarization_prompt,
    ensure_summary_completeness,
    generate_summary_with_lmql,
    parse_summary_structure,
)
from .core.quality import refresh_quality
from .core.refinement import run_structural_pass
from .utils import (
    detect_chat_mode,
    extract_action_items,
    format_links_artifacts,
    map_profile,
    prepare_conversation_text,
)

if TYPE_CHECKING:
    from ...aggregation.large_context_processor import LargeContextProcessor

logger = logging.getLogger(__name__)


class SessionSummarizer:
    """Класс для структурной саммаризации сессий.
    
    Args:
        strict_mode: Если True, выбрасывает исключения вместо использования fallback при ошибках LLM.
                     По умолчанию False для обратной совместимости."""

    def __init__(
        self,
        embedding_client: Optional[LangChainLLMAdapter] = None,
        summaries_dir: Path = Path("artifacts/reports"),
        instruction_manager: Optional[InstructionManager] = None,
        enable_quality_check: bool = True,
        enable_iterative_refinement: bool = True,
        min_quality_score: float = 80.0,
        strict_mode: bool = False,
        lmql_adapter: Optional[LMQLAdapter] = None,
    ):
        """
        Инициализация саммаризатора

        Args:
            embedding_client: LangChain LLM клиент для генерации текста (если None, создаётся новый)
            summaries_dir: Директория с саммаризациями для контекста
            instruction_manager: Менеджер специальных инструкций
            enable_quality_check: Включить проверку качества
            enable_iterative_refinement: Включить итеративное улучшение
            min_quality_score: Минимальный приемлемый балл качества
            strict_mode: Если True, выбрасывает исключения вместо использования fallback при ошибках LLM
            lmql_adapter: Опциональный LMQL адаптер для структурированной генерации.
                         Если не указан, создается из настроек окружения.
        """
        if embedding_client is None:
            embedding_client = get_llm_client_factory()
            if embedding_client is None:
                raise ValueError(
                    "Не удалось инициализировать LangChain LLM клиент. "
                    "Убедитесь, что LangChain установлен и MEMORY_MCP_LMSTUDIO_LLM_MODEL настроен."
                )
        self.embedding_client = embedding_client
        settings = get_settings()
        self.entity_extractor = EntityExtractor()
        self.context_manager = ContextManager(summaries_dir)
        self.incremental_context_manager = IncrementalContextManager()
        self.instruction_manager = instruction_manager or InstructionManager()

        self.enable_quality_check = enable_quality_check
        self.enable_iterative_refinement = enable_iterative_refinement
        self.quality_evaluator = QualityEvaluator(min_quality_score=min_quality_score)
        self.iterative_refiner = IterativeRefiner(
            self,
            max_iterations=5,
            target_score=max(85.0, min_quality_score - 2.0),  # Целевой балл чуть ниже min_quality для margin
        )

        from ...aggregation.large_context_processor import LargeContextProcessor
        
        settings = get_settings()
        self.large_context_processor = LargeContextProcessor(
            max_tokens=settings.large_context_max_tokens,
            prompt_reserve_tokens=settings.large_context_prompt_reserve,
            embedding_client=self.embedding_client,
        )
        
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None
            logger.debug("LMQL адаптер не настроен для SessionSummarizer")
        
        self.strict_mode = strict_mode

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

        chat_mode = detect_chat_mode(messages)
        previous_context = self.context_manager.get_previous_context(chat, session_id)

        extended_context = None
        if len(messages) < 20:  # Расширенный контекст для малых сессий
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

        entities = self.entity_extractor.extract_from_messages(messages)

        estimated_tokens = self.large_context_processor.estimate_messages_tokens(messages)
        use_large_context = estimated_tokens > 100000  # Обработка большим контекстом для сессий > 100K токенов

        if use_large_context:
            logger.info(
                f"Используется обработка большим контекстом для сессии {session_id} "
                f"(~{estimated_tokens} токенов)"
            )
            base_prompt = create_summarization_prompt(
                "",
                chat,
                dominant_language,
                session,
                chat_mode,
                previous_context,
                extended_context,
                instruction_manager=self.instruction_manager,
                incremental_context_manager=self.incremental_context_manager,
            )
            
            large_context_result = await self.large_context_processor.process_large_context(
                messages, chat, base_prompt
            )
            
            summary_text = large_context_result.get("summary", "")
            if not summary_text and large_context_result.get("detailed_summaries"):
                summary_text = "\n\n".join([
                    s.get("summary", "") for s in large_context_result["detailed_summaries"]
                ])
        else:
            conversation_text = prepare_conversation_text(messages)
            prompt = create_summarization_prompt(
                conversation_text,
                chat,
                dominant_language,
                session,
                chat_mode,
                previous_context,
                extended_context,
                instruction_manager=self.instruction_manager,
                incremental_context_manager=self.incremental_context_manager,
            )

            summary_structure = None
            summary_text = ""
            
            if self.lmql_adapter:
                try:
                    logger.debug("Используется LMQL для генерации структурированной саммаризации")
                    summary_structure = await generate_summary_with_lmql(
                        self.lmql_adapter, prompt, chat_mode, dominant_language
                    )
                except Exception as e:
                    logger.warning(f"Ошибка при использовании LMQL для саммаризации: {e}, используем LLM")
            
            if not summary_structure:
                async with self.embedding_client:
                    summary_text = await self.embedding_client.generate_summary(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=30000,  # Уменьшено для предотвращения таймаутов
                        top_p=0.93,
                        presence_penalty=0.05,
                    )
                summary_structure = parse_summary_structure(summary_text)

        summary_structure, fallback_used = ensure_summary_completeness(
            messages, chat, summary_structure, strict_mode=self.strict_mode
        )

        if fallback_used:
            logger.debug(
                f"Саммаризация {session_id} дополнена эвристиками (это нормально)"
            )

        action_items = await extract_action_items(messages, summary_structure)

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
            "links_artifacts": format_links_artifacts(entities),
            "raw_summary": summary_text,
            "fallback_used": fallback_used,
        }

        metrics = None

        if self.enable_quality_check:
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

            if not self.quality_evaluator.is_acceptable(metrics):
                logger.warning(
                    f"Качество саммаризации {session_id} ниже порога ({metrics.score:.1f}). Будет применён структурный IterativeRefiner."
                )
            else:
                logger.info("Качество саммаризации приемлемо")

        canonical_summary, aux_data = build_canonical_summary_internal(
            session=session,
            messages=messages,
            legacy_summary=legacy_summary,
            quality_metrics=metrics,
            chat_mode=chat_mode,
            profile=map_profile(chat_mode, messages),
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

    async def summarize_batch_sessions(
        self,
        sessions: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Саммаризация нескольких сессий в одном запросе с использованием накопительного контекста.

        Args:
            sessions: Список сессий для саммаризации
            chat_name: Название чата
            accumulative_context: Накопительный контекст для улучшения понимания

        Returns:
            Список саммаризаций сессий
        """
        return await summarize_batch_sessions(
            sessions,
            chat_name,
            self.large_context_processor,
            self.context_manager,
            accumulative_context,
        )



    def _run_structural_pass(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        session: Dict[str, Any],
        iteration: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Выполняет структурный проход рефайнинга."""
        return run_structural_pass(summary, aux_data, session, iteration)

    def _refresh_quality(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Пересчитывает метрики и качество для канонической саммаризации."""
        return refresh_quality(summary, aux_data)
