#!/usr/bin/env python3
"""
Модуль для оценки качества саммаризации и детерминированного улучшения.
"""

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Метрики качества саммаризации"""

    score: float  # Общий балл 0-100
    has_context: bool  # Есть контекст
    context_length: int  # Длина контекста в символах
    has_discussion: bool  # Есть пункты дискуссии
    discussion_count: int  # Количество пунктов
    has_decisions: bool  # Есть решения
    decisions_count: int  # Количество решений
    has_risks: bool  # Есть риски
    risks_count: int  # Количество рисков
    has_links: bool  # Есть ссылки
    links_count: int  # Количество ссылок
    language_matches: bool  # Язык совпадает с ожидаемым
    issues: List[str]  # Список проблем
    suggestions: List[str]  # Список предложений по улучшению


class QualityEvaluator:
    """Оценщик качества саммаризации"""

    def __init__(
        self,
        min_context_length: int = 30,
        min_discussion_items: int = 2,
        min_quality_score: float = 80.0,
    ):
        """
        Инициализация оценщика

        Args:
            min_context_length: Минимальная длина контекста
            min_discussion_items: Минимальное количество пунктов дискуссии
            min_quality_score: Минимальный приемлемый балл качества
        """
        self.min_context_length = min_context_length
        self.min_discussion_items = min_discussion_items
        self.min_quality_score = min_quality_score

    def evaluate(
        self, summary: Dict[str, Any], expected_language: str = "ru"
    ) -> QualityMetrics:
        """
        Оценка качества саммаризации

        Args:
            summary: Саммаризация для оценки
            expected_language: Ожидаемый язык (ru, en, etc.)

        Returns:
            QualityMetrics с детальной оценкой
        """
        issues = []
        suggestions = []

        # Проверяем контекст
        context = summary.get("context", "").strip()
        has_context = len(context) > 10
        context_length = len(context)

        if not has_context:
            issues.append("Контекст отсутствует или слишком короткий")
            suggestions.append("Добавить описание общего контекста разговора")
        elif context_length < self.min_context_length:
            issues.append(f"Контекст слишком короткий ({context_length} символов)")
            suggestions.append(
                f"Расширить контекст до минимум {self.min_context_length} символов"
            )

        # Проверяем язык контекста
        language_matches = self._check_language(context, expected_language)
        if not language_matches and has_context:
            issues.append(
                f"Контекст на неправильном языке (ожидался {expected_language})"
            )
            suggestions.append(f"Переписать контекст на {expected_language}")

        # Проверяем дискуссию
        discussion = summary.get("discussion", [])
        has_discussion = len(discussion) > 0
        discussion_count = len(discussion)

        if not has_discussion:
            issues.append("Ход дискуссии не структурирован")
            suggestions.append("Выделить основные пункты обсуждения")
        elif discussion_count < self.min_discussion_items:
            issues.append(f"Слишком мало пунктов дискуссии ({discussion_count})")
            suggestions.append("Добавить больше деталей дискуссии")

        # Проверяем решения
        decisions = summary.get("decisions_next", [])
        has_decisions = len(decisions) > 0
        decisions_count = len(decisions)

        # Проверяем риски
        risks = summary.get("risks_open", [])
        has_risks = len(risks) > 0
        risks_count = len(risks)

        # Проверяем ссылки
        links = summary.get("links_artifacts", [])
        has_links = len(links) > 0
        links_count = len(links)

        # Проверяем наличие ссылок в исходных сообщениях
        message_count = summary.get("message_count", 0)
        if message_count > 0 and not has_links:
            # Это не всегда проблема, но стоит проверить
            suggestions.append("Проверить наличие ссылок и артефактов в сообщениях")

        # Вычисляем общий балл
        score = self._calculate_score(
            has_context=has_context,
            context_length=context_length,
            has_discussion=has_discussion,
            discussion_count=discussion_count,
            has_decisions=has_decisions,
            decisions_count=decisions_count,
            has_risks=has_risks,
            risks_count=risks_count,
            has_links=has_links,
            links_count=links_count,
            language_matches=language_matches,
        )

        return QualityMetrics(
            score=score,
            has_context=has_context,
            context_length=context_length,
            has_discussion=has_discussion,
            discussion_count=discussion_count,
            has_decisions=has_decisions,
            decisions_count=decisions_count,
            has_risks=has_risks,
            risks_count=risks_count,
            has_links=has_links,
            links_count=links_count,
            language_matches=language_matches,
            issues=issues,
            suggestions=suggestions,
        )

    def _check_language(self, text: str, expected_language: str) -> bool:
        """
        Проверка языка текста

        Args:
            text: Текст для проверки
            expected_language: Ожидаемый язык

        Returns:
            True если язык совпадает
        """
        if not text or len(text) < 10:
            return True  # Слишком короткий текст

        if expected_language == "ru":
            # Проверяем наличие кириллицы
            cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
            latin_count = len(re.findall(r"[a-zA-Z]", text))

            # Русский текст должен содержать больше кириллицы
            return cyrillic_count > latin_count

        elif expected_language == "en":
            # Проверяем наличие латиницы
            cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
            latin_count = len(re.findall(r"[a-zA-Z]", text))

            # Английский текст должен содержать больше латиницы
            return latin_count > cyrillic_count

        # Для других языков пока не проверяем
        return True

    def _calculate_score(self, **kwargs) -> float:
        """
        Вычисление общего балла качества

        Returns:
            Балл от 0 до 100
        """
        score = 0.0

        # Контекст (30 баллов)
        if kwargs["has_context"]:
            context_score = min(30, (kwargs["context_length"] / 100) * 30)
            score += context_score

        # Дискуссия (30 баллов)
        if kwargs["has_discussion"]:
            discussion_score = min(30, kwargs["discussion_count"] * 10)
            score += discussion_score

        # Решения (15 баллов)
        if kwargs["has_decisions"]:
            decisions_score = min(15, kwargs["decisions_count"] * 7.5)
            score += decisions_score

        # Риски (10 баллов)
        if kwargs["has_risks"]:
            risks_score = min(10, kwargs["risks_count"] * 5)
            score += risks_score

        # Ссылки (5 баллов)
        if kwargs["has_links"]:
            links_score = min(5, kwargs["links_count"] * 2.5)
            score += links_score

        # Язык (10 баллов)
        if kwargs["language_matches"]:
            score += 10

        return min(100.0, score)

    def is_acceptable(self, metrics: QualityMetrics) -> bool:
        """
        Проверка приемлемости качества

        Args:
            metrics: Метрики качества

        Returns:
            True если качество приемлемо
        """
        return metrics.score >= self.min_quality_score


class IterativeRefiner:
    """Детерминированный улучшатель канонического summary (спецификация §9)."""

    def __init__(self, summarizer, max_iterations: int = 5, target_score: float = 85.0):
        self.summarizer = summarizer
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.last_iterations = 0
        self.iteration_history: List[Dict[str, Any]] = []

    def _should_stop_iteration(
        self,
        new_score: float,
        target_score: float,
        changed: bool,
        delta: float,
        iteration: int,
        pass_count: int,
        consecutive_no_change: int,
        max_consecutive_no_change: int,
    ) -> tuple[bool, str]:
        """
        Определяет, следует ли остановить итерации улучшения.
        
        Returns:
            Кортеж (should_stop, reason) - нужно ли остановиться и причина
        """
        # Целевой балл достигнут
        if new_score >= target_score:
            return True, "target_score_reached"
        
        # Слишком много итераций подряд без изменений
        if consecutive_no_change >= max_consecutive_no_change:
            return True, "consecutive_no_change"
        
        # Если были изменения, продолжаем
        if changed:
            return False, "changed"
        
        # Если еще не все итерации пройдены и прирост мал, продолжаем
        if iteration < pass_count:
            if delta < 2.0:
                return False, "small_delta_continue"
            return False, "continue"
        
        # Все итерации пройдены или прирост мал - останавливаемся
        if delta < 2.0:
            return True, "small_delta_stop"
        
        # Изменений не обнаружено
        return True, "no_changes"

    async def refine(
        self,
        summary: Dict[str, Any],
        aux_data: Dict[str, Any],
        session: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Запускает детерминированные проходы улучшения."""
        from ..config import get_settings
        
        settings = get_settings()
        epsilon = settings.quality_score_epsilon

        if aux_data.get("small_session_info"):
            logger.info("⚠️ Малый объём сессии — IterativeRefiner пропущен")
            self.last_iterations = 0
            self.iteration_history = []
            return summary

        improved = copy.deepcopy(summary)
        self.iteration_history = []
        self.last_iterations = 0

        baseline_score = improved.get("quality", {}).get("score", 0.0)
        logger.info(
            "🔄 Детерминированный IterativeRefiner: стартовый балл %.1f",
            baseline_score,
        )

        previous_score = baseline_score
        best_summary = copy.deepcopy(improved)
        best_score = baseline_score
        best_iteration = 0
        pass_count = min(
            self.max_iterations,
            getattr(self.summarizer, "STRUCTURAL_PASS_COUNT", self.max_iterations),
        )

        # Добавляем дополнительные итерации для низкокачественных саммаризаций
        if baseline_score < 70.0:
            pass_count = min(pass_count + 2, 7)  # До 7 итераций для плохих саммаризаций
            logger.info(
                f"🔧 Низкое качество ({baseline_score:.1f}) - увеличиваем итерации до {pass_count}"
            )

        consecutive_no_change = 0
        max_consecutive_no_change = 2

        for iteration in range(1, pass_count + 1):
            self.last_iterations = iteration
            improved, pass_info = self.summarizer._run_structural_pass(
                improved, aux_data, session, iteration
            )

            new_score = improved.get("quality", {}).get("score", previous_score)
            delta = new_score - previous_score
            changed = pass_info.get("changed", False)

            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "score_before": previous_score,
                    "score_after": new_score,
                    "delta": delta,
                    "changed": changed,
                }
            )

            logger.info(
                "   Итерация %d завершена: %.1f → %.1f (Δ%.1f)",
                iteration,
                previous_score,
                new_score,
                delta,
            )

            # Используем относительное сравнение с epsilon из конфига
            # Относительное сравнение: abs(a - b) <= epsilon * max(abs(a), abs(b), 1.0)
            score_diff = abs(new_score - best_score)
            max_score = max(abs(new_score), abs(best_score), 1.0)
            is_approximately_equal = score_diff <= epsilon * max_score
            
            if new_score > best_score + epsilon * max_score or (
                is_approximately_equal and changed
            ):
                best_summary = copy.deepcopy(improved)
                best_score = new_score
                best_iteration = iteration

            # Отслеживаем отсутствие изменений
            if not changed and abs(delta) < 0.1:
                consecutive_no_change += 1
                logger.warning(
                    f"⚠️ Итерация {iteration} не внесла изменений (подряд: {consecutive_no_change})"
                )
            else:
                consecutive_no_change = 0

            # Проверяем условия остановки
            should_stop, reason = self._should_stop_iteration(
                new_score=new_score,
                target_score=self.target_score,
                changed=changed,
                delta=delta,
                iteration=iteration,
                pass_count=pass_count,
                consecutive_no_change=consecutive_no_change,
                max_consecutive_no_change=max_consecutive_no_change,
            )

            # Логируем причину остановки или продолжения
            if should_stop:
                if reason == "target_score_reached":
                    logger.info(
                        "✅ Целевой балл %.1f достигнут после %d итераций",
                        self.target_score,
                        iteration,
                    )
                elif reason == "consecutive_no_change":
                    logger.warning(
                        f"🛑 Остановка: {consecutive_no_change} итераций подряд без изменений"
                    )
                elif reason == "small_delta_stop":
                    logger.info(
                        "⚠️ Прирост < 2 баллов (%.1f). Дальнейшие итерации остановлены.",
                        delta,
                    )
                elif reason == "no_changes":
                    logger.info("⚠️ Изменений не обнаружено, завершаем улучшение.")
                previous_score = new_score
                break
            else:
                if reason == "small_delta_continue":
                    logger.info(
                        "⚠️ Прирост < 2 баллов (%.1f). Продолжаем со следующей фазой.",
                        delta,
                    )
                elif reason == "continue" and not changed:
                    logger.debug(
                        "Фаза %d не изменила структуру, переходим к следующей",
                        iteration,
                    )
                previous_score = new_score
                continue

        else:
            previous_score = improved.get("quality", {}).get("score", baseline_score)

        final_summary = best_summary
        final_score = best_score
        logger.info(
            "🔚 IterativeRefiner завершён: финальный балл %.1f, итераций %d (лучший результат на итерации %d)",
            final_score,
            self.last_iterations,
            best_iteration,
        )

        details = final_summary.setdefault("quality", {}).setdefault("details", {})
        details["refinement_history"] = self.iteration_history
        details["best_iteration"] = best_iteration
        details["best_score"] = final_score

        return final_summary
