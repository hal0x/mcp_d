#!/usr/bin/env python3
"""Контекстно-осведомленный LLM клиент с переиспользованием контекста."""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from threading import Lock
from typing import Any, Optional, Protocol, cast, runtime_checkable

from .base_client import ConversationHistory, LLMClient, OptimizedLLMClient

logger = logging.getLogger(__name__)


@runtime_checkable
class ContextAwareLLMClient(Protocol):
    """LLM клиент с поддержкой переиспользования контекста."""

    def generate(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> tuple[str, ConversationHistory]:
        """Генерация с сохранением истории диалога."""

    def generate_simple(self, prompt: str) -> str:
        """Простая генерация без истории (обратная совместимость)."""

    def stream(self, prompt: str, history: Optional[ConversationHistory] = None) -> Any:
        """Стриминг с сохранением истории диалога."""

    def stream_simple(self, prompt: str) -> Any:
        """Простой стриминг без истории (обратная совместимость)."""


class ContextAwareWrapper:
    """Обертка для LLM клиента с поддержкой сохранения истории."""

    def __init__(self, client: LLMClient):
        self.client = client
        self._history: Optional[ConversationHistory] = None
        self._lock = Lock()

        try:
            signature = inspect.signature(client.generate)
            params = signature.parameters
            self._supports_history = "history" in params or "context" in params
        except (TypeError, ValueError):
            self._supports_history = False

        if self._supports_history:
            logger.info("✅ LLM клиент поддерживает передачу истории диалога")
        else:
            logger.info(
                "⚠️ LLM клиент не поддерживает историю, используем режим совместимости"
            )

    def generate(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> tuple[str, ConversationHistory]:
        """Генерация ответа с учетом истории."""

        base_history = history if history is not None else self._history

        if self._supports_history:
            try:
                optimized_client = cast(OptimizedLLMClient, self.client)
                response, new_history = optimized_client.generate(prompt, base_history)
                with self._lock:
                    self._history = new_history
                return response, new_history
            except TypeError:
                logger.warning("Клиент не поддерживает расширенный API, используем fallback")

        response = self.client.generate(prompt)
        new_history = self._append_history(base_history, prompt, response)
        with self._lock:
            self._history = new_history
        return response, new_history

    def generate_simple(self, prompt: str) -> str:
        """Простая генерация без сохраненной истории (обратная совместимость)."""
        response, _ = self.generate(prompt)
        return response

    def stream(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> Any:
        """Стриминг ответа с учетом истории."""

        base_history = history if history is not None else self._history

        if self._supports_history and hasattr(self.client, "stream"):
            try:
                optimized_client = cast(OptimizedLLMClient, self.client)
                for chunk, new_history in optimized_client.stream(prompt, base_history):
                    with self._lock:
                        self._history = new_history
                    yield chunk, new_history
                return
            except TypeError:
                logger.warning(
                    "Клиент не поддерживает расширенный stream API, используем fallback"
                )

        if hasattr(self.client, "stream"):
            user_turn = {"role": "user", "content": prompt}
            assistant_turn = {"role": "assistant", "content": ""}
            accumulated = [dict(turn) for turn in (base_history or [])]
            accumulated.append(user_turn)

            for chunk in self.client.stream(prompt):
                assistant_turn["content"] += chunk
                current_history = accumulated + [assistant_turn.copy()]
                yield chunk, current_history

            final_history = accumulated + [assistant_turn]
            with self._lock:
                self._history = final_history
        else:
            response = self.client.generate(prompt)
            new_history = self._append_history(base_history, prompt, response)
            with self._lock:
                self._history = new_history
            yield response, new_history

    def stream_simple(self, prompt: str) -> Any:
        """Простой стриминг без сохраненной истории (обратная совместимость)."""
        for chunk, _ in self.stream(prompt):
            yield chunk

    def get_context(self) -> Optional[ConversationHistory]:
        """Получить текущую историю (совместимость со старым API)."""
        with self._lock:
            return deepcopy(self._history) if self._history is not None else None

    def clear_context(self) -> None:
        """Очистить сохраненную историю."""
        with self._lock:
            self._history = None

    def set_context(self, history: ConversationHistory) -> None:
        """Установить историю (совместимость со старым API)."""
        with self._lock:
            self._history = [dict(turn) for turn in history]

    def _append_history(
        self,
        history: Optional[ConversationHistory],
        prompt: str,
        response: str,
    ) -> ConversationHistory:
        base = [dict(turn) for turn in (history or [])]
        base.append({"role": "user", "content": prompt})
        base.append({"role": "assistant", "content": response})
        return base


class ContextAwareCodeGenerator:
    """Генератор кода с поддержкой сохранения истории."""

    def __init__(self, client: ContextAwareWrapper):
        self.client = client
        self._history: Optional[ConversationHistory] = None

    def generate(
        self, description: str, max_attempts: int = 3, *, error_reason: str = ""
    ) -> str:
        """Генерация кода с учетом истории диалога."""
        from llm.prompts import make_code_prompt

        prompt = make_code_prompt(description, error_reason)
        last_reason = error_reason

        for attempt in range(max_attempts):
            logger.info(f"Попытка генерации кода {attempt + 1}/{max_attempts}")

            # Используем историю для ускорения повторных попыток
            response, new_history = self.client.generate(prompt, self._history)
            self._history = new_history

            # Остальная логика остается той же
            code = self._extract_code_from_markdown(response)
            code = self._clean_unicode_chars(code)

            try:
                import ast

                ast.parse(code)
                logger.info(f"Генерация кода успешна на попытке {attempt + 1}")
                return code
            except SyntaxError as exc:
                reason = f"syntax error: {exc}"
                logger.warning(
                    f"Синтаксическая ошибка на попытке {attempt + 1}: {reason}"
                )
                last_reason = reason

                if attempt == max_attempts - 1:
                    break
                prompt = make_code_prompt(description, last_reason)

        raise Exception(
            f"Не удалось сгенерировать корректный код после {max_attempts} попыток: {last_reason}"
        )

    def _extract_code_from_markdown(self, text: str) -> str:
        """Извлечение кода из markdown."""
        # Простая реализация - можно улучшить
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        return text.strip()

    def _clean_unicode_chars(self, code: str) -> str:
        """Очистка Unicode символов."""
        # Простая реализация - можно улучшить
        replacements = {
            "→": "->",
            "–": "-",
            "—": "-",
        }

        for old, new in replacements.items():
            code = code.replace(old, new)

        return code


class ContextAwareSearchClient:
    """Поисковый клиент с поддержкой сохранения истории."""

    def __init__(self, client: ContextAwareWrapper):
        self.client = client
        self._history: Optional[ConversationHistory] = None

    def summarize(self, text: str) -> str:
        """Суммаризация с учетом истории."""
        from llm.prompts import make_web_summary_prompt

        prompt = make_web_summary_prompt(text[:4000])

        try:
            response, new_history = self.client.generate(prompt, self._history)
            self._history = new_history
            return response
        except Exception as exc:
            logger.error(f"Ошибка суммаризации: {exc}")
            return text

    def get_context(self) -> Optional[ConversationHistory]:
        """Получить текущую историю (совместимость со старым API)."""
        return deepcopy(self._history) if self._history is not None else None

    def clear_context(self) -> None:
        """Очистить историю."""
        self._history = None
