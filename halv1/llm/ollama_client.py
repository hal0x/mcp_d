"""Client for interacting with a local Ollama model."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional

import requests  # type: ignore[import-untyped]

from metrics import LLM_LATENCY, LLM_TOKENS_INFLIGHT, ERRORS
from .base_client import ConversationHistory, ConversationMessage

logger = logging.getLogger(__name__)


class OllamaClient:
    """Minimal wrapper over the Ollama HTTP API with conversation history support.

    Parameters
    ----------
    model: str
        Name of the model registered in the local Ollama instance.
    host: str
        Host where the Ollama service is listening.
    port: int
        Port of the service. By default the official service uses ``11434``.
    keep_alive: str
        How long to keep the model loaded in memory (e.g., "5m", "30m", "1h").
    num_batch: int
        Batch size for processing tokens (higher = faster but more memory).
    num_ctx: int
        Context window size in tokens (default: 32000 for gemma3n:e4b-it-q8_0).
    """

    def __init__(
        self,
        model: str = "gemma3n:e4b-it-q8_0",
        host: str = "localhost",
        port: int = 11434,
        keep_alive: str = "5m",
        num_batch: int = 512,
        num_ctx: int = 32000
    ):
        self.model = model
        self.base_url = f"http://{host}:{port}"
        # Backward-compatibility attributes (kept for external callers/tests)
        self.url = f"{self.base_url}/api/chat"
        # Internal endpoints
        self._url_native_chat = self.url
        self._url_native_generate = f"{self.base_url}/api/generate"
        self._url_openai_chat = f"{self.base_url}/v1/chat/completions"
        self.keep_alive = keep_alive
        self.num_batch = num_batch
        self.num_ctx = num_ctx
        # Cached conversation history for lightweight reuse between calls
        self._history: ConversationHistory = []
        # Флаг готовности модели
        self._is_warmed_up = False

    async def warmup(self, test_prompt: str = "ping") -> bool:
        """Прогрев модели для ускорения последующих запросов.
        
        Parameters
        ----------
        test_prompt: str
            Простой тестовый промпт для прогрева модели.
            
        Returns
        -------
        bool
            True если прогрев успешен, False в противном случае.
        """
        try:
            logger.info(f"Warming up LLM model: {self.model}")
            start_time = time.perf_counter()
            
            # Выполняем простой запрос для прогрева
            response, _ = self.generate(test_prompt)
            
            elapsed = time.perf_counter() - start_time
            self._is_warmed_up = True
            
            logger.info(f"LLM warmup completed: {self.model} in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"LLM warmup failed: {e}")
            self._is_warmed_up = False
            return False

    def is_warmed_up(self) -> bool:
        """Проверка готовности модели."""
        return self._is_warmed_up

    async def health_check(self) -> bool:
        """Проверка доступности Ollama сервера."""
        try:
            # Проверяем доступность через API tags
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LLM health check failed: {e}")
            return False

    def generate(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> tuple[str, ConversationHistory]:
        """Generate a reply for *prompt* using the configured model.

        Parameters
        ----------
        prompt: str
            The input prompt to generate a response for.
        history: Optional[ConversationHistory]
            Prior dialog turns to prepend to the request.

        Returns
        -------
        tuple[str, ConversationHistory]
            Generated response and the updated history including the new turn.

        Any errors in the HTTP request are logged and re-raised. The method is
        synchronous to keep dependencies minimal; it can be adapted to
        ``aiohttp`` for async support.
        """
        t0 = time.perf_counter()
        LLM_TOKENS_INFLIGHT.labels("in").inc()
        
        try:
            active_history = self._clone_history(history)
            full_prompt = self._build_prompt(prompt, active_history)

            prompt_length = len(full_prompt.split())
            if prompt_length > self.num_ctx * 0.9:
                logger.warning(
                    "Prompt length (%d tokens) is close to context limit (%d). "
                    "Consider shortening the prompt to avoid truncation.",
                    prompt_length,
                    self.num_ctx,
                )

            chat_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": self._build_messages(prompt, active_history),
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_batch": self.num_batch,
                    "num_ctx": self.num_ctx,
                },
            }
            legacy_payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_batch": self.num_batch,
                    "num_ctx": self.num_ctx,
                },
            }
            openai_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": chat_payload["messages"],
                "stream": False,
            }

            logger.debug("Sending prompt to Ollama: %s", full_prompt)

            response = self._request_with_fallback(
                (
                    (self._url_native_chat, chat_payload),
                    (self._url_native_generate, legacy_payload),
                ),
                stream=False,
            )

            data = response.json()
            response.close()
            response_text = self._extract_response_text(data)

            new_history = self._append_history(active_history, prompt, response_text)
            self._history = new_history

            return response_text, new_history
        except Exception as e:
            ERRORS.labels(component="llm", etype=type(e).__name__).inc()
            raise
        finally:
            LLM_TOKENS_INFLIGHT.labels("in").dec()
            LLM_LATENCY.labels(model=self.model, phase="gen").observe(time.perf_counter() - t0)
    
    def generate_simple(self, prompt: str) -> str:
        """Generate a reply for *prompt* using the configured model (backward compatibility).
        
        This method maintains backward compatibility with existing code.
        For optimal performance, use generate() with history management.
        """
        response, _ = self.generate(prompt)
        return response

    def stream(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> Iterator[tuple[str, ConversationHistory]]:
        """Yield reply chunks for *prompt* using Ollama's streaming API."""

        active_history = self._clone_history(history)
        full_prompt = self._build_prompt(prompt, active_history)

        prompt_length = len(full_prompt.split())
        if prompt_length > self.num_ctx * 0.9:
            logger.warning(
                "Prompt length (%d tokens) is close to context limit (%d). "
                "Consider shortening the prompt to avoid truncation.",
                prompt_length,
                self.num_ctx,
            )

        chat_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, active_history),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_batch": self.num_batch,
                "num_ctx": self.num_ctx,
            },
        }
        legacy_payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_batch": self.num_batch,
                "num_ctx": self.num_ctx,
            },
        }
        openai_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_payload["messages"],
            "stream": True,
        }

        logger.debug("Streaming prompt to Ollama: %s", full_prompt)
        try:
            response = self._request_with_fallback(
                (
                    (self._url_native_chat, chat_payload),
                    (self._url_native_generate, legacy_payload),
                ),
                stream=True,
            )
            with response as resp:
                user_turn: ConversationMessage = {"role": "user", "content": prompt}
                assistant_turn: ConversationMessage = {"role": "assistant", "content": ""}
                base_history = active_history + [user_turn]

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    if data.get("done"):
                        continue

                    token = self._extract_stream_token(data)
                    if not token:
                        continue

                    assistant_turn["content"] += token
                    current_history = base_history + [assistant_turn.copy()]
                    yield token, current_history

                final_history = base_history + [assistant_turn]
                self._history = final_history
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Ollama stream failed: %s", exc)
            return
    
    def stream_simple(self, prompt: str) -> Iterator[str]:
        """Yield reply chunks for *prompt* using Ollama's streaming API (backward compatibility).
        
        This method maintains backward compatibility with existing code.
        For optimal performance, use stream() with history management.
        """
        for token, _ in self.stream(prompt):
            yield token

    def _clone_history(
        self, history: Optional[ConversationHistory]
    ) -> ConversationHistory:
        """Return a deep copy of provided history or the cached instance history."""

        if history is None:
            return deepcopy(self._history)
        return deepcopy(history)

    def _build_prompt(
        self, prompt: str, history: ConversationHistory
    ) -> str:
        """Compose prompt text by prepending conversation history."""

        if not history:
            return prompt

        parts = []
        for message in history:
            role = self._format_role_header(message.get("role", "user"))
            parts.append(f"{role}: {message.get('content', '')}")
        parts.append(f"USER: {prompt}")
        return "\n\n".join(parts)

    def _build_messages(
        self, prompt: str, history: ConversationHistory
    ) -> ConversationHistory:
        """Compose OpenAI-style messages payload using existing history."""

        messages: ConversationHistory = []
        for message in history:
            messages.append(
                {
                    "role": self._normalize_role(message.get("role", "user")),
                    "content": message.get("content", ""),
                }
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    def _append_history(
        self, history: ConversationHistory, prompt: str, response: str
    ) -> ConversationHistory:
        """Return a new history list with the latest turn appended."""

        updated = deepcopy(history)
        updated.append({"role": "user", "content": prompt})
        updated.append({"role": "assistant", "content": response})
        return updated

    @staticmethod
    def _normalize_role(role: str) -> str:
        role_lower = role.strip().lower()
        if role_lower in {"system", "user", "assistant"}:
            return role_lower
        return "user"

    @staticmethod
    def _format_role_header(role: str) -> str:
        mapping = {
            "system": "SYSTEM",
            "user": "USER",
            "assistant": "ASSISTANT",
        }
        normalized = role.strip().lower()
        return mapping.get(normalized, normalized.upper() or "USER")

    def _request_with_fallback(
        self,
        endpoints: tuple[tuple[str, Dict[str, Any]], ...],
        *,
        stream: bool,
    ) -> requests.Response:
        """Try a sequence of endpoints until one succeeds."""

        last_error: Exception | None = None
        for url, payload in endpoints:
            try:
                logger.debug(f"Trying endpoint: {url}")
                response = requests.post(url, json=payload, stream=stream)
                logger.debug(f"Response status: {response.status_code}")
            except Exception as exc:  # pragma: no cover - network errors
                logger.debug(f"Network error for {url}: {exc}")
                last_error = exc
                continue

            if response.status_code in {404, 405}:
                logger.debug(f"Endpoint {url} not available: {response.status_code}")
                response.close()
                continue

            try:
                response.raise_for_status()
                logger.debug(f"Success with endpoint: {url}")
                return response
            except Exception as exc:  # pragma: no cover - http errors
                logger.debug(f"HTTP error for {url}: {exc}")
                response.close()
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(f"Ollama request failed: {last_error}") from last_error
        raise RuntimeError("Ollama request failed: no available endpoints")

    @staticmethod
    def _extract_response_text(data: Dict[str, Any]) -> str:
        """Extract assistant text from Ollama or OpenAI-style responses."""

        if not isinstance(data, dict):
            return ""

        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", ""))

        if "response" in data:
            return str(data.get("response", ""))

        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", ""))
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return str(delta.get("content", ""))

        return ""

    @staticmethod
    def _extract_stream_token(data: Dict[str, Any]) -> str:
        """Extract a streaming token from multiple protocol formats."""

        if not isinstance(data, dict):
            return ""

        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", ""))

        if "response" in data:
            return str(data.get("response", ""))

        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return str(delta.get("content", ""))

        return ""
