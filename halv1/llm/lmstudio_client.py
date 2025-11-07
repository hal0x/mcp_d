"""Client for interacting with LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests  # type: ignore[import-untyped]

from metrics import LLM_LATENCY, LLM_TOKENS_INFLIGHT, ERRORS

logger = logging.getLogger(__name__)


import threading


class LMStudioClient:
    """Minimal client for LM Studio's OpenAI-like endpoints.

    By default uses the Chat Completions endpoint at ``/v1/chat/completions``.
    """

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 1234,
        api_key: Optional[str] = None,
        use_chat: bool = True,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        num_ctx: Optional[int] = None,
        num_keep: Optional[int] = None,
        max_concurrency: int = 1,
    ) -> None:
        self.model = model
        self.base_url = f"http://{host}:{port}/v1"
        self.api_key = api_key
        self.use_chat = use_chat
        # Default generation parameters
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_max_tokens = max_tokens
        # Reasonable safe stops to avoid fenced blocks and sentinel markers
        self.default_stop = stop or ["```", "INCOMPLETE", "COMPLETE"]
        self.default_seed = seed
        self.default_num_ctx = num_ctx
        self.default_num_keep = num_keep
        # Serialize access to LM Studio to reduce channel/stream errors
        self._sem = threading.BoundedSemaphore(max_concurrency if max_concurrency > 0 else 1)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str) -> str:
        """Generate a reply using LM Studio.

        Prefers Chat Completions; falls back to Completions if ``use_chat`` is False.
        """
        t0 = time.perf_counter()
        LLM_TOKENS_INFLIGHT.labels("in").inc()
        
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.use_chat:
                        url = f"{self.base_url}/chat/completions"
                        payload: Dict[str, Any] = {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are HAL, a helpful AI assistant. "
                                        "Follow the user's instructions carefully and provide accurate responses. "
                                        "Use the format requested by the user."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                        }
                        # Attach default sampling/format constraints if provided
                        if self.default_temperature is not None:
                            payload["temperature"] = self.default_temperature
                        if self.default_top_p is not None:
                            payload["top_p"] = self.default_top_p
                        if self.default_max_tokens is not None:
                            payload["max_tokens"] = self.default_max_tokens
                        if self.default_seed is not None:
                            payload["seed"] = self.default_seed
                        if self.default_stop:
                            payload["stop"] = self.default_stop
                        if self.default_num_ctx is not None:
                            payload["num_ctx"] = self.default_num_ctx
                        if self.default_num_keep is not None:
                            # Keep small to avoid "tokens to keep > context length" errors
                            payload["num_keep"] = self.default_num_keep
                        logger.debug(
                            "Sending chat prompt to LM Studio (attempt %d/%d): %s",
                            attempt + 1,
                            max_retries,
                            prompt,
                        )
                        print(f"LLM request payload: {payload}")
                        with self._sem:
                            resp = requests.post(
                                url, json=payload, headers=self._headers(), timeout=1200
                            )
                        resp.raise_for_status()
                        data = resp.json()
                        logger.debug(f"LM Studio response data: {data}")
                        choices: List[Dict[str, Any]] = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            content = choices[0]["message"].get("content", "")
                            logger.debug(f"LM Studio response content: '{content}'")
                            return str(content)
                        logger.error(f"LM Studio chat response missing choices or message. Data: {data}")
                        raise RuntimeError(
                            "LM Studio chat response missing choices or message"
                        )
                    else:
                        url = f"{self.base_url}/completions"
                        payload = {"model": self.model, "prompt": prompt, "stream": False}
                        if self.default_temperature is not None:
                            payload["temperature"] = self.default_temperature
                        if self.default_top_p is not None:
                            payload["top_p"] = self.default_top_p
                        if self.default_max_tokens is not None:
                            payload["max_tokens"] = self.default_max_tokens
                        if self.default_seed is not None:
                            payload["seed"] = self.default_seed
                        if self.default_stop:
                            payload["stop"] = self.default_stop
                        if self.default_num_ctx is not None:
                            payload["num_ctx"] = self.default_num_ctx
                        if self.default_num_keep is not None:
                            payload["num_keep"] = self.default_num_keep
                        logger.debug(
                            "Sending completion prompt to LM Studio (attempt %d/%d): %s",
                            attempt + 1,
                            max_retries,
                            prompt,
                        )
                        with self._sem:
                            resp = requests.post(
                                url, json=payload, headers=self._headers(), timeout=1200
                            )
                        resp.raise_for_status()
                        data = resp.json()
                        logger.debug(f"LM Studio completion response data: {data}")
                        choices = data.get("choices", [])
                        if choices and "text" in choices[0]:
                            content = choices[0].get("text", "")
                            logger.debug(f"LM Studio completion response content: '{content}'")
                            return str(content)
                        logger.error(
                            f"LM Studio completion response missing choices or text. Data: {data}"
                        )
                        raise RuntimeError(
                            "LM Studio completion response missing choices or text"
                        )
                except Exception as exc:  # pragma: no cover - network/runtime errors
                    response_text = ""
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        response_text = exc.response.text
                    if attempt < max_retries - 1:
                        logger.warning(
                            "LM Studio request failed (attempt %d/%d): %s%s, retrying...",
                            attempt + 1,
                            max_retries,
                            exc,
                            f", response: {response_text}" if response_text else "",
                        )
                        # simple exponential backoff to reduce channel churn
                        time.sleep(min(1.0 * (2**attempt), 4.0))
                        continue
                    logger.error(
                        "LM Studio request failed after %d attempts: %s%s",
                        max_retries,
                        exc,
                        f", response: {response_text}" if response_text else "",
                    )
                    raise RuntimeError(
                        f"LM Studio request failed after {max_retries} attempts: {exc}: {response_text}"
                    )
            raise RuntimeError("LM Studio request failed without a response")
        except Exception as e:
            ERRORS.labels(component="llm", etype=type(e).__name__).inc()
            raise
        finally:
            LLM_TOKENS_INFLIGHT.labels("in").dec()
            LLM_LATENCY.labels(model=self.model, phase="gen").observe(time.perf_counter() - t0)

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield reply chunks using LM Studio's streaming API."""
        if self.use_chat:
            url = f"{self.base_url}/chat/completions"
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are HAL, a helpful AI assistant. "
                            "Follow the user's instructions carefully and provide accurate responses. "
                            "Use the format requested by the user."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            }
            if self.default_temperature is not None:
                payload["temperature"] = self.default_temperature
            if self.default_top_p is not None:
                payload["top_p"] = self.default_top_p
            if self.default_max_tokens is not None:
                payload["max_tokens"] = self.default_max_tokens
            if self.default_seed is not None:
                payload["seed"] = self.default_seed
            if self.default_stop:
                payload["stop"] = self.default_stop
            if self.default_num_ctx is not None:
                payload["num_ctx"] = self.default_num_ctx
            if self.default_num_keep is not None:
                payload["num_keep"] = self.default_num_keep
        else:
            url = f"{self.base_url}/completions"
            payload = {"model": self.model, "prompt": prompt, "stream": True}
            if self.default_temperature is not None:
                payload["temperature"] = self.default_temperature
            if self.default_top_p is not None:
                payload["top_p"] = self.default_top_p
            if self.default_max_tokens is not None:
                payload["max_tokens"] = self.default_max_tokens
            if self.default_seed is not None:
                payload["seed"] = self.default_seed
            if self.default_stop:
                payload["stop"] = self.default_stop
            if self.default_num_ctx is not None:
                payload["num_ctx"] = self.default_num_ctx
            if self.default_num_keep is not None:
                payload["num_keep"] = self.default_num_keep
        try:
            with self._sem:
                resp_ctx = requests.post(
                    url, json=payload, headers=self._headers(), stream=True, timeout=1200
                )
            with resp_ctx as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.strip() == b"data: [DONE]":
                        break
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                        except Exception:
                            continue
                        choices = data.get("choices", [])
                        if choices:
                            if self.use_chat:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                            else:
                                content = choices[0].get("text")
                            if content:
                                yield content
        except Exception as exc:  # pragma: no cover - network/runtime errors
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                logger.error(
                    "LM Studio stream failed: %s, response: %s", exc, exc.response.text
                )
            else:
                logger.error("LM Studio stream failed: %s", exc)
            # Fallback: return a non-streamed single chunk to preserve UX
            try:
                text = self.generate(prompt)
                if text:
                    yield text
            except Exception:
                return
