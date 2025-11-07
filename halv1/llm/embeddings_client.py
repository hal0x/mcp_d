"""Embeddings clients for LM Studio/Ollama-compatible APIs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Tuple, cast

import aiohttp

DEFAULT_MODEL = "dengcao/Qwen3-Embedding-4B:Q5_K_M"


class AsyncEmbeddingsClient:
    """Lightweight async client for the `/v1/embeddings` endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 1234,
        api_key: str | None = None,
        timeout_seconds: int = 60,  # Увеличено для стабильности
        provider: str = "lmstudio",  # lmstudio, ollama, or auto
        max_retries: int = 3,  # Количество попыток ретрая
        retry_delays: List[float] = None,  # Задержки между попытками
        num_ctx: int = 32000,  # Максимальный контекст для эмбеддингов
        num_batch: int = 1024,  # Размер батча для обработки
        keep_alive: str = "600s",  # Время жизни модели
    ) -> None:
        self.base_url = f"http://{host}:{port}/v1"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [0.5, 1.0, 2.0]  # Экспоненциальная задержка
        self.num_ctx = num_ctx
        self.num_batch = num_batch
        self.keep_alive = keep_alive
        self._fallback_cache: Dict[str, List[float]] = {}  # Кеш для fallback embeddings
        self._server_available: bool = True  # Флаг доступности сервера
        self.logger = logging.getLogger(__name__)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health_check(self) -> bool:
        """Проверка доступности сервера эмбеддингов."""
        try:
            # Определяем URL для health check
            if self.provider == "ollama":
                url = f"http://{self.base_url.split('//')[1].split(':')[0]}:{self.base_url.split(':')[-1].split('/')[0]}/api/tags"
            else:
                url = f"{self.base_url}/models"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self._headers()) as resp:
                    return resp.status == 200
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False

    async def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> Tuple[List[List[float]], int]:
        """Выполняет POST запрос с ретраями и экспоненциальной задержкой."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_ts = time.perf_counter()
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=self._headers()) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                
                elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
                
                # Обрабатываем ответ в зависимости от провайдера
                if self.provider == "ollama":
                    # Ollama возвращает embeddings напрямую в массиве
                    embeddings = [cast(List[float], data.get("embedding", []))]
                else:
                    # LM Studio возвращает в формате OpenAI API
                    embeddings = [
                        cast(List[float], d.get("embedding", []))
                        for d in data.get("data", [])
                    ]
                
                # Успешный запрос - сбрасываем флаг недоступности
                self._server_available = True
                return embeddings, elapsed_ms
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Embeddings attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                # Если это не последняя попытка, ждём перед повтором
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    self.logger.debug(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        
        # Все попытки исчерпаны
        self.logger.error(f"All {self.max_retries} attempts failed. Last error: {last_exception}")
        self._server_available = False
        return [], 0

    async def post_embeddings(
        self, payload: Dict[str, Any]
    ) -> Tuple[List[List[float]], int]:
        """Post ``payload`` and return ``(vectors, elapsed_ms)``."""

        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return ([], 0)

        # Определяем URL в зависимости от провайдера
        if self.provider == "ollama":
            # Для Ollama используем правильный URL
            host_port = self.base_url.replace("http://", "").replace("/v1", "")
            url = f"http://{host_port}/api/embeddings"
        else:
            url = f"{self.base_url}/embeddings"

        # Используем новый метод с ретраями
        return await self._post_with_retry(url, payload)

    def _create_fallback_embedding(self, text: str) -> List[float]:
        """Create a simple fallback embedding when the server is unavailable."""
        # Создаем детерминированный embedding на основе хеша текста
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        # Конвертируем хеш в вектор фиксированной длины (1024 измерения для mxbai-embed-large)
        embedding = []
        for i in range(0, len(text_hash), 2):
            # Берем по 2 символа хеша и конвертируем в число
            hex_pair = text_hash[i : i + 2]
            value = int(hex_pair, 16) / 255.0  # Нормализуем к [0, 1]
            embedding.append(value)

        # Дополняем до 1024 измерений, повторяя паттерн
        while len(embedding) < 1024:
            embedding.extend(embedding[: min(1024 - len(embedding), len(embedding))])

        return embedding[:1024]

    async def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text``."""

        # Проверяем кеш fallback embeddings
        if text in self._fallback_cache:
            return self._fallback_cache[text]

        # Если сервер недоступен, проверяем его доступность
        if not self._server_available:
            self.logger.debug("Server marked as unavailable, performing health check...")
            if await self.health_check():
                self.logger.info("Server is back online")
                self._server_available = True
            else:
                self.logger.warning("Server still unavailable, using fallback")
                fallback_embedding = self._create_fallback_embedding(text)
                self._fallback_cache[text] = fallback_embedding
                return fallback_embedding

        # Для Ollama используем prompt, для LM Studio - input
        if self.provider == "ollama":
            payload = {
                "model": self.model, 
                "prompt": text,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_batch": self.num_batch,
                    "keep_alive": self.keep_alive
                }
            }
        else:
            payload = {
                "model": self.model, 
                "input": text,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_batch": self.num_batch,
                    "keep_alive": self.keep_alive
                }
            }
        embeddings, _ = await self.post_embeddings(payload)
        if embeddings:
            # Кешируем полученный embedding, чтобы повторные вызовы
            # не обращались к серверу повторно
            self._fallback_cache[text] = embeddings[0]
            return embeddings[0]
        else:
            # Сервер недоступен, переключаемся на fallback режим
            self._server_available = False
            fallback_embedding = self._create_fallback_embedding(text)
            self._fallback_cache[text] = fallback_embedding
            return fallback_embedding

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for ``texts`` via a single request."""

        # Проверяем кеш для всех текстов
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._fallback_cache:
                cached_embeddings.append(self._fallback_cache[text])
            else:
                cached_embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Если все тексты в кеше, возвращаем их
        if not uncached_texts:
            return cached_embeddings

        # Если сервер недоступен, проверяем его доступность
        if not self._server_available:
            self.logger.debug("Server marked as unavailable, performing health check...")
            if await self.health_check():
                self.logger.info("Server is back online")
                self._server_available = True
            else:
                self.logger.warning("Server still unavailable, using fallback")
                for i, text in enumerate(uncached_texts):
                    original_index = uncached_indices[i]
                    fallback_embedding = self._create_fallback_embedding(text)
                    self._fallback_cache[text] = fallback_embedding
                    cached_embeddings[original_index] = fallback_embedding
                return cached_embeddings

        # Для Ollama используем prompt, для LM Studio - input
        if self.provider == "ollama":
            payload = {
                "model": self.model, 
                "prompt": uncached_texts,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_batch": self.num_batch,
                    "keep_alive": self.keep_alive
                }
            }
        else:
            payload = {
                "model": self.model, 
                "input": uncached_texts,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_batch": self.num_batch,
                    "keep_alive": self.keep_alive
                }
            }
        embeddings, _ = await self.post_embeddings(payload)

        if embeddings:
            # Заполняем кеш и результат
            for i, embedding in enumerate(embeddings):
                text = uncached_texts[i]
                original_index = uncached_indices[i]
                self._fallback_cache[text] = embedding
                cached_embeddings[original_index] = embedding
        else:
            # Сервер недоступен, переключаемся на fallback режим
            self._server_available = False
            for i, text in enumerate(uncached_texts):
                original_index = uncached_indices[i]
                fallback_embedding = self._create_fallback_embedding(text)
                self._fallback_cache[text] = fallback_embedding
                cached_embeddings[original_index] = fallback_embedding

        return cached_embeddings


class EmbeddingsClient:
    """Synchronous adapter over :class:`AsyncEmbeddingsClient`."""

    def __init__(
        self,
        *args: Any,
        async_client: AsyncEmbeddingsClient | None = None,
        **kwargs: Any,
    ) -> None:
        self._async = async_client or AsyncEmbeddingsClient(*args, **kwargs)

    def embed(self, text: str) -> List[float]:
        """Return an embedding for ``text``."""
        try:
            # Пытаемся получить текущий event loop
            loop = asyncio.get_running_loop()
            # Если мы в event loop, используем run_coroutine_threadsafe
            return asyncio.run_coroutine_threadsafe(
                self._async.embed(text), loop
            ).result(timeout=30)
        except RuntimeError:
            # Если нет запущенного event loop, используем asyncio.run
            return asyncio.run(self._async.embed(text))

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for multiple ``texts``."""
        try:
            # Пытаемся получить текущий event loop
            loop = asyncio.get_running_loop()
            # Если мы в event loop, используем run_coroutine_threadsafe
            return asyncio.run_coroutine_threadsafe(
                self._async.embed_many(texts), loop
            ).result(timeout=60)
        except RuntimeError:
            # Если нет запущенного event loop, используем asyncio.run
            return asyncio.run(self._async.embed_many(texts))
