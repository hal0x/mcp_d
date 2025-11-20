#!/usr/bin/env python3
"""Клиент для работы с LM Studio Server API для генерации эмбеддингов."""

import asyncio
import logging
import random
import re
from typing import Any, Dict, List, Optional

import aiohttp

from .constants import (
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_MAX_RETRY,
    HTTP_CONNECTOR_LIMIT,
    HTTP_CONNECTOR_LIMIT_PER_HOST,
)

logger = logging.getLogger(__name__)


class LMStudioEmbeddingClient:
    """
    Клиент для генерации эмбеддингов через LM Studio Server.
    
    model_name: только для эмбеддингов (/v1/embeddings)
    llm_model_name: для генерации текста (/v1/chat/completions)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-qwen3-embedding-0.6b",
        llm_model_name: Optional[str] = None,
        base_url: str = "http://127.0.0.1:1234",
        max_text_length: int = 16384,
    ):
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.base_url = base_url.rstrip("/")
        self.max_text_length = max_text_length
        self.session = None
        self._embedding_dimension: Optional[int] = None
        self._reasoning_models = {
            "gpt-oss-20b",
            "gpt-oss-20b:latest",
            "gpt-oss",
            "deepseek",
            "deepseek-reasoner",
        }

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход."""
        if self.session:
            await self.session.close()

    async def check_model_availability(self) -> bool:
        """Проверка доступности модели эмбеддингов через реальный запрос."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model.get("id", "") for model in data.get("data", [])]
                    if self.model_name not in models:
                        logger.warning(
                            f"Модель '{self.model_name}' не найдена в списке доступных моделей. "
                            f"Доступные модели: {', '.join(models[:5])}"
                        )
            
            test_payload = {"model": self.model_name, "input": "test"}
            async with self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and len(data.get("data", [])) > 0:
                        embedding = data["data"][0].get("embedding")
                        if embedding and isinstance(embedding, list):
                            if self._embedding_dimension is None:
                                self._embedding_dimension = len(embedding)
                            return True
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Модель '{self.model_name}' недоступна для эмбеддингов. "
                        f"HTTP {response.status}: {error_text[:200]}"
                    )
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут при проверке модели '{self.model_name}'")
            return False
        except Exception as e:
            logger.warning(f"Ошибка при проверке модели '{self.model_name}': {e}")
            return False

    async def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для одного текста."""
        embeddings = await self.generate_embeddings([text])
        if embeddings and self._embedding_dimension is None:
            self._embedding_dimension = len(embeddings[0])
        return embeddings[0] if embeddings else [0.0] * (self._embedding_dimension or 1024)

    async def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов."""
        if not texts:
            return []

        if not await self.check_model_availability():
            logger.error("LM Studio Server недоступен или модель не найдена")
            default_dim = self._embedding_dimension or 1024
            return [[0.0] * default_dim] * len(texts)

        if len(texts) == 1:
            text_preview = texts[0][:30] + "..." if len(texts[0]) > 30 else texts[0]
            logger.debug(f"🔤 Создание эмбеддинга: {text_preview}")
            embedding = await self._process_single_text_async(texts[0], 0, 1)
            return [embedding] if embedding else [[0.0] * (self._embedding_dimension or 1024)]

        logger.info(f"🔤 Создание эмбеддингов для {len(texts)} текстов батчами по {batch_size}...")

        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i, batch))

        all_embeddings = []
        default_dim = self._embedding_dimension or 1024

        for batch_idx, (start_idx, batch_texts) in enumerate(batches):
            try:
                logger.debug(f"🔤 Обработка батча {batch_idx + 1}/{len(batches)} ({len(batch_texts)} текстов)")
                
                batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    if self._embedding_dimension is None and batch_embeddings[0]:
                        self._embedding_dimension = len(batch_embeddings[0])
                else:
                    logger.warning(f"Батч {batch_idx + 1} не удался, создаем пустые эмбеддинги")
                    all_embeddings.extend([[0.0] * default_dim] * len(batch_texts))
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке батча {batch_idx + 1}: {e}")
                all_embeddings.extend([[0.0] * default_dim] * len(batch_texts))

        return all_embeddings

    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для батча текстов."""
        if not texts:
            return []
        
        processed_texts = []
        for text in texts:
            if len(text) > self.max_text_length:
                logger.warning(
                    f"Текст превышает лимит ({len(text)} > {self.max_text_length}), обрезаем"
                )
                text = text[:self.max_text_length]
            processed_texts.append(text)
        
        payload = {"model": self.model_name, "input": processed_texts}
        
        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=HTTP_CONNECTOR_LIMIT,
                limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True,
                force_close=True,
                ssl=False,
            )
            timeout = aiohttp.ClientTimeout(
                total=DEFAULT_TIMEOUT_MAX_RETRY,
                connect=DEFAULT_TIMEOUT_CONNECT,
                sock_read=DEFAULT_TIMEOUT_MAX_RETRY,
                sock_connect=DEFAULT_TIMEOUT_CONNECT,
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Connection": "close"},
            )
        
        default_dim = self._embedding_dimension or 1024
        for attempt in range(3):
            try:
                timeout_seconds = 60 + (attempt * 30)
                
                if attempt > 0:
                    logger.debug(f"Повторная попытка батч-запроса к LM Studio ({attempt + 1}/3)")
                
                async with asyncio.timeout(timeout_seconds):
                    async with self.session.post(
                        f"{self.base_url}/v1/embeddings", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "data" in data and isinstance(data["data"], list):
                                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                                embeddings = [item.get("embedding") for item in sorted_data]
                                
                                if len(embeddings) == len(processed_texts) and all(emb for emb in embeddings):
                                    if self._embedding_dimension is None and embeddings[0]:
                                        self._embedding_dimension = len(embeddings[0])
                                    logger.info(f"✅ Получено {len(embeddings)} эмбеддингов батчем")
                                    return embeddings
                                else:
                                    logger.warning(
                                        f"Получено {len(embeddings)} эмбеддингов вместо {len(processed_texts)}"
                                    )
                                    result = []
                                    for i, emb in enumerate(embeddings):
                                        if emb and isinstance(emb, list):
                                            result.append(emb)
                                        else:
                                            result.append([0.0] * default_dim)
                                    while len(result) < len(processed_texts):
                                        result.append([0.0] * default_dim)
                                    return result
                            else:
                                logger.error("LM Studio вернул неожиданный формат данных для батча")
                                if attempt < 2:
                                    await asyncio.sleep(2 + attempt)
                                    continue
                                return [[0.0] * default_dim] * len(processed_texts)
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"Ошибка API LM Studio при батч-запросе: {response.status} - {error_text[:200]}"
                            )
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt)
                                continue
                            return [[0.0] * default_dim] * len(processed_texts)
            except asyncio.TimeoutError:
                logger.error(
                    f"Таймаут батч-запроса к LM Studio (попытка {attempt + 1}/3)"
                )
                if attempt < 2:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return [[0.0] * default_dim] * len(processed_texts)
            except aiohttp.ClientError as e:
                error_str = str(e)
                is_connection_reset = "Connection reset" in error_str or "Errno 54" in error_str
                
                if is_connection_reset:
                    logger.warning(
                        f"Соединение с LM Studio разорвано при батч-запросе (попытка {attempt + 1}/3). "
                        f"Ошибка: {e}"
                    )
                    if attempt < 2:
                        try:
                            if self.session:
                                await self.session.close()
                        except Exception:
                            pass
                        connector = aiohttp.TCPConnector(
                            limit=HTTP_CONNECTOR_LIMIT,
                            limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                            ttl_dns_cache=300,
                            use_dns_cache=True,
                            enable_cleanup_closed=True,
                            force_close=True,
                            ssl=False,
                        )
                        timeout = aiohttp.ClientTimeout(
                            total=DEFAULT_TIMEOUT_MAX_RETRY,
                            connect=DEFAULT_TIMEOUT_CONNECT,
                            sock_read=DEFAULT_TIMEOUT_MAX_RETRY,
                            sock_connect=DEFAULT_TIMEOUT_CONNECT,
                        )
                        self.session = aiohttp.ClientSession(
                            connector=connector,
                            timeout=timeout,
                            headers={"Connection": "close"},
                        )
                        delay = (5 * (2 ** attempt)) + random.uniform(0, 2)
                        logger.debug(f"Ожидание {delay:.2f} секунд перед повторной попыткой батч-запроса...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    logger.error(
                        f"Ошибка соединения с LM Studio при батч-запросе (попытка {attempt + 1}/3): {e}"
                    )
                    if attempt < 2:
                        delay = (2 ** attempt) * 2 + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                return [[0.0] * default_dim] * len(processed_texts)
            except Exception as e:
                logger.error(f"Неожиданная ошибка при батч-запросе к LM Studio: {e}")
                if attempt < 2:
                    delay = (2 ** attempt) * 2 + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return [[0.0] * default_dim] * len(processed_texts)
        
        return [[0.0] * default_dim] * len(processed_texts)

    async def _process_single_text_async(
        self, text: str, index: int, total: int
    ) -> List[float]:
        """Асинхронная обработка одного текста с разбиением на части при необходимости."""
        try:
            text_chunks = self._split_text_into_chunks(text)

            if len(text_chunks) == 1:
                embedding = await self._generate_single_embedding(text)
                if embedding:
                    return embedding
                else:
                    logger.warning(
                        f"Не удалось получить эмбеддинг для текста: {text[:50]}..."
                    )
                    default_dim = self._embedding_dimension or 1024
                    return [0.0] * default_dim
            else:
                chunk_embeddings = []
                for j, chunk in enumerate(text_chunks):
                    if len(text_chunks) > 3 and j == 0:
                        logger.debug(f"Обработка длинного текста по частям ({len(text_chunks)} частей)")
                    chunk_embedding = await self._generate_single_embedding(chunk)
                    if chunk_embedding:
                        chunk_embeddings.append(chunk_embedding)
                    else:
                        logger.warning(f"Не удалось получить эмбеддинг для части {j+1}")
                        default_dim = self._embedding_dimension or 1024
                        chunk_embeddings.append([0.0] * default_dim)

                if chunk_embeddings:
                    averaged_embedding = self._average_embeddings(chunk_embeddings)
                    if len(chunk_embeddings) > 3:
                        logger.debug(
                            f"Усреднен эмбеддинг из {len(chunk_embeddings)} частей"
                        )
                    return averaged_embedding
                else:
                    default_dim = self._embedding_dimension or 1024
                    return [0.0] * default_dim

        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга: {e}")
            default_dim = self._embedding_dimension or 1024
            return [0.0] * default_dim

    def _split_text_into_chunks(self, text: str, max_length: int = None) -> List[str]:
        """Разбивка текста на части для обработки длинных текстов."""
        if max_length is None:
            max_length = self.max_text_length

        estimated_tokens = len(text) // 1.2
        max_tokens = max_length // 3.5

        if estimated_tokens <= max_tokens:
            return [text]

        chunks = []
        start = 0
        chunk_size_chars = max_tokens * 4

        while start < len(text):
            end = start + chunk_size_chars

            if end >= len(text):
                chunks.append(text[start:])
                break

            last_space = text.rfind(" ", start, end)
            if last_space > start + chunk_size_chars * 0.7:
                chunks.append(text[start:last_space])
                start = last_space + 1
            else:
                chunks.append(text[start:end])
                start = end

        if len(chunks) > 1:
            chunk_tokens = [len(chunk) // 3.5 for chunk in chunks]
            logger.debug(
                f"Длинный текст разбит на {len(chunks)} частей "
                f"(~{estimated_tokens} токенов -> ~{sum(chunk_tokens)} токенов в чанках)"
            )

        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Усреднение нескольких эмбеддингов в один."""
        if not embeddings:
            default_dim = self._embedding_dimension or 1024
            return [0.0] * default_dim

        if len(embeddings) == 1:
            return embeddings[0]

        dimension = len(embeddings[0])
        averaged = []

        for i in range(dimension):
            sum_val = sum(emb[i] for emb in embeddings)
            averaged.append(sum_val / len(embeddings))

        return averaged

    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Обрезка текста до максимальной длины."""
        if max_length is None:
            max_length = self.max_text_length

        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            result = truncated[:last_space]
        else:
            result = truncated

        logger.warning(
            f"Текст обрезан с {len(text)} до {len(result)} символов"
        )
        return result

    async def _generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Генерация эмбеддинга для одного текста."""
        if len(text) > self.max_text_length:
            logger.warning(
                f"Текст превышает лимит ({len(text)} > {self.max_text_length}), обрезаем"
            )
            text = text[: self.max_text_length]

        payload = {"model": self.model_name, "input": text}

        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=HTTP_CONNECTOR_LIMIT,
                limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True,
                force_close=True,
                ssl=False,
            )
            timeout = aiohttp.ClientTimeout(
                total=DEFAULT_TIMEOUT_MAX_RETRY,
                connect=DEFAULT_TIMEOUT_CONNECT,
                sock_read=DEFAULT_TIMEOUT_MAX_RETRY,
                sock_connect=DEFAULT_TIMEOUT_CONNECT,
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Connection": "close"},
            )

        for attempt in range(3):
            try:
                timeout_seconds = 60 + (attempt * 30)

                if attempt > 0:
                    logger.debug(f"Повторная попытка к LM Studio ({attempt + 1}/3)")

                async with asyncio.timeout(timeout_seconds):
                    async with self.session.post(
                        f"{self.base_url}/v1/embeddings", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "data" in data and len(data["data"]) > 0:
                                embedding = data["data"][0].get("embedding")
                                if embedding:
                                    if self._embedding_dimension is None:
                                        self._embedding_dimension = len(embedding)
                                    return embedding
                                else:
                                    logger.error("LM Studio вернул пустой embedding")
                                    if attempt < 2:
                                        await asyncio.sleep(2 + attempt)
                                        continue
                                    return None
                            else:
                                logger.error("LM Studio вернул неожиданный формат данных")
                                if attempt < 2:
                                    await asyncio.sleep(2 + attempt)
                                    continue
                                return None
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"Ошибка API LM Studio: {response.status} - {error_text}"
                            )
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt)
                                continue
                            return None
            except asyncio.TimeoutError:
                logger.error(
                    f"Таймаут запроса к LM Studio (попытка {attempt + 1}/3)"
                )
                if attempt < 2:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return None
            except aiohttp.ClientError as e:
                error_str = str(e)
                is_connection_reset = "Connection reset" in error_str or "Errno 54" in error_str
                
                if is_connection_reset:
                    logger.warning(
                        f"Соединение с LM Studio разорвано сервером (попытка {attempt + 1}/3). "
                        f"Возможно, сервер перегружен или перезапускается. "
                        f"Ошибка: {e}"
                    )
                    if attempt < 2:
                        try:
                            if self.session:
                                await self.session.close()
                        except Exception:
                            pass
                        # Создаем новую сессию
                        connector = aiohttp.TCPConnector(
                            limit=HTTP_CONNECTOR_LIMIT,
                            limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                            ttl_dns_cache=300,
                            use_dns_cache=True,
                            enable_cleanup_closed=True,
                            force_close=True,
                            ssl=False,
                        )
                        timeout = aiohttp.ClientTimeout(
                            total=DEFAULT_TIMEOUT_MAX_RETRY,
                            connect=DEFAULT_TIMEOUT_CONNECT,
                            sock_read=DEFAULT_TIMEOUT_MAX_RETRY,
                            sock_connect=DEFAULT_TIMEOUT_CONNECT,
                        )
                        self.session = aiohttp.ClientSession(
                            connector=connector,
                            timeout=timeout,
                            headers={"Connection": "close"},
                        )
                        delay = (5 * (2 ** attempt)) + random.uniform(0, 2)
                        logger.debug(f"Ожидание {delay:.2f} секунд перед повторной попыткой...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    logger.error(
                        f"Ошибка соединения с LM Studio (попытка {attempt + 1}/3): {e}"
                    )
                    if attempt < 2:
                        delay = (2 ** attempt) * 2 + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                return None
            except aiohttp.InvalidURL as e:
                logger.error(
                    f"Некорректный URL в запросе к LM Studio (попытка {attempt + 1}/3): {e}"
                )
                return None
            except aiohttp.ServerTimeoutError as e:
                logger.error(f"Таймаут сервера LM Studio (попытка {attempt + 1}/3): {e}")
                if attempt < 2:
                    delay = (2 ** attempt) * 3 + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка при запросе к LM Studio: {e}")
                if attempt < 2:
                    delay = (2 ** attempt) * 2 + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return None

        return None

    async def test_connection(self) -> Dict[str, Any]:
        """Тестирование подключения к LM Studio Server."""
        result = {
            "lmstudio_available": False,
            "model_available": False,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "error": None,
        }

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            try:
                async with self.session.get(
                    f"{self.base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result["lmstudio_available"] = True
                        data = await response.json()
                        models = data.get("data", [])
                        result["available_models"] = [m.get("id", "") for m in models]
                    else:
                        result["error"] = f"HTTP {response.status}: {await response.text()}"
            except asyncio.TimeoutError:
                result["error"] = "Таймаут при подключении к LM Studio Server"
                return result
            except Exception as e:
                result["error"] = f"Ошибка подключения: {str(e)}"
                return result

            if result["lmstudio_available"]:
                result["model_available"] = await self.check_model_availability()
                if not result["model_available"]:
                    if not result.get("error"):
                        result["error"] = (
                            f"Модель '{self.model_name}' недоступна для эмбеддингов. "
                            f"Убедитесь, что модель загружена в LM Studio и доступна через /v1/embeddings endpoint."
                        )

        except Exception as e:
            logger.error(f"Ошибка при тестировании подключения: {e}")
            result["error"] = str(e)

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Оценка количества токенов в тексте."""
        return len(text) // 1.2

    def _split_prompt(self, prompt: str, max_prompt_tokens: int = 31000) -> List[str]:
        """Разбивка промпта на части при превышении лимита токенов."""
        estimated_tokens = self._estimate_tokens(prompt)

        if estimated_tokens <= max_prompt_tokens:
            return [prompt]

        logger.warning(
            f"⚠️  Промпт слишком длинный ({estimated_tokens} токенов), разбиваем на части (лимит {max_prompt_tokens})"
        )

        conversation_markers = [
            "ИСХОДНЫЙ РАЗГОВОР:",
            "Разговор:",
            "Сообщения:",
            "conversation_text:",
            "conversation:",
        ]

        system_part = ""
        conversation_part = ""

        for marker in conversation_markers:
            if marker in prompt:
                parts = prompt.split(marker, 1)
                if len(parts) == 2:
                    system_part = parts[0] + marker
                    conversation_part = parts[1]
                    break

        if not conversation_part:
            max_chars = int(max_prompt_tokens * 1.2)
            chunks = []
            for i in range(0, len(prompt), max_chars):
                chunks.append(prompt[i : i + max_chars])
            return chunks

        max_conversation_tokens = (
            max_prompt_tokens - self._estimate_tokens(system_part) - 100
        )
        max_conversation_chars = int(max_conversation_tokens * 1.2)

        conversation_chunks = []
        for i in range(0, len(conversation_part), max_conversation_chars):
            conversation_chunks.append(
                conversation_part[i : i + max_conversation_chars]
            )

        prompts = []
        for i, chunk in enumerate(conversation_chunks):
            if i == 0:
                chunk_prompt = system_part + chunk
            else:
                chunk_prompt = f"{system_part}\n(ПРОДОЛЖЕНИЕ РАЗГОВОРА - часть {i+1}/{len(conversation_chunks)})\n{chunk}"
            prompts.append(chunk_prompt)

        logger.info(f"📝 Промпт разбит на {len(prompts)} частей")
        return prompts

    async def generate_summary(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 131072,
        top_p: float = 0.93,
        presence_penalty: float = 0.05,
        max_prompt_tokens: int = 100000,
    ) -> str:
        """Генерация саммаризации через LLM с автоматической разбивкой длинных промптов."""
        prompt_parts = self._split_prompt(prompt, max_prompt_tokens)

        if len(prompt_parts) == 1:
            return await self._generate_single_summary(
                prompt_parts[0], temperature, max_tokens, top_p, presence_penalty
            )
        else:
            logger.info(
                f"🔄 Генерация саммаризации по частям ({len(prompt_parts)} частей)"
            )
            summaries = []

            for i, part_prompt in enumerate(prompt_parts):
                logger.info(f"📝 Обработка части {i+1}/{len(prompt_parts)}")
                part_summary = await self._generate_single_summary(
                    part_prompt, temperature, max_tokens, top_p, presence_penalty
                )
                summaries.append(part_summary)

            combined = "\n\n".join(summaries)
            logger.info(f"✅ Объединены {len(summaries)} частей саммаризации")
            return combined

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Проверка, является ли модель reasoning-моделью."""
        if not model_name:
            return False
        model_lower = model_name.lower()
        for reasoning_model in self._reasoning_models:
            if reasoning_model.lower() in model_lower:
                return True
        return False

    async def _generate_single_summary(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        presence_penalty: float,
    ) -> str:
        """Генерация саммаризации для одного промпта через LM Studio Server."""
        llm_model = self.llm_model_name
        if not llm_model:
            error_msg = (
                f"ОШИБКА: Для генерации текста нужна LLM модель, а не модель эмбеддингов. "
                f"Текущая модель эмбеддингов: '{self.model_name}'. "
                f"Установите llm_model_name при создании LMStudioEmbeddingClient или используйте Ollama для генерации текста."
            )
            logger.error(error_msg)
            return f"Ошибка: {error_msg}"
        
        payload = {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Follow the user's instructions carefully and provide accurate responses.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": False,
        }

        for attempt in range(2):
            try:
                if not self.session:
                    connector = aiohttp.TCPConnector(
                        limit=HTTP_CONNECTOR_LIMIT,
                        limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        enable_cleanup_closed=True,
                        force_close=True,
                        ssl=False,
                    )
                    timeout = aiohttp.ClientTimeout(
                        total=1800,  # 30 минут для LLM (увеличено для генерации длинных саммари)
                        connect=DEFAULT_TIMEOUT_CONNECT,
                        sock_read=1800,  # Таймаут чтения сокета
                    )
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"Connection": "keep-alive"},  # Keep-alive для длительных запросов
                    )
                
                # Для очень длинных промптов используем streaming, чтобы предотвратить отключение
                # Streaming позволяет клиенту получать данные во время генерации
                use_streaming = len(prompt) > 50000  # ~12.5k токенов
                
                if use_streaming:
                    payload["stream"] = True
                    logger.debug(f"Используется streaming для длинного промпта ({len(prompt)} символов)")
                    
                    async with self.session.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    ) as response:
                        if response.status == 200:
                            content_parts = []
                            async for line in response.content:
                                if line:
                                    try:
                                        line_text = line.decode('utf-8').strip()
                                        if line_text.startswith('data: '):
                                            json_str = line_text[6:]  # Убираем 'data: '
                                            if json_str == '[DONE]':
                                                break
                                            chunk_data = await asyncio.to_thread(lambda: __import__('json').loads(json_str))
                                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                                delta = chunk_data['choices'][0].get('delta', {})
                                                if 'content' in delta:
                                                    content_parts.append(delta['content'])
                                    except Exception as e:
                                        logger.debug(f"Ошибка парсинга streaming chunk: {e}")
                                        continue
                            
                            content = ''.join(content_parts).strip()
                            if content:
                                return content
                            else:
                                logger.warning("Streaming вернул пустой контент, пробуем обычный запрос")
                                payload["stream"] = False
                
                # Обычный запрос (без streaming или fallback)
                if not use_streaming or payload.get("stream") is False:
                    payload["stream"] = False
                    async with self.session.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                message = choice.get("message", {})
                                
                                content = message.get("content", "").strip()
                                reasoning = message.get("reasoning", "")
                                finish_reason = choice.get("finish_reason", "")
                                
                                if not content and reasoning:
                                    logger.warning(
                                        f"Получен пустой контент от reasoning-модели '{llm_model}'. "
                                        f"Используем reasoning как fallback. "
                                        f"Finish reason: {finish_reason}. "
                                        f"Возможно, max_tokens ({max_tokens}) недостаточно для генерации content."
                                    )
                                    json_match = re.search(r"\{.*\}", reasoning, re.DOTALL)
                                    if json_match:
                                        content = json_match.group(0)
                                        logger.info("Извлечен JSON из reasoning")
                                    else:
                                        content = reasoning.strip()
                                
                                if finish_reason == "length" and not content:
                                    logger.error(
                                        f"Достигнут лимит токенов для модели '{llm_model}'. "
                                        f"Текущий max_tokens: {max_tokens}. "
                                        f"Все токены ушли на reasoning. Увеличьте max_tokens."
                                    )
                                
                                return content if content else "Ошибка: пустой ответ от модели"
                            else:
                                logger.error("LM Studio вернул неожиданный формат данных")
                                return "Ошибка генерации: неожиданный формат ответа"
                        else:
                            error_text = await response.text()
                            logger.error(f"Ошибка API LM Studio: {response.status} - {error_text}")
                            if attempt < 1:
                                await asyncio.sleep(3 + random.uniform(0, 1))
                                continue
                            return f"Ошибка генерации: HTTP {response.status}"
            except aiohttp.ClientError as e:
                error_str = str(e)
                is_connection_reset = "Connection reset" in error_str or "Errno 54" in error_str
                
                if is_connection_reset:
                    logger.warning(
                        f"Соединение с LM Studio разорвано при генерации саммари (попытка {attempt + 1}/2). "
                        f"Ошибка: {e}"
                    )
                    if attempt < 1:
                        try:
                            if self.session:
                                await self.session.close()
                        except Exception:
                            pass
                        # Создаем новую сессию
                        connector = aiohttp.TCPConnector(
                            limit=HTTP_CONNECTOR_LIMIT,
                            limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST,
                            ttl_dns_cache=300,
                            use_dns_cache=True,
                            enable_cleanup_closed=True,
                            force_close=True,
                            ssl=False,
                        )
                        timeout = aiohttp.ClientTimeout(
                            total=1800,  # 30 минут для LLM (увеличено для генерации длинных саммари)
                            connect=DEFAULT_TIMEOUT_CONNECT,
                            sock_read=1800,  # Таймаут чтения сокета
                        )
                        self.session = aiohttp.ClientSession(
                            connector=connector,
                            timeout=timeout,
                            # Убираем "Connection": "close" для длительных запросов
                        )
                        delay = 5 + random.uniform(0, 2)
                        logger.debug(f"Ожидание {delay:.2f} секунд перед повторной попыткой генерации...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    logger.error(f"Ошибка соединения с LM Studio при генерации саммари: {e}")
                    if attempt < 1:
                        delay = 3 + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                return f"Ошибка генерации: {str(e)}"
            except aiohttp.InvalidURL as e:
                logger.error(f"Некорректный URL в запросе к LM Studio: {e}")
                return "Ошибка генерации: некорректный URL"
            except asyncio.TimeoutError:
                logger.error(f"Таймаут при генерации саммари (попытка {attempt + 1}/2)")
                if attempt < 1:
                    delay = 5 + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return "Ошибка генерации: таймаут запроса"
            except Exception as e:
                logger.error(f"Ошибка при генерации саммаризации: {e}")
                if attempt < 1:
                    delay = 3 + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return f"Ошибка генерации: {str(e)}"
        
        return "Ошибка генерации: не удалось выполнить запрос после повторных попыток"

