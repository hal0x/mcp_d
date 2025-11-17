#!/usr/bin/env python3
"""
Клиент для работы с Ollama API для генерации эмбеддингов
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .constants import (
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_MAX_RETRY,
    HTTP_CONNECTOR_LIMIT,
    HTTP_CONNECTOR_LIMIT_PER_HOST,
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Клиент для генерации эмбеддингов через Ollama"""

    def __init__(
        self,
        model_name: str = "hf.co/lmstudio-community/Magistral-Small-2509-GGUF:Q4_K_M",
        llm_model_name: str = "hf.co/lmstudio-community/Magistral-Small-2509-GGUF:Q4_K_M",
        base_url: str = "http://localhost:11434",
        max_text_length: int = 16384,  # 4096 токенов * 4 символа/токен для безопасного лимита
        llm_thinking_level: str | None = None,
    ):
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.base_url = base_url
        self.max_text_length = max_text_length
        self.llm_thinking_level = llm_thinking_level
        self.session = None

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        if self.session:
            await self.session.close()

    async def check_model_availability(self) -> bool:
        """Проверка доступности модели"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.model_name in models
                return False
        except Exception as e:
            logger.error(f"Ошибка при проверке модели: {e}")
            return False

    async def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для одного текста"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * 5120

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов с поддержкой длинных текстов"""
        if not texts:
            return []

        # Проверяем доступность Ollama перед началом работы
        if not await self.check_model_availability():
            logger.error("Ollama недоступен или модель не найдена")
            return [[0.0] * 5120] * len(texts)

        embeddings = []

        # Обрабатываем тексты параллельно для лучшей производительности
        if len(texts) == 1:
            # Для одного текста логируем только если это не повторная обработка
            text_preview = texts[0][:30] + "..." if len(texts[0]) > 30 else texts[0]
            logger.debug(f"🔤 Создание эмбеддинга: {text_preview}")
        else:
            logger.info(f"🔤 Создание эмбеддингов для {len(texts)} текстов параллельно...")

        # Создаем задачи для параллельной обработки
        tasks = []
        for i, text in enumerate(texts):
            text_preview = text[:30] + "..." if len(text) > 30 else text
            # Логируем только каждое 10-е сообщение для экономии логов (только для множественных текстов)
            if len(texts) > 1 and ((i + 1) % 10 == 0 or i == 0):
                logger.info(
                    f"🔤 Подготовка эмбеддинга [{i+1}/{len(texts)}]: {text_preview}"
                )
            task = self._process_single_text_async(text, i, len(texts))
            tasks.append(task)

        # Выполняем задачи с ограничением на количество параллельных запросов
        # Ограничиваем до 10 параллельных запросов для лучшей производительности
        semaphore = asyncio.Semaphore(10)

        async def limited_task(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Обрабатываем результаты
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Ошибка при создании эмбеддинга {i+1}: {result}")
                embeddings.append([0.0] * 5120)
            else:
                embeddings.append(result)

        return embeddings

    async def _process_single_text_async(
        self, text: str, index: int, total: int
    ) -> List[float]:
        """Асинхронная обработка одного текста"""
        try:
            # Разбиваем длинные тексты на части
            text_chunks = self._split_text_into_chunks(text)

            if len(text_chunks) == 1:
                # Короткий текст - обрабатываем как обычно
                embedding = await self._generate_single_embedding(text)
                if embedding:
                    return embedding
                else:
                    logger.warning(
                        f"Не удалось получить эмбеддинг для текста: {text[:50]}..."
                    )
                    return [0.0] * 5120
            else:
                # Длинный текст - обрабатываем по частям и усредняем
                chunk_embeddings = []
                for j, chunk in enumerate(text_chunks):
                    # Логируем только для больших текстов (более 3 частей)
                    if len(text_chunks) > 3 and j == 0:
                        logger.debug(f"Обработка длинного текста по частям ({len(text_chunks)} частей)")
                    chunk_embedding = await self._generate_single_embedding(chunk)
                    if chunk_embedding:
                        chunk_embeddings.append(chunk_embedding)
                    else:
                        logger.warning(f"Не удалось получить эмбеддинг для части {j+1}")
                        chunk_embeddings.append([0.0] * 5120)

                # Усредняем эмбеддинги всех частей
                if chunk_embeddings:
                    averaged_embedding = self._average_embeddings(chunk_embeddings)
                    # Усреднение эмбеддингов (логируем только для больших групп)
                    if len(chunk_embeddings) > 3:
                        logger.debug(
                            f"Усреднен эмбеддинг из {len(chunk_embeddings)} частей"
                        )
                    return averaged_embedding
                else:
                    return [0.0] * 5120

        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга: {e}")
            return [0.0] * 5120

    def _split_text_into_chunks(self, text: str, max_length: int = None) -> List[str]:
        """Разбивка текста на части для обработки длинных текстов"""
        if max_length is None:
            max_length = self.max_text_length

        # Проверяем по токенам, а не по символам
        # Для Magistral-Small-2509 используем оценку: ~1.2 символа/токен
        estimated_tokens = len(text) // 1.2
        max_tokens = max_length // 3.5

        if estimated_tokens <= max_tokens:
            return [text]

        chunks = []
        start = 0
        chunk_size_chars = max_tokens * 4  # Размер чанка в символах

        while start < len(text):
            end = start + chunk_size_chars

            if end >= len(text):
                # Последний кусок
                chunks.append(text[start:])
                break

            # Ищем последний пробел в пределах chunk_size_chars
            last_space = text.rfind(" ", start, end)
            if (
                last_space > start + chunk_size_chars * 0.7
            ):  # Если пробел не слишком далеко
                chunks.append(text[start:last_space])
                start = last_space + 1  # Пропускаем пробел
            else:
                # Если пробел слишком далеко, обрезаем по символам
                chunks.append(text[start:end])
                start = end

        # Логируем информацию о размерах чанков
        if len(chunks) > 1:
            chunk_tokens = [len(chunk) // 3.5 for chunk in chunks]
            logger.debug(
                f"Длинный текст разбит на {len(chunks)} частей "
                f"(~{estimated_tokens} токенов -> ~{sum(chunk_tokens)} токенов в чанках)"
            )

        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Усреднение нескольких эмбеддингов в один"""
        if not embeddings:
            return [0.0] * 5120

        if len(embeddings) == 1:
            return embeddings[0]

        # Усредняем по каждому измерению
        dimension = len(embeddings[0])
        averaged = []

        for i in range(dimension):
            sum_val = sum(emb[i] for emb in embeddings)
            averaged.append(sum_val / len(embeddings))

        return averaged

    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Обрезка текста до максимальной длины для избежания превышения лимита токенов"""
        if max_length is None:
            max_length = self.max_text_length

        if len(text) <= max_length:
            return text

        # Обрезаем до последнего пробела, чтобы не разрывать слова
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:  # Если последний пробел не слишком далеко
            result = truncated[:last_space]
        else:
            result = truncated

        logger.warning(
            f"Текст обрезан с {len(text)} до {len(result)} символов для избежания превышения лимита токенов"
        )
        return result

    async def _generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Генерация эмбеддинга для одного текста"""
        # Дополнительная проверка длины текста
        if len(text) > self.max_text_length:
            logger.warning(
                f"Текст превышает лимит ({len(text)} > {self.max_text_length}), обрезаем"
            )
            text = text[: self.max_text_length]

        payload = {"model": self.model_name, "prompt": text}

        # Инициализируем сессию один раз перед циклом retry, если не существует
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
            # Используем базовый таймаут для сессии, конкретные таймауты будут в запросах
            timeout = aiohttp.ClientTimeout(
                total=DEFAULT_TIMEOUT_MAX_RETRY,  # Максимальный таймаут для retry
                connect=DEFAULT_TIMEOUT_CONNECT,
                sock_read=DEFAULT_TIMEOUT_MAX_RETRY,
                sock_connect=DEFAULT_TIMEOUT_CONNECT,
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Connection": "close"},
            )

        # Повторные попытки с разумными таймаутами
        for attempt in range(3):
            try:
                # Увеличенные таймауты для Qwen3-Embedding-4B: 60, 90, 120 секунд
                timeout_seconds = 60 + (attempt * 30)

                # Логируем только при повторных попытках
                if attempt > 0:
                    logger.debug(f"Повторная попытка к Ollama ({attempt + 1}/3)")

                # Добавляем таймаут для всего запроса
                async with asyncio.timeout(timeout_seconds):
                    async with self.session.post(
                        f"{self.base_url}/api/embeddings", json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("embedding")
                        else:
                            logger.error(
                                f"Ошибка API Ollama: {response.status}"
                            )
                            if attempt < 2:  # Не последняя попытка
                                await asyncio.sleep(2 + attempt)  # 2, 3 секунды
                                continue
                            return None

            except asyncio.TimeoutError:
                logger.error(
                    f"Таймаут при генерации эмбеддинга (попытка {attempt + 1}/3)"
                )
                if attempt < 2:  # Не последняя попытка
                    await asyncio.sleep(2**attempt)
                    continue
                return None
            except aiohttp.ClientError as e:
                logger.error(
                    f"Ошибка соединения с Ollama (попытка {attempt + 1}/3): {e}"
                )
                if attempt < 2:  # Не последняя попытка
                    await asyncio.sleep(5 + (attempt * 2))  # 5, 7, 9 секунд
                    continue
                return None
            except aiohttp.InvalidURL as e:
                logger.error(
                    f"Некорректный URL в запросе к Ollama (попытка {attempt + 1}/3): {e}"
                )
                # Для ошибок URL не повторяем попытки
                return None
            except aiohttp.ServerTimeoutError as e:
                logger.error(f"Таймаут сервера Ollama (попытка {attempt + 1}/3): {e}")
                if attempt < 2:  # Не последняя попытка
                    await asyncio.sleep(5 + (attempt * 2))
                    continue
                return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка при запросе к Ollama: {e}")
                if attempt < 2:  # Не последняя попытка
                    await asyncio.sleep(5 + (attempt * 2))
                    continue
                return None

    async def test_connection(self) -> Dict[str, Any]:
        """Тестирование подключения к Ollama"""
        result = {
            "ollama_available": False,
            "model_available": False,
            "llm_model_available": False,
            "model_name": self.model_name,
            "llm_model_name": self.llm_model_name,
            "base_url": self.base_url,
        }

        try:
            # Инициализируем сессию если не инициализирована
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Проверяем доступность Ollama
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    result["ollama_available"] = True
                    version_data = await response.json()
                    result["ollama_version"] = version_data.get("version", "unknown")

            # Проверяем доступность модели эмбеддингов
            if result["ollama_available"]:
                result["model_available"] = await self.check_model_availability()
                # Проверяем доступность LLM модели
                result[
                    "llm_model_available"
                ] = await self.check_llm_model_availability()

        except Exception as e:
            logger.error(f"Ошибка при тестировании подключения: {e}")
            result["error"] = str(e)

        return result

    async def check_llm_model_availability(self) -> bool:
        """Проверка доступности LLM модели"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.llm_model_name in models
                return False
        except Exception as e:
            logger.error(f"Ошибка при проверке LLM модели: {e}")
            return False

    def _estimate_tokens(self, text: str) -> int:
        """
        Примерная оценка количества токенов в тексте
        Для Magistral-Small-2509 используем оценку: ~1.2 символа = 1 токен
        """
        return len(text) // 1.2

    def _split_prompt(self, prompt: str, max_prompt_tokens: int = 31000) -> List[str]:
        """
        Разбивает промпт на части, если он превышает лимит токенов

        Args:
            prompt: Исходный промпт
            max_prompt_tokens: Максимальное количество токенов в промпте

        Returns:
            Список частей промпта
        """
        estimated_tokens = self._estimate_tokens(prompt)

        if estimated_tokens <= max_prompt_tokens:
            return [prompt]

        # Разбиваем промпт на части
        logger.warning(
            f"⚠️  Промпт слишком длинный ({estimated_tokens} токенов), разбиваем на части (лимит {max_prompt_tokens})"
        )

        # Пытаемся найти разделители для более умного разбиения
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
            # Если структура промпта другая, разбиваем по символам
            max_chars = max_prompt_tokens * 1.2
            chunks = []
            for i in range(0, len(prompt), max_chars):
                chunks.append(prompt[i : i + max_chars])
            return chunks

        # Разбиваем разговор на части
        max_conversation_tokens = (
            max_prompt_tokens - self._estimate_tokens(system_part) - 100
        )  # запас
        max_conversation_chars = max_conversation_tokens * 1.2

        conversation_chunks = []
        for i in range(0, len(conversation_part), max_conversation_chars):
            conversation_chunks.append(
                conversation_part[i : i + max_conversation_chars]
            )

        # Собираем промпты для каждой части
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
        max_tokens: int = 900,
        top_p: float = 0.93,
        presence_penalty: float = 0.05,
        max_prompt_tokens: int = 30000,
    ) -> str:
        """
        Генерация саммаризации через LLM с автоматической разбивкой длинных промптов

        Args:
            prompt: Промпт для генерации
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            top_p: Top-p параметр
            presence_penalty: Presence penalty
            max_prompt_tokens: Максимальное количество токенов в промпте (по умолчанию 30000, безопасный лимит для 32768 контекста)

        Returns:
            Сгенерированный текст
        """
        # Проверяем длину промпта и разбиваем при необходимости
        prompt_parts = self._split_prompt(prompt, max_prompt_tokens)

        if len(prompt_parts) == 1:
            # Обычная генерация
            return await self._generate_single_summary(
                prompt_parts[0], temperature, max_tokens, top_p, presence_penalty
            )
        else:
            # Генерация по частям и объединение результатов
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

            # Объединяем результаты
            combined = "\n\n".join(summaries)
            logger.info(f"✅ Объединены {len(summaries)} частей саммаризации")
            return combined

    async def _generate_single_summary(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        presence_penalty: float,
    ) -> str:
        """Генерация саммаризации для одного промпта"""
        payload = {
            "model": self.llm_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "num_ctx": 8192,  # Максимальный размер контекста для Gemma3n
            },
        }

        if self.llm_thinking_level:
            payload["thinking"] = {"level": self.llm_thinking_level}

        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 минут для LLM
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "").strip()
                    else:
                        logger.error(f"Ошибка API Ollama: {response.status}")
                        return f"Ошибка генерации: HTTP {response.status}"
        except aiohttp.InvalidURL as e:
            logger.error(f"Некорректный URL в запросе к Ollama: {e}")
            return "Ошибка генерации: некорректный URL"
        except Exception as e:
            logger.error(f"Ошибка при генерации саммаризации: {e}")
            return f"Ошибка генерации: {str(e)}"


# Синхронная версия для совместимости
class OllamaEmbeddingClientSync:
    """Синхронная версия клиента Ollama"""

    def __init__(
        self,
        model_name: str = "hf.co/lmstudio-community/Magistral-Small-2509-GGUF:Q4_K_M",
        base_url: str = "http://localhost:11434",
        max_text_length: int = 16384,  # 4096 токенов * 4 символа/токен для безопасного лимита
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_text_length = max_text_length

    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Обрезка текста до максимальной длины для избежания превышения лимита токенов"""
        if max_length is None:
            max_length = self.max_text_length

        if len(text) <= max_length:
            return text

        # Обрезаем до последнего пробела, чтобы не разрывать слова
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:  # Если последний пробел не слишком далеко
            result = truncated[:last_space]
        else:
            result = truncated

        logger.warning(
            f"Текст обрезан с {len(text)} до {len(result)} символов для избежания превышения лимита токенов"
        )
        return result

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Синхронная генерация эмбеддингов с поддержкой длинных текстов"""

        embeddings = []

        for text in texts:
            try:
                # Разбиваем длинные тексты на части
                text_chunks = self._split_text_into_chunks(text)

                if len(text_chunks) == 1:
                    # Короткий текст - обрабатываем как обычно
                    embedding = self._generate_single_embedding_sync(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        logger.warning(
                            f"Не удалось получить эмбеддинг для текста: {text[:50]}..."
                        )
                        embeddings.append([0.0] * 5120)
                else:
                    # Длинный текст - обрабатываем по частям и усредняем
                    chunk_embeddings = []
                    for j, chunk in enumerate(text_chunks):
                        logger.debug(
                            f"Обработка части {j+1}/{len(text_chunks)} (длина: {len(chunk)})"
                        )
                        chunk_embedding = self._generate_single_embedding_sync(chunk)
                        if chunk_embedding:
                            chunk_embeddings.append(chunk_embedding)
                        else:
                            logger.warning(
                                f"Не удалось получить эмбеддинг для части {j+1}"
                            )
                            chunk_embeddings.append([0.0] * 5120)

                    # Усредняем эмбеддинги всех частей
                    if chunk_embeddings:
                        averaged_embedding = self._average_embeddings(chunk_embeddings)
                        embeddings.append(averaged_embedding)
                        # Усреднение эмбеддингов (логируем только для больших групп)
                    if len(chunk_embeddings) > 3:
                        logger.debug(
                            f"Усреднен эмбеддинг из {len(chunk_embeddings)} частей"
                        )
                    else:
                        embeddings.append([0.0] * 5120)

            except Exception as e:
                logger.error(f"Ошибка при генерации эмбеддинга: {e}")
                embeddings.append([0.0] * 5120)

        return embeddings

    def _split_text_into_chunks(self, text: str, max_length: int = None) -> List[str]:
        """Разбивка текста на части для обработки длинных текстов"""
        if max_length is None:
            max_length = self.max_text_length

        # Проверяем по токенам, а не по символам
        # Для Magistral-Small-2509 используем оценку: ~1.2 символа/токен
        estimated_tokens = len(text) // 1.2
        max_tokens = max_length // 3.5

        if estimated_tokens <= max_tokens:
            return [text]

        chunks = []
        start = 0
        chunk_size_chars = max_tokens * 4  # Размер чанка в символах

        while start < len(text):
            end = start + chunk_size_chars

            if end >= len(text):
                # Последний кусок
                chunks.append(text[start:])
                break

            # Ищем последний пробел в пределах chunk_size_chars
            last_space = text.rfind(" ", start, end)
            if (
                last_space > start + chunk_size_chars * 0.7
            ):  # Если пробел не слишком далеко
                chunks.append(text[start:last_space])
                start = last_space + 1  # Пропускаем пробел
            else:
                # Если пробел слишком далеко, обрезаем по символам
                chunks.append(text[start:end])
                start = end

        # Логируем информацию о размерах чанков
        if len(chunks) > 1:
            chunk_tokens = [len(chunk) // 3.5 for chunk in chunks]
            logger.debug(
                f"Длинный текст разбит на {len(chunks)} частей "
                f"(~{estimated_tokens} токенов -> ~{sum(chunk_tokens)} токенов в чанках)"
            )

        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Усреднение нескольких эмбеддингов в один"""
        if not embeddings:
            return [0.0] * 5120

        if len(embeddings) == 1:
            return embeddings[0]

        # Усредняем по каждому измерению
        dimension = len(embeddings[0])
        averaged = []

        for i in range(dimension):
            sum_val = sum(emb[i] for emb in embeddings)
            averaged.append(sum_val / len(embeddings))

        return averaged

    def _generate_single_embedding_sync(self, text: str) -> Optional[List[float]]:
        """Синхронная генерация эмбеддинга для одного текста"""
        import requests

        # Дополнительная проверка длины текста
        if len(text) > self.max_text_length:
            logger.warning(
                f"Текст превышает лимит ({len(text)} > {self.max_text_length}), обрезаем"
            )
            text = text[: self.max_text_length]

        payload = {"model": self.model_name, "prompt": text}

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("embedding")
            else:
                logger.error(f"Ошибка API Ollama: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга: {e}")
            return None

    def test_connection(self) -> Dict[str, Any]:
        """Синхронное тестирование подключения"""
        import requests

        result = {
            "ollama_available": False,
            "model_available": False,
            "model_name": self.model_name,
            "base_url": self.base_url,
        }

        try:
            # Проверяем доступность Ollama
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            if response.status_code == 200:
                result["ollama_available"] = True
                version_data = response.json()
                result["ollama_version"] = version_data.get("version", "unknown")

            # Проверяем доступность модели
            if result["ollama_available"]:
                response = requests.get(f"{self.base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    result["model_available"] = self.model_name in models

        except Exception as e:
            logger.error(f"Ошибка при тестировании подключения: {e}")
            result["error"] = str(e)

        return result
