"""LangChain адаптеры для работы с эмбеддингами и LLM."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    LangChainEmbeddings = None  # type: ignore
    BaseChatModel = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore

from ...config import get_settings

logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter:
    """Адаптер LangChain Embeddings.
    
    Обертка над LangChain Embeddings, которая предоставляет интерфейс
    для работы с эмбеддингами через LangChain.
    """

    def __init__(
        self,
        embeddings: LangChainEmbeddings,
        *,
        timeout: float = 10.0,
    ) -> None:
        """Инициализация адаптера.
        
        Args:
            embeddings: Экземпляр LangChain Embeddings
            timeout: Таймаут для операций (не используется напрямую)
        """
        if LangChainEmbeddings is None:
            raise ImportError(
                "LangChain не установлен. Установите: pip install langchain langchain-community langchain-openai"
            )
        self.embeddings = embeddings
        self.timeout = timeout
        self._dimension: Optional[int] = None
        self._base_url: Optional[str] = None
        self._model_name: Optional[str] = None

    def available(self) -> bool:
        """Проверка доступности сервиса эмбеддингов."""
        return self.embeddings is not None

    @property
    def dimension(self) -> Optional[int]:
        """Размерность эмбеддингов."""
        if self._dimension is None:
            # Попытка определить размерность через тестовый запрос
            try:
                test_embedding = self.embed("test")
                if test_embedding:
                    self._dimension = len(test_embedding)
            except Exception as e:
                logger.warning(f"Не удалось определить размерность эмбеддингов: {e}")
        return self._dimension

    def embed(self, text: str) -> Optional[List[float]]:
        """Генерация эмбеддинга для одного текста.
        
        Args:
            text: Текст для генерации эмбеддинга
            
        Returns:
            Список чисел (вектор эмбеддинга) или None при ошибке
        """
        if not text or not text.strip():
            return None
        
        try:
            # LangChain embeddings.embed_query синхронный
            vector = self.embeddings.embed_query(text.strip())
            if self._dimension is None and vector:
                self._dimension = len(vector)
            return vector
        except Exception as exc:
            logger.warning(f"LangChain embedding error: {exc}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Генерация эмбеддингов для батча текстов.
        
        Args:
            texts: Список текстов для генерации эмбеддингов
            
        Returns:
            Список векторов эмбеддингов (может содержать None при ошибках)
        """
        if not texts:
            return []
        
        # Очищаем и фильтруем тексты
        processed_texts = [text.strip() for text in texts if text.strip()]
        if not processed_texts:
            return [None] * len(texts)
        
        try:
            # LangChain embeddings.embed_documents для батча
            embeddings_list = self.embeddings.embed_documents(processed_texts)
            
            # Сохраняем размерность при первом успешном запросе
            if self._dimension is None and embeddings_list and embeddings_list[0]:
                self._dimension = len(embeddings_list[0])
            
            # Создаем результат с None для пустых текстов
            result = []
            processed_idx = 0
            for text in texts:
                if text.strip():
                    emb = embeddings_list[processed_idx] if processed_idx < len(embeddings_list) else None
                    result.append(emb)
                    processed_idx += 1
                else:
                    result.append(None)
            
            logger.info(f"✅ Получено {len([e for e in result if e])} эмбеддингов из {len(texts)} текстов")
            return result
        except Exception as exc:
            logger.warning(f"LangChain embedding batch error: {exc}")
            return [None] * len(texts)

    def close(self) -> None:
        """Закрытие соединений."""
        # LangChain embeddings обычно не требуют явного закрытия
        pass


class LangChainLLMAdapter:
    """Адаптер LangChain LLM.
    
    Обертка над LangChain ChatModel, которая предоставляет методы generate_summary
    и другие для работы с LLM через LangChain.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Инициализация адаптера.
        
        Args:
            llm: Экземпляр LangChain ChatModel
            model_name: Имя модели эмбеддингов
            llm_model_name: Имя LLM модели
            base_url: Базовый URL
        """
        if BaseChatModel is None:
            raise ImportError(
                "LangChain не установлен. Установите: pip install langchain langchain-community langchain-openai"
            )
        self.llm = llm
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.base_url = base_url
        self.session = None  # Для async context manager

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход."""
        pass

    async def generate_summary(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 131072,
        top_p: float = 0.93,
        presence_penalty: float = 0.05,
        max_prompt_tokens: int = 100000,
    ) -> str:
        """Генерация саммаризации через LLM.
        
        Args:
            prompt: Промпт для генерации
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            top_p: Top-p параметр
            presence_penalty: Presence penalty (не поддерживается напрямую в LangChain)
            max_prompt_tokens: Максимальное количество токенов в промпте
            
        Returns:
            Сгенерированный текст
        """
        try:
            # Устанавливаем параметры модели, если поддерживается
            if hasattr(self.llm, "temperature"):
                self.llm.temperature = temperature
            if hasattr(self.llm, "max_tokens"):
                self.llm.max_tokens = max_tokens
            if hasattr(self.llm, "top_p"):
                self.llm.top_p = top_p
            
            # Разбиваем длинные промпты (если нужно)
            prompt_parts = self._split_prompt(prompt, max_prompt_tokens)
            
            if len(prompt_parts) == 1:
                return await self._generate_single_summary(prompt_parts[0])
            else:
                logger.info(f"🔄 Генерация саммаризации по частям ({len(prompt_parts)} частей)")
                summaries = []
                for i, part_prompt in enumerate(prompt_parts):
                    logger.info(f"📝 Обработка части {i+1}/{len(prompt_parts)}")
                    part_summary = await self._generate_single_summary(part_prompt)
                    summaries.append(part_summary)
                combined = "\n\n".join(summaries)
                logger.info(f"✅ Объединены {len(summaries)} частей саммаризации")
                return combined
        except Exception as e:
            logger.error(f"Ошибка генерации саммаризации через LangChain: {e}")
            return f"Ошибка генерации: {str(e)}"

    async def _generate_single_summary(self, prompt: str) -> str:
        """Генерация саммаризации для одного промпта."""
        try:
            # Создаем сообщения для LangChain
            messages = [
                SystemMessage(content="You are a helpful AI assistant. Follow the user's instructions carefully and provide accurate responses."),
                HumanMessage(content=prompt),
            ]
            
            # Вызываем LLM (LangChain поддерживает async)
            response = await self.llm.ainvoke(messages)
            
            # Извлекаем контент из ответа
            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Ошибка при генерации саммаризации: {e}")
            return f"Ошибка генерации: {str(e)}"

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
        
        max_conversation_tokens = max_prompt_tokens - self._estimate_tokens(system_part) - 100
        max_conversation_chars = int(max_conversation_tokens * 1.2)
        
        conversation_chunks = []
        for i in range(0, len(conversation_part), max_conversation_chars):
            conversation_chunks.append(conversation_part[i : i + max_conversation_chars])
        
        prompts = []
        for i, chunk in enumerate(conversation_chunks):
            if i == 0:
                chunk_prompt = system_part + chunk
            else:
                chunk_prompt = f"{system_part}\n(ПРОДОЛЖЕНИЕ РАЗГОВОРА - часть {i+1}/{len(conversation_chunks)})\n{chunk}"
            prompts.append(chunk_prompt)
        
        logger.info(f"📝 Промпт разбит на {len(prompts)} частей")
        return prompts


def build_langchain_embeddings_from_env() -> Optional[LangChainEmbeddingAdapter]:
    """Создание LangChain Embeddings адаптера из настроек окружения.
    
    Returns:
        LangChainEmbeddingAdapter или None, если конфигурация не найдена
    """
    if OpenAIEmbeddings is None:
        logger.warning("LangChain не установлен, невозможно создать embeddings адаптер")
        return None
    
    settings = get_settings()
    
    # Priority 1: embeddings_url
    url = settings.get_embeddings_url()
    model_name = None
    
    if url:
        # Используем OpenAIEmbeddings для OpenAI-совместимых API
        # Определяем формат по URL
        if "/v1" in url or ":1234" in url:
            # LM Studio или OpenAI-совместимый API
            model_name = settings.lmstudio_model
            embeddings = OpenAIEmbeddings(
                model=model_name or "text-embedding-ada-002",
                openai_api_base=url,
                openai_api_key="not-needed",  # LM Studio не требует ключ
            )
        else:
            # text-embeddings-inference или другой формат
            # Для TEI может потребоваться кастомная интеграция
            logger.warning(f"TEI формат ({url}) требует кастомной интеграции, используем OpenAIEmbeddings")
            embeddings = OpenAIEmbeddings(
                openai_api_base=url,
                openai_api_key="not-needed",
            )
    else:
        # Priority 2: LM Studio variables
        host = settings.lmstudio_host
        port = str(settings.lmstudio_port)
        model_name = settings.lmstudio_model
        
        base_url = f"http://{host}:{port}/v1"
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key="not-needed",
        )
    
    adapter = LangChainEmbeddingAdapter(embeddings)
    adapter._base_url = url or f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
    adapter._model_name = model_name or settings.lmstudio_model
    
    # Warm-up для определения размерности
    test_vector = adapter.embed("warmup sentence for embeddings")
    if test_vector is None:
        logger.warning("LangChain embedding service не вернул вектор при warm-up")
        return None
    
    logger.info(
        f"LangChain embedding service initialized: URL={adapter._base_url}, "
        f"Model={adapter._model_name}, Dimension={adapter.dimension}"
    )
    return adapter


def build_langchain_llm_from_env() -> Optional[LangChainLLMAdapter]:
    """Создание LangChain LLM адаптера из настроек окружения.
    
    Returns:
        LangChainLLMAdapter или None, если конфигурация не найдена
    """
    if ChatOpenAI is None:
        logger.warning("LangChain не установлен, невозможно создать LLM адаптер")
        return None
    
    settings = get_settings()
    
    # Используем только LM Studio через LangChain
    if not settings.lmstudio_llm_model:
        logger.debug("LM Studio LLM модель не задана, LangChain адаптер не создан")
        return None
    
    # Используем LM Studio через ChatOpenAI
    base_url = f"http://{settings.lmstudio_host}:{settings.lmstudio_port}/v1"
    llm = ChatOpenAI(
        model=settings.lmstudio_llm_model,
        base_url=base_url,
        api_key="not-needed",
        temperature=0.3,
    )
    adapter = LangChainLLMAdapter(
        llm,
        model_name=settings.lmstudio_model,
        llm_model_name=settings.lmstudio_llm_model,
        base_url=base_url,
    )
    
    logger.info(f"LangChain LLM adapter initialized: Model={adapter.llm_model_name}")
    return adapter


def get_llm_client_factory() -> Optional[LangChainLLMAdapter]:
    """Фабричная функция для получения LangChain LLM клиента.
    
    Returns:
        LangChainLLMAdapter или None
    """
    try:
        adapter = build_langchain_llm_from_env()
        if adapter:
            logger.debug("Используется LangChain LLM адаптер")
            return adapter
        else:
            logger.error("LangChain LLM failed to initialize")
            return None
    except ImportError as e:
        logger.error(f"LangChain not available: {e}. Install: pip install langchain langchain-community langchain-openai")
        return None
    except Exception as e:
        logger.error(f"Error initializing LangChain LLM: {e}")
        return None

