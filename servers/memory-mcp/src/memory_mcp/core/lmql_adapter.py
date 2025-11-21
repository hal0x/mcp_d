"""LMQL адаптер для структурированной генерации с LLM."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import lmql
    from lmql import F
except ImportError:
    lmql = None  # type: ignore
    F = None  # type: ignore

from ..config import get_settings

logger = logging.getLogger(__name__)


class LMQLAdapter:
    """Адаптер для работы с LMQL.

    Предоставляет методы для выполнения структурированных запросов к LLM
    с гарантированным форматом ответа.
    """

    def __init__(
        self,
        model: str,
        backend: str = "lmstudio",
        base_url: Optional[str] = None,
    ) -> None:
        """Инициализация LMQL адаптера.

        Args:
            model: Название модели для LMQL
            backend: Бэкенд для LMQL (игнорируется, всегда используется lmstudio)
            base_url: Базовый URL для API (для lmstudio)
        """
        if lmql is None:
            raise ImportError(
                "LMQL не установлен. Установите: pip install lmql>=0.9.0"
            )

        self.model = model
        self.backend = "lmstudio"  # Всегда используем lmstudio
        self.base_url = base_url

        # Всегда используем lmstudio с OpenAI-совместимым API
        self.model_identifier = f"openai/{model}"
        self.api_config = {"api_base": base_url} if base_url else {}

        logger.info(
            f"Инициализирован LMQLAdapter (модель: {self.model_identifier}, "
            f"бэкенд: lmstudio)"
        )

    def available(self) -> bool:
        """Проверка доступности LMQL."""
        return lmql is not None

    def _extract_result_value(self, result: Any) -> str:
        """Извлекает значение из результата LMQL.
        
        Args:
            result: Результат выполнения LMQL запроса (может быть списком или объектом)
            
        Returns:
            Строковое представление результата
        """
        # Если результат - список, берем первый элемент
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Пытаемся извлечь variables или prompt, иначе конвертируем в строку
        if hasattr(result, "variables"):
            return str(result.variables)
        if hasattr(result, "prompt"):
            return str(result.prompt)
        return str(result)

    async def execute_query(
        self, query: str, temperature: float = 0.3, max_tokens: int = 2048
    ) -> str:
        """Выполнение произвольного LMQL запроса.

        Args:
            query: LMQL запрос в виде строки
            temperature: Температура для генерации
            max_tokens: Максимальное количество токенов

        Returns:
            Результат выполнения запроса

        Raises:
            RuntimeError: Если LMQL недоступен или произошла ошибка
        """
        if not self.available():
            raise RuntimeError("LMQL недоступен")

        try:
            # Выполняем LMQL запрос
            result = await lmql.run(
                query,
                model=self.model_identifier,
                temperature=temperature,
                max_tokens=max_tokens,
                **self.api_config,
            )

            # Извлекаем результат
            return self._extract_result_value(result)

        except Exception as exc:
            logger.error(f"Ошибка выполнения LMQL запроса: {exc}")
            raise RuntimeError(f"Ошибка выполнения LMQL запроса: {exc}") from exc

    async def execute_json_query(
        self,
        prompt: str,
        json_schema: str,
        constraints: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Выполнение LMQL запроса с гарантированным JSON выводом.

        Args:
            prompt: Текст промпта для LLM
            json_schema: Схема JSON с переменными в квадратных скобках
            constraints: Дополнительные ограничения для переменных (опционально)
            temperature: Температура для генерации
            max_tokens: Максимальное количество токенов

        Returns:
            Распарсенный JSON словарь

        Raises:
            RuntimeError: Если LMQL недоступен или произошла ошибка парсинга
        """
        if not self.available():
            raise RuntimeError("LMQL недоступен")

        # Формируем полный LMQL запрос
        where_clause = constraints if constraints else "True"
        query = f'''
argmax
    """{prompt}"""
    
    """{json_schema}"""
from
    "{self.model_identifier}"
where
    {where_clause}
'''

        result = await self.execute_query(query, temperature, max_tokens)

        # Извлекаем JSON из результата
        json_str = self._extract_json_from_result(result)

        if not json_str:
            raise RuntimeError("Не удалось извлечь JSON из результата LMQL")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ошибка парсинга JSON: {e}") from e

    async def execute_validation_query(
        self,
        prompt: str,
        valid_responses: List[str],
        temperature: float = 0.1,
        max_tokens: int = 10,
    ) -> str:
        """Выполнение LMQL запроса для валидации (ДА/НЕТ и т.д.).

        Args:
            prompt: Текст промпта для LLM
            valid_responses: Список допустимых ответов (например, ["ДА", "НЕТ"])
            temperature: Температура для генерации (низкая для детерминированности)
            max_tokens: Максимальное количество токенов (малое для коротких ответов)

        Returns:
            Один из допустимых ответов

        Raises:
            RuntimeError: Если LMQL недоступен или результат невалиден
        """
        if not self.available():
            raise RuntimeError("LMQL недоступен")

        # Формируем список допустимых ответов для where clause
        valid_responses_str = ", ".join([f'"{r}"' for r in valid_responses])

        query = f'''
argmax
    """{prompt}"""
    
    "[RESULT]"
from
    "{self.model_identifier}"
where
    RESULT in [{valid_responses_str}]
'''

        result = await self.execute_query(query, temperature, max_tokens)

        # Извлекаем результат валидации
        validation_result = self._extract_validation_result(result, valid_responses)

        if not validation_result:
            raise RuntimeError(f"Невалидный результат валидации: {result}")

        return validation_result

    def _extract_json_from_result(self, result: str) -> str:
        """Извлечение JSON из результата LMQL.

        Args:
            result: Результат выполнения LMQL запроса

        Returns:
            JSON строка

        Raises:
            RuntimeError: Если JSON не найден
        """
        if not result:
            raise RuntimeError("Пустой результат LMQL")

        # Пытаемся найти JSON в результате
        result = result.strip()

        # Убираем markdown code blocks, если есть
        if "```json" in result:
            start = result.find("```json") + 7
            end = result.find("```", start)
            if end != -1:
                result = result[start:end].strip()
        elif "```" in result:
            start = result.find("```") + 3
            end = result.find("```", start)
            if end != -1:
                result = result[start:end].strip()

        # Ищем JSON объект или массив
        json_start = result.find("{")
        if json_start == -1:
            json_start = result.find("[")

        if json_start != -1:
            # Находим соответствующий закрывающий символ
            bracket = result[json_start]
            close_bracket = "}" if bracket == "{" else "]"
            depth = 0
            json_end = -1

            for i in range(json_start, len(result)):
                if result[i] == bracket:
                    depth += 1
                elif result[i] == close_bracket:
                    depth -= 1
                    if depth == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                json_str = result[json_start:json_end]
                # Проверяем, что это валидный JSON
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass

        # Если не нашли JSON, пытаемся распарсить весь результат
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            pass

        raise RuntimeError(f"Не удалось извлечь JSON из результата: {result[:200]}")

    def _extract_validation_result(
        self, result: str, valid_responses: List[str]
    ) -> str:
        """Извлечение результата валидации из ответа LMQL.

        Args:
            result: Результат выполнения LMQL запроса
            valid_responses: Список допустимых ответов

        Returns:
            Один из допустимых ответов

        Raises:
            RuntimeError: Если результат невалиден
        """
        if not result:
            raise RuntimeError("Пустой результат валидации")

        result = result.strip().upper()

        # Проверяем, содержит ли результат один из допустимых ответов
        for valid_response in valid_responses:
            valid_upper = valid_response.upper()
            if valid_upper in result:
                return valid_response

        # Если не нашли точное совпадение, проверяем начало строки
        for valid_response in valid_responses:
            valid_upper = valid_response.upper()
            if result.startswith(valid_upper):
                return valid_response

        raise RuntimeError(f"Невалидный результат валидации: {result}, ожидалось одно из: {valid_responses}")


def build_lmql_adapter_from_env() -> LMQLAdapter:
    """Создание LMQL адаптера из настроек окружения.

    Returns:
        Экземпляр LMQLAdapter

    Raises:
        RuntimeError: Если LMQL не настроен или недоступен
    """
    settings = get_settings()

    # Проверяем, включен ли LMQL
    if not settings.use_lmql:
        raise RuntimeError(
            "LMQL отключен в настройках. Установите MEMORY_MCP_USE_LMQL=true"
        )

    # Определяем модель
    model = settings.lmql_model or settings.lmstudio_llm_model
    if not model:
        raise RuntimeError(
            "LMQL включен, но модель не указана. "
            "Установите MEMORY_MCP_LMQL_MODEL или MEMORY_MCP_LMSTUDIO_LLM_MODEL"
        )

    # Всегда используем lmstudio
    base_url = f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"

    try:
        adapter = LMQLAdapter(
            model=model,
            backend="lmstudio",
            base_url=base_url,
        )
        logger.info("LMQL адаптер успешно создан")
        return adapter
    except ImportError as e:
        raise RuntimeError(f"LMQL не установлен: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка создания LMQL адаптера: {e}") from e

