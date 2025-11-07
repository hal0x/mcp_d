"""Базовые классы для модулей анализа памяти."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from memory import MemoryServiceAdapter
from llm.prompt_manager import PromptManager
from llm.utils import unwrap_response
from metrics import ERRORS

logger = logging.getLogger(__name__)


@dataclass
class ModuleResult:
    """Результат работы модуля анализа."""
    
    module_name: str
    memory_level: str
    analysis: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RequestContext:
    """Контекст запроса для модулей."""
    
    def __init__(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None,
    ):
        self.query = query
        self.user_context = user_context or {}
        self.available_tools = available_tools or []


class PromptModule(ABC):
    """Базовый класс для модулей анализа памяти."""
    
    def __init__(
        self,
        llm_client,
        memory_store: MemoryServiceAdapter,
        prompt_manager: PromptManager,
    ):
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.prompt_manager = prompt_manager
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя модуля."""
        pass
    
    @property
    @abstractmethod
    def memory_levels(self) -> List[str]:
        """Уровни памяти, с которыми работает модуль."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Приоритет модуля (меньше = выше приоритет)."""
        pass
    
    async def prepare(
        self, 
        memory: MemoryServiceAdapter, 
        context: RequestContext
    ) -> Dict[str, Any]:
        """Подготовка данных для анализа."""
        return {
            "query": context.query,
            "user_context": context.user_context,
            "available_tools": context.available_tools,
        }
    
    async def render_prompts(
        self, 
        cfg: PromptManager, 
        vars: Dict[str, Any]
    ) -> Dict[str, str]:
        """Рендеринг промтов для всех уровней памяти."""
        prompts = {}
        
        for level in self.memory_levels:
            try:
                # Получаем данные памяти для уровня
                memory_data = self._get_memory_data(level)
                
                # Получаем промт модуля
                prompt = cfg.get_module_prompt(
                    self.name,
                    level,
                    **{f"{level}_memory": memory_data},
                    query_context=vars.get("query", ""),
                )
                prompts[level] = prompt
                
            except Exception as exc:
                logger.error(f"Ошибка рендеринга промта для {self.name}.{level}: {exc}")
                prompts[level] = f"Ошибка подготовки промта: {str(exc)}"
        
        return prompts
    
    async def postprocess(self, output: Any) -> ModuleResult:
        """Постобработка результатов анализа."""
        if isinstance(output, ModuleResult):
            return output
        
        return ModuleResult(
            module_name=self.name,
            memory_level="unknown",
            analysis=str(output),
            confidence=0.5,
        )
    
    def _get_memory_data(self, level: str) -> str:
        """Получение данных памяти для указанного уровня."""
        try:
            if level == "short_term":
                memory_data = self.memory_store.recall(long_term=False)
            elif level == "long_term":
                memory_data = self.memory_store.recall(long_term=True)
            elif level == "episodic":
                # Для эпизодической памяти используем семантический поиск
                memory_data = self.memory_store.semantic_search(
                    query="recent events and experiences",
                    limit=20
                )
            else:
                memory_data = []
            
            # Преобразуем в строку для промта
            if isinstance(memory_data, list):
                return "\n".join(str(item) for item in memory_data[:10])
            else:
                return str(memory_data)
                
        except Exception as exc:
            logger.error(f"Ошибка получения данных памяти для {level}: {exc}")
            return "Данные памяти недоступны"
    
    async def analyze(
        self, 
        context: RequestContext
    ) -> List[ModuleResult]:
        """Выполнить анализ для всех уровней памяти."""
        results = []
        
        # Подготавливаем данные
        prepared_data = await self.prepare(self.memory_store, context)
        
        # Рендерим промты
        prompts = await self.render_prompts(self.prompt_manager, prepared_data)
        
        # Выполняем анализ для каждого уровня
        for level, prompt in prompts.items():
            try:
                # Генерируем анализ через LLM
                raw_analysis = await self.llm_client.generate(prompt)
                analysis, _ = unwrap_response(raw_analysis)

                # Создаем результат
                result = ModuleResult(
                    module_name=self.name,
                    memory_level=level,
                    analysis=analysis,
                    confidence=self._calculate_confidence(analysis),
                    metadata={
                        "query": context.query,
                        "prompt_length": len(prompt),
                        "analysis_length": len(analysis),
                    }
                )
                
                results.append(result)
                
            except Exception as exc:
                logger.error(f"Ошибка анализа {self.name}.{level}: {exc}")
                # Записываем метрику ошибки
                ERRORS.labels(component=f"module:{self.name}", etype=type(exc).__name__).inc()
                results.append(ModuleResult(
                    module_name=self.name,
                    memory_level=level,
                    analysis=f"Ошибка анализа: {str(exc)}",
                    confidence=0.0,
                    metadata={"error": str(exc)}
                ))
        
        return results
    
    def _calculate_confidence(self, analysis: str) -> float:
        """Вычисление уверенности в результате анализа."""
        # Простая эвристика на основе длины и содержания
        if not analysis or analysis.startswith("Ошибка"):
            return 0.0
        
        # Базовый уровень уверенности
        confidence = 0.5
        
        # Увеличиваем уверенность за структурированность
        if "•" in analysis or "-" in analysis or "1." in analysis:
            confidence += 0.2
        
        # Увеличиваем уверенность за конкретные данные
        if any(word in analysis.lower() for word in ["встреча", "дедлайн", "проект", "задача"]):
            confidence += 0.2
        
        # Увеличиваем уверенность за временные метки
        if any(word in analysis for word in ["завтра", "сегодня", "неделя", "месяц"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
