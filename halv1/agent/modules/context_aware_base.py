#!/usr/bin/env python3
"""Контекстно-осведомленные модули анализа памяти."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from memory import MemoryServiceAdapter
from llm.prompt_manager import PromptManager
from llm.base_client import ConversationHistory
from llm.context_aware_client import ContextAwareWrapper
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


class ContextAwarePromptModule(ABC):
    """Базовый класс для контекстно-осведомленных модулей анализа памяти."""
    
    def __init__(
        self,
        name: str,
        memory_store: MemoryServiceAdapter,
        llm_client: ContextAwareWrapper,
        prompt_manager: PromptManager,
    ):
        self.name = name
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self._history: Optional[ConversationHistory] = None
    
    @abstractmethod
    async def prepare(self, memory_store: MemoryServiceAdapter, context: RequestContext) -> Dict[str, Any]:
        """Подготовить данные для анализа."""
        pass
    
    @abstractmethod
    async def render_prompts(self, prompt_manager: PromptManager, data: Dict[str, Any]) -> Dict[str, str]:
        """Рендерить промпты для анализа."""
        pass
    
    async def analyze(
        self, 
        context: RequestContext
    ) -> List[ModuleResult]:
        """Выполнить анализ для всех уровней памяти с переиспользованием контекста."""
        results = []
        
        # Подготавливаем данные
        prepared_data = await self.prepare(self.memory_store, context)
        
        # Рендерим промты
        prompts = await self.render_prompts(self.prompt_manager, prepared_data)
        
        # Выполняем анализ для каждого уровня с переиспользованием контекста
        for level, prompt in prompts.items():
            try:
                # Генерируем анализ через LLM с контекстом
                analysis, new_history = self.llm_client.generate(prompt, self._history)
                self._history = new_history
                
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
                        "context_reused": self._history is not None,
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
        """Вычислить уверенность в результате анализа."""
        if not analysis or len(analysis) < 10:
            return 0.0
        
        # Простая эвристика на основе длины и содержания
        confidence = min(1.0, len(analysis) / 100.0)
        
        # Бонус за наличие ключевых слов
        keywords = ["анализ", "результат", "вывод", "рекомендация", "проблема", "решение"]
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in analysis.lower())
        confidence += keyword_count * 0.1
        
        return min(1.0, confidence)
    
    def get_context(self) -> Optional[ConversationHistory]:
        """Получить текущий контекст."""
        return self._history
    
    def clear_context(self):
        """Очистить контекст."""
        self._history = None
    
    def set_context(self, context: ConversationHistory):
        """Установить контекст."""
        self._history = context


class ContextAwareEventsModule(ContextAwarePromptModule):
    """Контекстно-осведомленный модуль анализа событий."""
    
    def __init__(
        self,
        memory_store: MemoryServiceAdapter,
        llm_client: ContextAwareWrapper,
        prompt_manager: PromptManager,
    ):
        super().__init__("events", memory_store, llm_client, prompt_manager)
    
    async def prepare(self, memory_store: MemoryServiceAdapter, context: RequestContext) -> Dict[str, Any]:
        """Подготовить данные событий."""
        return {
            "short_term_memory": self._get_memory_data(memory_store, "short_term"),
            "long_term_memory": self._get_memory_data(memory_store, "long_term"),
            "episodic_memory": self._get_memory_data(memory_store, "episodic"),
            "current_time": context.user_context.get("current_time", ""),
            "query_context": context.query,
        }
    
    async def render_prompts(self, prompt_manager: PromptManager, data: Dict[str, Any]) -> Dict[str, str]:
        """Рендерить промпты для анализа событий."""
        prompts = {}
        
        # Краткосрочные события
        if data["short_term_memory"]:
            prompts["short_term"] = prompt_manager.render_template(
                "module_prompts.events.short_term",
                short_term_memory=data["short_term_memory"],
                current_time=data["current_time"]
            )
        
        # Долгосрочные события
        if data["long_term_memory"]:
            prompts["long_term"] = prompt_manager.render_template(
                "module_prompts.events.long_term",
                long_term_memory=data["long_term_memory"]
            )
        
        # Эпизодические события
        if data["episodic_memory"]:
            prompts["episodic"] = prompt_manager.render_template(
                "module_prompts.events.episodic",
                episodic_memory=data["episodic_memory"],
                query_context=data["query_context"]
            )
        
        return prompts
    
    def _get_memory_data(self, memory_store: MemoryServiceAdapter, level: str) -> str:
        """Получить данные памяти для уровня."""
        try:
            if level == "short_term":
                return memory_store.get_short_term_memory()
            elif level == "long_term":
                return memory_store.get_long_term_memory()
            elif level == "episodic":
                return memory_store.get_episodic_memory()
            else:
                return ""
        except Exception as exc:
            logger.error(f"Ошибка получения данных памяти {level}: {exc}")
            return "Данные памяти недоступны"


class ContextAwareThemesModule(ContextAwarePromptModule):
    """Контекстно-осведомленный модуль анализа тем."""
    
    def __init__(
        self,
        memory_store: MemoryServiceAdapter,
        llm_client: ContextAwareWrapper,
        prompt_manager: PromptManager,
    ):
        super().__init__("themes", memory_store, llm_client, prompt_manager)
    
    async def prepare(self, memory_store: MemoryServiceAdapter, context: RequestContext) -> Dict[str, Any]:
        """Подготовить данные тем."""
        return {
            "short_term_memory": self._get_memory_data(memory_store, "short_term"),
            "long_term_memory": self._get_memory_data(memory_store, "long_term"),
            "episodic_memory": self._get_memory_data(memory_store, "episodic"),
            "query_context": context.query,
        }
    
    async def render_prompts(self, prompt_manager: PromptManager, data: Dict[str, Any]) -> Dict[str, str]:
        """Рендерить промпты для анализа тем."""
        prompts = {}
        
        # Краткосрочные темы
        if data["short_term_memory"]:
            prompts["short_term"] = prompt_manager.render_template(
                "module_prompts.themes.short_term",
                short_term_memory=data["short_term_memory"]
            )
        
        # Долгосрочные темы
        if data["long_term_memory"]:
            prompts["long_term"] = prompt_manager.render_template(
                "module_prompts.themes.long_term",
                long_term_memory=data["long_term_memory"]
            )
        
        # Эпизодические темы
        if data["episodic_memory"]:
            prompts["episodic"] = prompt_manager.render_template(
                "module_prompts.themes.episodic",
                episodic_memory=data["episodic_memory"],
                query_context=data["query_context"]
            )
        
        return prompts
    
    def _get_memory_data(self, memory_store: MemoryServiceAdapter, level: str) -> str:
        """Получить данные памяти для уровня."""
        try:
            if level == "short_term":
                return memory_store.get_short_term_memory()
            elif level == "long_term":
                return memory_store.get_long_term_memory()
            elif level == "episodic":
                return memory_store.get_episodic_memory()
            else:
                return ""
        except Exception as exc:
            logger.error(f"Ошибка получения данных памяти {level}: {exc}")
            return "Данные памяти недоступны"
