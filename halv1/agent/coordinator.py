"""Координатор модулей для HAL AI-агента."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from llm.base_client import LLMClient
from llm.prompt_manager import PromptManager
from memory import MemoryServiceAdapter
from .modules import EventsModule, ThemesModule, EmotionsModule, RequestContext
from metrics import COORDINATOR_DECISION, ERRORS

logger = logging.getLogger(__name__)


class ModuleCoordinator:
    """Координирует работу модулей (события, темы, эмоции) и инструментов."""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_store: MemoryServiceAdapter,
        prompt_manager: PromptManager,
    ):
        """Инициализация координатора.
        
        Args:
            llm_client: Клиент LLM для генерации ответов
            memory_store: Хранилище памяти
            prompt_manager: Менеджер промтов
        """
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.prompt_manager = prompt_manager
        
        # Инициализируем модули анализа
        self.modules = {
            "events": EventsModule(llm_client, memory_store, prompt_manager),
            "themes": ThemesModule(llm_client, memory_store, prompt_manager),
            "emotions": EmotionsModule(llm_client, memory_store, prompt_manager),
        }

    async def process_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя с использованием модулей анализа."""
        # Создаем контекст запроса
        context = RequestContext(
            query=query,
            user_context=user_context,
            available_tools=available_tools
        )
        
        # Выбираем релевантные модули
        selected_modules = self.select_modules(context)
        
        # Выполняем анализ через модули
        module_results = {}
        for module_name in selected_modules:
            try:
                module = self.modules[module_name]
                results = await module.analyze(context)
                module_results[module_name] = results
                # Записываем метрику успешного решения
                COORDINATOR_DECISION.labels(policy="module_selection", module=module_name).inc()
            except Exception as exc:
                logger.error(f"Ошибка выполнения модуля {module_name}: {exc}")
                module_results[module_name] = []
                # Записываем метрику ошибки
                ERRORS.labels(component="coordinator", etype=type(exc).__name__).inc()
        
        # Формируем итоговый ответ
        return {
            "query": query,
            "module_results": module_results,
            "selected_modules": selected_modules,
            "user_context": user_context,
            "available_tools": available_tools,
        }
    
    def select_modules(self, context: RequestContext) -> List[str]:
        """Выбирает релевантные модули для анализа запроса."""
        selected = []
        query_lower = context.query.lower()
        
        # Простая эвристика выбора модулей на основе ключевых слов
        if any(word in query_lower for word in ["событие", "встреча", "дедлайн", "время", "планы", "расписание"]):
            selected.append("events")
        
        if any(word in query_lower for word in ["проект", "тема", "знание", "навык", "изучение", "работа"]):
            selected.append("themes")
        
        if any(word in query_lower for word in ["настроение", "эмоция", "чувство", "стресс", "радость", "проблема"]):
            selected.append("emotions")
        
        # Если ничего не выбрано, используем все модули
        if not selected:
            selected = list(self.modules.keys())
        
        # Сортируем по приоритету
        selected.sort(key=lambda name: self.modules[name].priority)
        
        return selected
    
    async def get_module_analysis(
        self,
        module_name: str,
        context: RequestContext
    ) -> List[Any]:
        """Получить анализ от конкретного модуля."""
        if module_name not in self.modules:
            raise ValueError(f"Модуль {module_name} не найден")
        
        module = self.modules[module_name]
        return await module.analyze(context)
    
    async def get_events_analysis(self, context: RequestContext) -> List[Any]:
        """Получить анализ событий."""
        return await self.get_module_analysis("events", context)
    
    async def get_themes_analysis(self, context: RequestContext) -> List[Any]:
        """Получить анализ тем."""
        return await self.get_module_analysis("themes", context)
    
    async def get_emotions_analysis(self, context: RequestContext) -> List[Any]:
        """Получить анализ эмоций."""
        return await self.get_module_analysis("emotions", context)
    
    async def get_specialized_analysis(
        self,
        query: str,
        analysis_type: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Получить специализированный анализ от конкретного модуля."""
        context = RequestContext(
            query=query,
            user_context=user_context
        )
        
        if analysis_type == "events":
            return await self.get_events_analysis(context)
        elif analysis_type == "themes":
            return await self.get_themes_analysis(context)
        elif analysis_type == "emotions":
            return await self.get_emotions_analysis(context)
        else:
            raise ValueError(f"Неизвестный тип анализа: {analysis_type}")
    
    def get_available_modules(self) -> List[str]:
        """Получить список доступных модулей."""
        return list(self.modules.keys())
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Получить информацию о модуле."""
        if module_name not in self.modules:
            raise ValueError(f"Модуль {module_name} не найден")
        
        module = self.modules[module_name]
        return {
            "name": module.name,
            "priority": module.priority,
            "memory_levels": module.memory_levels,
        }