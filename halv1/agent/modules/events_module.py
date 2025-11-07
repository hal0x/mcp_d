"""Модуль анализа событий и временных данных."""

from __future__ import annotations

import logging
from typing import List

from .base import PromptModule, ModuleResult, RequestContext

logger = logging.getLogger(__name__)


class EventsModule(PromptModule):
    """Модуль для анализа событий, дедлайнов и временных данных."""
    
    @property
    def name(self) -> str:
        return "events"
    
    @property
    def memory_levels(self) -> List[str]:
        return ["short_term", "long_term", "episodic"]
    
    @property
    def priority(self) -> int:
        return 1  # Высокий приоритет для событий
    
    async def analyze_events(
        self, 
        context: RequestContext
    ) -> List[ModuleResult]:
        """Анализ событий с фокусом на временные данные."""
        results = await self.analyze(context)
        
        # Дополнительная обработка для событий
        for result in results:
            if result.memory_level == "short_term":
                result.metadata["focus"] = "current_deadlines"
            elif result.memory_level == "long_term":
                result.metadata["focus"] = "historical_patterns"
            elif result.memory_level == "episodic":
                result.metadata["focus"] = "specific_episodes"
        
        return results
    
    async def extract_deadlines(self, context: RequestContext) -> List[str]:
        """Извлечение дедлайнов из анализа."""
        results = await self.analyze_events(context)
        deadlines = []
        
        for result in results:
            if "дедлайн" in result.analysis.lower() or "deadline" in result.analysis.lower():
                # Простое извлечение дедлайнов из текста
                lines = result.analysis.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["дедлайн", "deadline", "до", "к"]):
                        deadlines.append(line.strip())
        
        return deadlines
    
    async def extract_meetings(self, context: RequestContext) -> List[str]:
        """Извлечение встреч из анализа."""
        results = await self.analyze_events(context)
        meetings = []
        
        for result in results:
            if "встреча" in result.analysis.lower() or "meeting" in result.analysis.lower():
                lines = result.analysis.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["встреча", "meeting", "звонок", "call"]):
                        meetings.append(line.strip())
        
        return meetings
    
    def _calculate_confidence(self, analysis: str) -> float:
        """Специализированная оценка уверенности для событий."""
        confidence = super()._calculate_confidence(analysis)
        
        # Дополнительные факторы для событий
        if any(word in analysis.lower() for word in ["время", "дата", "час", "день"]):
            confidence += 0.1
        
        if any(word in analysis.lower() for word in ["срочно", "важно", "приоритет"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
