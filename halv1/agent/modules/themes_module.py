"""Модуль анализа тем и проектов."""

from __future__ import annotations

import logging
from typing import List

from .base import PromptModule, ModuleResult, RequestContext

logger = logging.getLogger(__name__)


class ThemesModule(PromptModule):
    """Модуль для анализа тем, проектов и тематических знаний."""
    
    @property
    def name(self) -> str:
        return "themes"
    
    @property
    def memory_levels(self) -> List[str]:
        return ["short_term", "long_term", "episodic"]
    
    @property
    def priority(self) -> int:
        return 2  # Средний приоритет для тем
    
    async def analyze_themes(
        self, 
        context: RequestContext
    ) -> List[ModuleResult]:
        """Анализ тем с фокусом на проекты и знания."""
        results = await self.analyze(context)
        
        # Дополнительная обработка для тем
        for result in results:
            if result.memory_level == "short_term":
                result.metadata["focus"] = "active_projects"
            elif result.memory_level == "long_term":
                result.metadata["focus"] = "accumulated_knowledge"
            elif result.memory_level == "episodic":
                result.metadata["focus"] = "discussions_insights"
        
        return results
    
    async def extract_active_projects(self, context: RequestContext) -> List[str]:
        """Извлечение активных проектов из анализа."""
        results = await self.analyze_themes(context)
        projects = []
        
        for result in results:
            if result.memory_level == "short_term":
                lines = result.analysis.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["проект", "project", "разработка", "development"]):
                        projects.append(line.strip())
        
        return projects
    
    async def extract_knowledge_areas(self, context: RequestContext) -> List[str]:
        """Извлечение областей знаний из анализа."""
        results = await self.analyze_themes(context)
        knowledge_areas = []
        
        for result in results:
            if result.memory_level == "long_term":
                lines = result.analysis.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["изучение", "learning", "знание", "knowledge", "навык", "skill"]):
                        knowledge_areas.append(line.strip())
        
        return knowledge_areas
    
    async def link_themes_to_events(self, context: RequestContext) -> Dict[str, List[str]]:
        """Связывание тем с событиями."""
        results = await self.analyze_themes(context)
        theme_links = {}
        
        for result in results:
            theme_links[result.memory_level] = []
            lines = result.analysis.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ["связано", "related", "связь", "connection"]):
                    theme_links[result.memory_level].append(line.strip())
        
        return theme_links
    
    def _calculate_confidence(self, analysis: str) -> float:
        """Специализированная оценка уверенности для тем."""
        confidence = super()._calculate_confidence(analysis)
        
        # Дополнительные факторы для тем
        if any(word in analysis.lower() for word in ["тема", "theme", "проект", "project"]):
            confidence += 0.1
        
        if any(word in analysis.lower() for word in ["знание", "knowledge", "навык", "skill"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
