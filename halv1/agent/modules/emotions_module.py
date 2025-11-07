"""Модуль анализа эмоций и настроения."""

from __future__ import annotations

import logging
from typing import List, Dict

from .base import PromptModule, ModuleResult, RequestContext

logger = logging.getLogger(__name__)


class EmotionsModule(PromptModule):
    """Модуль для анализа эмоций, настроения и стресс-факторов."""
    
    @property
    def name(self) -> str:
        return "emotions"
    
    @property
    def memory_levels(self) -> List[str]:
        return ["short_term", "long_term", "episodic"]
    
    @property
    def priority(self) -> int:
        return 3  # Низкий приоритет для эмоций
    
    async def analyze_emotions(
        self, 
        context: RequestContext
    ) -> List[ModuleResult]:
        """Анализ эмоций с фокусом на настроение и стресс."""
        results = await self.analyze(context)
        
        # Дополнительная обработка для эмоций
        for result in results:
            if result.memory_level == "short_term":
                result.metadata["focus"] = "current_mood"
            elif result.memory_level == "long_term":
                result.metadata["focus"] = "emotional_patterns"
            elif result.memory_level == "episodic":
                result.metadata["focus"] = "emotional_moments"
        
        return results
    
    async def extract_stress_factors(self, context: RequestContext) -> List[str]:
        """Извлечение стресс-факторов из анализа."""
        results = await self.analyze_emotions(context)
        stress_factors = []
        
        for result in results:
            lines = result.analysis.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ["стресс", "stress", "напряжение", "tension", "проблема", "problem"]):
                    stress_factors.append(line.strip())
        
        return stress_factors
    
    async def extract_positive_moments(self, context: RequestContext) -> List[str]:
        """Извлечение позитивных моментов из анализа."""
        results = await self.analyze_emotions(context)
        positive_moments = []
        
        for result in results:
            lines = result.analysis.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ["радость", "joy", "успех", "success", "хорошо", "good", "отлично", "great"]):
                    positive_moments.append(line.strip())
        
        return positive_moments
    
    async def analyze_emotional_patterns(self, context: RequestContext) -> Dict[str, List[str]]:
        """Анализ эмоциональных паттернов."""
        results = await self.analyze_emotions(context)
        patterns = {
            "triggers": [],
            "responses": [],
            "cycles": []
        }
        
        for result in results:
            lines = result.analysis.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(word in line_lower for word in ["триггер", "trigger", "причина", "cause"]):
                    patterns["triggers"].append(line.strip())
                elif any(word in line_lower for word in ["реакция", "response", "ответ", "answer"]):
                    patterns["responses"].append(line.strip())
                elif any(word in line_lower for word in ["цикл", "cycle", "повтор", "repeat"]):
                    patterns["cycles"].append(line.strip())
        
        return patterns
    
    def _calculate_confidence(self, analysis: str) -> float:
        """Специализированная оценка уверенности для эмоций."""
        confidence = super()._calculate_confidence(analysis)
        
        # Дополнительные факторы для эмоций
        if any(word in analysis.lower() for word in ["эмоция", "emotion", "настроение", "mood"]):
            confidence += 0.1
        
        if any(word in analysis.lower() for word in ["чувство", "feeling", "ощущение", "sensation"]):
            confidence += 0.1
        
        # Снижаем уверенность для эмоционального анализа (более субъективно)
        confidence *= 0.8
        
        return min(confidence, 1.0)
