"""Pattern analysis service for identifying successful and failed patterns."""

from typing import Dict, List, Any
from collections import defaultdict, Counter
from ..models import Pattern
from ..config import get_settings

settings = get_settings()


class PatternAnalyzer:
    """Service for analyzing patterns in execution data."""
    
    def __init__(self):
        self.patterns: List[Pattern] = []
    
    def analyze_patterns(
        self,
        facts: List[Dict[str, Any]],
        pattern_type: str = "all"
    ) -> List[Pattern]:
        """Analyze facts to identify patterns."""
        
        # Группируем факты по correlation_id
        fact_groups = defaultdict(list)
        for fact in facts:
            corr_id = fact.get("correlation_id")
            if corr_id:
                fact_groups[corr_id].append(fact)
        
        # Анализируем каждую группу
        patterns = []
        
        for corr_id, group in fact_groups.items():
            pattern = self._identify_pattern(group)
            if pattern and (pattern_type == "all" or pattern.pattern_type == pattern_type):
                patterns.append(pattern)
        
        # Агрегируем похожие паттерны
        aggregated_patterns = self._aggregate_patterns(patterns)
        
        # Фильтруем по частоте и уверенности
        filtered_patterns = [
            p for p in aggregated_patterns
            if p.frequency >= settings.pattern_min_frequency
            and p.confidence >= settings.pattern_confidence_threshold
        ]
        
        return filtered_patterns
    
    def _identify_pattern(self, fact_group: List[Dict[str, Any]]) -> Pattern | None:
        """Identify pattern from a group of facts."""
        
        # Извлекаем ключевые характеристики
        plan_facts = [f for f in fact_group if f.get("kind") == "Fact:Plan"]
        exec_facts = [f for f in fact_group if f.get("kind") == "Fact:Execution"]
        error_facts = [f for f in fact_group if f.get("kind") == "Fact:Error"]
        
        if not plan_facts:
            return None
        
        # Определяем тип паттерна
        success_count = sum(
            1 for f in exec_facts
            if f.get("payload", {}).get("success", False)
        )
        
        if len(exec_facts) == 0:
            pattern_type = "neutral"
        elif success_count >= len(exec_facts) * 0.8:
            pattern_type = "success"
        elif error_facts or success_count < len(exec_facts) * 0.5:
            pattern_type = "failure"
        else:
            pattern_type = "neutral"
        
        # Извлекаем условия
        plan_steps = plan_facts[0].get("payload", {}).get("plan_steps", 0)
        
        conditions = {
            "plan_steps": plan_steps,
            "has_errors": len(error_facts) > 0,
            "retry_count": max(0, len(exec_facts) - plan_steps)
        }
        
        # Извлекаем результаты
        outcomes = {
            "success_rate": success_count / len(exec_facts) if exec_facts else 0,
            "error_count": len(error_facts),
            "completion_rate": len(exec_facts) / plan_steps if plan_steps > 0 else 0
        }
        
        # Генерируем описание
        description = self._generate_pattern_description(
            pattern_type,
            conditions,
            outcomes
        )
        
        # Вычисляем уверенность
        confidence = self._calculate_confidence(exec_facts, error_facts)
        
        import hashlib
        pattern_id = hashlib.md5(str(conditions).encode()).hexdigest()[:8]
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            frequency=1,  # Будет обновлено при агрегации
            confidence=confidence,
            conditions=conditions,
            outcomes=outcomes
        )
    
    def _aggregate_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Aggregate similar patterns."""
        
        # Группируем по conditions
        pattern_groups = defaultdict(list)
        
        for pattern in patterns:
            # Создаем ключ из conditions
            key = tuple(sorted(pattern.conditions.items()))
            pattern_groups[key].append(pattern)
        
        # Агрегируем каждую группу
        aggregated = []
        
        for group in pattern_groups.values():
            if not group:
                continue
            
            # Берем первый паттерн как базу
            base_pattern = group[0]
            
            # Обновляем частоту
            base_pattern.frequency = len(group)
            
            # Усредняем outcomes
            avg_outcomes = {}
            for key in base_pattern.outcomes:
                avg_outcomes[key] = sum(
                    p.outcomes.get(key, 0) for p in group
                ) / len(group)
            
            base_pattern.outcomes = avg_outcomes
            
            # Усредняем confidence
            base_pattern.confidence = sum(p.confidence for p in group) / len(group)
            
            aggregated.append(base_pattern)
        
        return aggregated
    
    def _generate_pattern_description(
        self,
        pattern_type: str,
        conditions: Dict[str, Any],
        outcomes: Dict[str, float]
    ) -> str:
        """Generate human-readable pattern description."""
        
        plan_steps = conditions.get("plan_steps", 0)
        has_errors = conditions.get("has_errors", False)
        success_rate = outcomes.get("success_rate", 0)
        
        if pattern_type == "success":
            return (
                f"Successful execution with {plan_steps} steps, "
                f"{success_rate:.1%} success rate"
            )
        elif pattern_type == "failure":
            error_info = " with errors" if has_errors else ""
            return (
                f"Failed execution with {plan_steps} steps{error_info}, "
                f"{success_rate:.1%} success rate"
            )
        else:
            return (
                f"Neutral execution with {plan_steps} steps, "
                f"{success_rate:.1%} success rate"
            )
    
    def _calculate_confidence(
        self,
        exec_facts: List[Dict[str, Any]],
        error_facts: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for pattern."""
        
        # Базовая уверенность на основе количества данных
        base_confidence = min(1.0, len(exec_facts) / 10.0)
        
        # Снижаем уверенность при наличии ошибок
        error_penalty = len(error_facts) * 0.1
        
        confidence = max(0.0, base_confidence - error_penalty)
        
        return confidence
    
    def get_successful_patterns(self, facts: List[Dict[str, Any]]) -> List[Pattern]:
        """Get only successful patterns."""
        return self.analyze_patterns(facts, pattern_type="success")
    
    def get_failed_patterns(self, facts: List[Dict[str, Any]]) -> List[Pattern]:
        """Get only failed patterns."""
        return self.analyze_patterns(facts, pattern_type="failure")
