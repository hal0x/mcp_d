"""Система адаптации и обучения на взаимодействиях."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.monitoring import send_alert, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Запись о взаимодействии с пользователем."""
    timestamp: datetime
    query: str
    response: str
    user_feedback: Optional[str] = None
    success: bool = True
    processing_time: float = 0.0
    modules_used: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Профиль пользователя для адаптации."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует профиль в словарь для сохранения."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "interaction_patterns": self.interaction_patterns,
            "feedback_history": self.feedback_history,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Создает профиль из словаря."""
        return cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            interaction_patterns=data.get("interaction_patterns", {}),
            feedback_history=data.get("feedback_history", []),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )


class AdaptationEngine:
    """Движок адаптации и обучения на взаимодействиях."""
    
    def __init__(
        self,
        profile_path: str = "db/adaptation/user_profiles.json",
        interaction_history_path: str = "db/adaptation/interactions.json",
        learning_threshold: int = 10,  # Минимум взаимодействий для обучения
    ):
        self.profile_path = Path(profile_path)
        self.interaction_history_path = Path(interaction_history_path)
        self.learning_threshold = learning_threshold
        
        # Создаем директории если не существуют
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Загружаем данные
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: List[Interaction] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Загружает профили пользователей и историю взаимодействий."""
        # Загружаем профили пользователей
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        self.user_profiles[user_id] = UserProfile.from_dict(profile_data)
                logger.info(f"Загружено {len(self.user_profiles)} профилей пользователей")
            except Exception as exc:
                logger.error(f"Ошибка загрузки профилей пользователей: {exc}")
        
        # Загружаем историю взаимодействий
        if self.interaction_history_path.exists():
            try:
                with open(self.interaction_history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for interaction_data in data:
                        interaction = Interaction(
                            timestamp=datetime.fromisoformat(interaction_data["timestamp"]),
                            query=interaction_data["query"],
                            response=interaction_data["response"],
                            user_feedback=interaction_data.get("user_feedback"),
                            success=interaction_data.get("success", True),
                            processing_time=interaction_data.get("processing_time", 0.0),
                            modules_used=interaction_data.get("modules_used", []),
                            confidence_scores=interaction_data.get("confidence_scores", {}),
                            metadata=interaction_data.get("metadata", {})
                        )
                        self.interaction_history.append(interaction)
                logger.info(f"Загружено {len(self.interaction_history)} взаимодействий")
            except Exception as exc:
                logger.error(f"Ошибка загрузки истории взаимодействий: {exc}")
    
    def _save_data(self) -> None:
        """Сохраняет профили пользователей и историю взаимодействий."""
        try:
            # Сохраняем профили пользователей
            profiles_data = {
                user_id: profile.to_dict() 
                for user_id, profile in self.user_profiles.items()
            }
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            
            # Сохраняем историю взаимодействий (только последние 1000)
            recent_interactions = self.interaction_history[-1000:]
            interactions_data = []
            for interaction in recent_interactions:
                interactions_data.append({
                    "timestamp": interaction.timestamp.isoformat(),
                    "query": interaction.query,
                    "response": interaction.response,
                    "user_feedback": interaction.user_feedback,
                    "success": interaction.success,
                    "processing_time": interaction.processing_time,
                    "modules_used": interaction.modules_used,
                    "confidence_scores": interaction.confidence_scores,
                    "metadata": interaction.metadata
                })
            
            with open(self.interaction_history_path, 'w', encoding='utf-8') as f:
                json.dump(interactions_data, f, ensure_ascii=False, indent=2)
            
            logger.debug("Данные адаптации сохранены")
        except Exception as exc:
            logger.error(f"Ошибка сохранения данных адаптации: {exc}")
    
    def record_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        success: bool = True,
        processing_time: float = 0.0,
        modules_used: Optional[List[str]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Записывает взаимодействие с пользователем."""
        interaction = Interaction(
            timestamp=datetime.now(),
            query=query,
            response=response,
            success=success,
            processing_time=processing_time,
            modules_used=modules_used or [],
            confidence_scores=confidence_scores or {},
            metadata=metadata or {}
        )
        
        self.interaction_history.append(interaction)
        
        # Обновляем профиль пользователя
        self._update_user_profile(user_id, interaction)
        
        # Сохраняем данные
        self._save_data()
        
        # Проверяем, нужно ли обучение
        if len(self.interaction_history) >= self.learning_threshold:
            self._trigger_learning()
    
    def record_feedback(
        self,
        user_id: str,
        query: str,
        feedback: str,
        rating: Optional[int] = None
    ) -> None:
        """Записывает обратную связь пользователя."""
        # Находим последнее взаимодействие с этим запросом
        for interaction in reversed(self.interaction_history):
            if interaction.query == query and not interaction.user_feedback:
                interaction.user_feedback = feedback
                if rating is not None:
                    interaction.metadata["rating"] = rating
                break
        
        # Обновляем профиль пользователя
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.feedback_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "feedback": feedback,
                "rating": rating
            })
            profile.last_updated = datetime.now()
        
        self._save_data()
    
    def _update_user_profile(self, user_id: str, interaction: Interaction) -> None:
        """Обновляет профиль пользователя на основе взаимодействия."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Обновляем паттерны взаимодействий
        query_lower = interaction.query.lower()
        
        # Анализируем типы запросов
        if any(word in query_lower for word in ["время", "встреча", "дедлайн"]):
            profile.interaction_patterns["events_queries"] = profile.interaction_patterns.get("events_queries", 0) + 1
        elif any(word in query_lower for word in ["проект", "тема", "знание"]):
            profile.interaction_patterns["themes_queries"] = profile.interaction_patterns.get("themes_queries", 0) + 1
        elif any(word in query_lower for word in ["настроение", "эмоция", "чувство"]):
            profile.interaction_patterns["emotions_queries"] = profile.interaction_patterns.get("emotions_queries", 0) + 1
        
        # Обновляем предпочтения по модулям
        for module in interaction.modules_used:
            profile.preferences[f"module_{module}_usage"] = profile.preferences.get(f"module_{module}_usage", 0) + 1
        
        # Обновляем предпочтения по времени ответа
        if interaction.processing_time > 0:
            avg_time = profile.preferences.get("avg_response_time", 0.0)
            count = profile.preferences.get("response_count", 0)
            new_avg = (avg_time * count + interaction.processing_time) / (count + 1)
            profile.preferences["avg_response_time"] = new_avg
            profile.preferences["response_count"] = count + 1
        
        profile.last_updated = datetime.now()
    
    def _trigger_learning(self) -> None:
        """Запускает процесс обучения на основе накопленных данных."""
        logger.info("🧠 Запуск процесса обучения на взаимодействиях")
        
        try:
            # Анализируем успешность взаимодействий
            recent_interactions = self.interaction_history[-50:]  # Последние 50
            success_rate = sum(1 for i in recent_interactions if i.success) / len(recent_interactions)
            
            # Анализируем время ответа
            avg_response_time = sum(i.processing_time for i in recent_interactions) / len(recent_interactions)
            
            # Анализируем использование модулей
            module_usage = {}
            for interaction in recent_interactions:
                for module in interaction.modules_used:
                    module_usage[module] = module_usage.get(module, 0) + 1
            
            # Генерируем инсайты
            insights = []
            
            if success_rate < 0.8:
                insights.append(f"Низкая успешность взаимодействий: {success_rate:.1%}")
            
            if avg_response_time > 5.0:
                insights.append(f"Медленные ответы: {avg_response_time:.1f}с в среднем")
            
            if module_usage:
                most_used = max(module_usage.items(), key=lambda x: x[1])
                insights.append(f"Наиболее используемый модуль: {most_used[0]} ({most_used[1]} раз)")
            
            # Отправляем алерт с инсайтами
            if insights:
                try:
                    import asyncio
                    asyncio.create_task(send_alert(
                        AlertLevel.INFO,
                        "Инсайты обучения",
                        "Анализ взаимодействий выявил следующие паттерны:\n\n" + "\n".join(f"• {insight}" for insight in insights)
                    ))
                except Exception as e:
                    logger.warning(f"Не удалось отправить алерт: {e}")
            
            logger.info(f"Обучение завершено. Инсайтов: {len(insights)}")
            
        except Exception as exc:
            logger.error(f"Ошибка при обучении: {exc}")
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Получает профиль пользователя."""
        return self.user_profiles.get(user_id)
    
    def get_adaptation_suggestions(self, user_id: str) -> List[str]:
        """Получает предложения по адаптации для пользователя."""
        profile = self.get_user_profile(user_id)
        if not profile:
            return []
        
        suggestions = []
        
        # Анализируем предпочтения по модулям
        module_usage = {
            k.replace("module_", "").replace("_usage", ""): v 
            for k, v in profile.preferences.items() 
            if k.startswith("module_") and k.endswith("_usage")
        }
        
        if module_usage:
            most_used = max(module_usage.items(), key=lambda x: x[1])
            suggestions.append(f"Пользователь часто использует модуль {most_used[0]}")
        
        # Анализируем время ответа
        avg_time = profile.preferences.get("avg_response_time", 0)
        if avg_time > 3.0:
            suggestions.append("Пользователь предпочитает быстрые ответы")
        
        # Анализируем паттерны запросов
        if profile.interaction_patterns.get("events_queries", 0) > 5:
            suggestions.append("Пользователь часто спрашивает о событиях и времени")
        
        return suggestions
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Получает статистику обучения."""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        recent_interactions = self.interaction_history[-100:]  # Последние 100
        
        return {
            "total_interactions": len(self.interaction_history),
            "recent_interactions": len(recent_interactions),
            "success_rate": sum(1 for i in recent_interactions if i.success) / len(recent_interactions),
            "avg_response_time": sum(i.processing_time for i in recent_interactions) / len(recent_interactions),
            "unique_users": len(self.user_profiles),
            "feedback_count": sum(len(p.feedback_history) for p in self.user_profiles.values())
        }


# Глобальный экземпляр для использования в других модулях
_adaptation_engine: Optional[AdaptationEngine] = None


def get_adaptation_engine() -> Optional[AdaptationEngine]:
    """Получает глобальный экземпляр движка адаптации."""
    return _adaptation_engine


def set_adaptation_engine(engine: AdaptationEngine) -> None:
    """Устанавливает глобальный экземпляр движка адаптации."""
    global _adaptation_engine
    _adaptation_engine = engine


def record_interaction(
    user_id: str,
    query: str,
    response: str,
    success: bool = True,
    processing_time: float = 0.0,
    modules_used: Optional[List[str]] = None,
    confidence_scores: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Записывает взаимодействие через глобальный движок адаптации."""
    engine = get_adaptation_engine()
    if engine:
        engine.record_interaction(
            user_id=user_id,
            query=query,
            response=response,
            success=success,
            processing_time=processing_time,
            modules_used=modules_used,
            confidence_scores=confidence_scores,
            metadata=metadata
        )
    else:
        logger.warning("Движок адаптации недоступен, взаимодействие не записано")
