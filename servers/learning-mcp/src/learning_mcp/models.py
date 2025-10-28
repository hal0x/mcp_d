"""Pydantic models for learning MCP."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionProfile(BaseModel):
    """Decision profile for agent behavior."""
    profile_id: str
    version: str
    name: str
    description: str
    
    # Веса для различных сигналов
    weights: Dict[str, float] = Field(default_factory=dict)
    
    # Пороги для принятия решений
    thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Ограничения рисков
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    
    # Метаданные
    created_at: datetime
    trained_on_samples: int
    confidence_score: float
    
    # Метрики производительности (из бэктеста)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class TrainingRequest(BaseModel):
    """Request for offline training."""
    window: str = Field(default="7d", description="Time window (7d, 30d)")
    min_samples: int = Field(default=100, description="Minimum samples required")
    focus_metric: str = Field(default="success_rate", description="Metric to optimize")
    constraints: Optional[Dict[str, Any]] = None


class TrainingResult(BaseModel):
    """Result of training process."""
    profile: DecisionProfile
    training_duration: float
    samples_used: int
    validation_score: float
    insights: List[str] = Field(default_factory=list)
    cv_scores: List[float] = Field(default_factory=list)
    best_params: Dict[str, Any] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    permutation_importance: Dict[str, float] = Field(default_factory=dict)


class Pattern(BaseModel):
    """Identified pattern in data."""
    pattern_id: str
    pattern_type: str  # "success", "failure", "neutral"
    description: str
    frequency: int
    confidence: float
    conditions: Dict[str, Any]
    outcomes: Dict[str, float]


class ComparisonResult(BaseModel):
    """Comparison between profiles."""
    profile_a_id: str
    profile_b_id: str
    metric_comparisons: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    confidence: float
    recommendation: str


class CorrelationAnalysis(BaseModel):
    """Correlation analysis result."""
    metric_pairs: List[Dict[str, Any]]
    significant_correlations: List[Dict[str, float]]
    insights: List[str]
