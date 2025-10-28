"""Offline and incremental training for Learning MCP."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from ..config import get_settings
from ..models import DecisionProfile, TrainingRequest, TrainingResult
from .feature_engineering import build_feature_matrix
from .explainability import compute_permutation_importance
from .model_registry import register_profile
from .supervisor_client import SupervisorClient

settings = get_settings()


class TrainerService:
    """Service for offline/online training on supervisor aggregates."""

    def __init__(
        self,
        supervisor_url: Optional[str] = None,
        supervisor_client: Optional[SupervisorClient] = None,
    ) -> None:
        base_url = supervisor_url or settings.supervisor_url
        self._supervisor_client = supervisor_client or SupervisorClient(
            base_url=base_url, timeout=settings.supervisor_timeout
        )

    async def train_offline(self, request: TrainingRequest) -> TrainingResult:
        start_time = time.time()

        aggregates = await self._supervisor_client.fetch_aggregates(
            window=request.window, kind="business"
        )
        facts = await self._supervisor_client.fetch_facts(window=request.window)

        X_df, y = build_feature_matrix(facts)
        if len(y) < request.min_samples:
            raise ValueError(
                f"Insufficient data: {len(y)} samples, need {request.min_samples}"
            )

        grid = self._tune_hyperparameters(X_df.values, y)
        model: RandomForestClassifier = grid.best_estimator_
        validation_score = float(grid.best_score_)
        cv_scores = cross_val_score(model, X_df.values, y, cv=3)

        feature_names = list(X_df.columns)
        feature_importance = self._extract_feature_importance(model, feature_names)
        permutation_scores = compute_permutation_importance(model, X_df, y)

        profile = self._generate_profile(
            feature_importance,
            aggregates,
            validation_score,
            len(y),
            request.focus_metric,
        )
        insights = self._generate_insights(feature_importance, aggregates, facts)

        duration = time.time() - start_time
        result = TrainingResult(
            profile=profile,
            training_duration=duration,
            samples_used=len(y),
            validation_score=validation_score,
            insights=insights,
            cv_scores=cv_scores.tolist(),
            best_params=grid.best_params_,
            feature_importance=feature_importance,
            permutation_importance=permutation_scores,
        )

        register_profile(profile, result)
        return result

    async def close(self) -> None:
        await self._supervisor_client.close()

    def _tune_hyperparameters(self, X: np.ndarray, y: List[int]) -> GridSearchCV:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 4],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring="accuracy",
        )
        grid.fit(X, y)
        return grid

    def _extract_feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: List[str],
    ) -> Dict[str, float]:
        importance = model.feature_importances_
        return dict(zip(feature_names, importance.tolist()))

    def _generate_profile(
        self,
        feature_importance: Dict[str, float],
        aggregates: Dict[str, Any],
        validation_score: float,
        samples_count: int,
        focus_metric: str,
    ) -> DecisionProfile:
        weights = {
            "complexity": feature_importance.get("plan_steps", 0.3),
            "duration": feature_importance.get("avg_duration", 0.2),
            "errors": feature_importance.get("error_count", 0.2),
            "efficiency": feature_importance.get("execution_span_seconds", 0.3),
        }
        total = sum(weights.values()) or 1.0
        weights = {key: value / total for key, value in weights.items()}

        metrics = aggregates.get("metrics", {})

        thresholds = {
            "min_confidence": max(0.5, validation_score),
            "max_error_rate": 0.1,
            "max_duration": metrics.get("avg_latency", 300.0) or 300.0,
        }
        risk_limits = {
            "max_concurrent_tasks": 5,
            "max_retry_attempts": 3,
            "timeout_seconds": 600.0,
        }

        performance_metrics = {
            "validation_score": validation_score,
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_duration": metrics.get("avg_latency", 0.0),
            "samples_count": samples_count,
        }

        return DecisionProfile(
            profile_id=f"profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            version="1.0.0",
            name=f"Optimized for {focus_metric}",
            description=f"Profile trained on {samples_count} samples (window {settings.online_learning_window})",
            weights=weights,
            thresholds=thresholds,
            risk_limits=risk_limits,
            created_at=datetime.utcnow(),
            trained_on_samples=samples_count,
            confidence_score=validation_score,
            performance_metrics=performance_metrics,
        )

    def _generate_insights(
        self,
        feature_importance: Dict[str, float],
        aggregates: Dict[str, Any],
        facts: List[Dict[str, Any]],
    ) -> List[str]:
        insights: List[str] = []

        if feature_importance:
            most_important = max(feature_importance.items(), key=lambda item: item[1])
            insights.append(
                f"Key driver: {most_important[0]} (importance {most_important[1]:.2f})"
            )

        metrics = aggregates.get("metrics", {})
        success_rate = metrics.get("success_rate", 0.0)
        insights.append(f"Supervisor success rate: {success_rate:.1%}")

        error_facts = sum(1 for fact in facts if fact.get("kind") == "Fact:Error")
        if error_facts:
            insights.append(f"Detected {error_facts} error facts in training window")

        return insights
