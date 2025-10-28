"""Online learning and A/B testing service for Learning MCP."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import get_settings
from ..models import DecisionProfile, TrainingRequest, TrainingResult
from ..services.policy_client import PolicyClient
from ..services.supervisor_client import SupervisorClient
from ..services.trainer import TrainerService


class OnlineLearningService:
    """Coordinates incremental training cycles and A/B evaluation."""

    def __init__(
        self,
        trainer_service: TrainerService,
        supervisor_client: SupervisorClient,
        policy_client: PolicyClient,
    ) -> None:
        self._trainer_service = trainer_service
        self._supervisor_client = supervisor_client
        self._policy_client = policy_client
        settings = get_settings()
        self._window = settings.online_learning_window
        self._interval_seconds = settings.online_learning_interval_seconds
        self._threshold = settings.ab_test_threshold
        self._min_samples = max(settings.min_samples // 2, 10)
        self._focus_metric = settings.default_metric

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    async def run_cycle(self) -> Dict[str, Any]:
        """Run a single online learning + A/B evaluation cycle."""
        request = TrainingRequest(
            window=self._window,
            min_samples=self._min_samples,
            focus_metric=self._focus_metric,
        )

        result = await self._trainer_service.train_offline(request)

        # Persist candidate profile without activation
        await self._policy_client.upsert_profile(result.profile, activate=False)

        comparison = await self._evaluate_candidate(result.profile)

        return {
            "profile_id": result.profile.profile_id,
            "activated": comparison.get("activated", False),
            "comparison": comparison,
            "training_duration": result.training_duration,
            "samples_used": result.samples_used,
            "validation_score": result.validation_score,
        }

    async def _evaluate_candidate(self, candidate: DecisionProfile) -> Dict[str, Any]:
        """Compare candidate profile with current active and decide activation."""
        try:
            current_response = await self._policy_client.get_active_profile()
            current_dict = current_response.get("profile")
        except Exception:
            current_dict = None

        if not current_dict:
            await self._policy_client.upsert_profile(candidate, activate=True)
            return {"activated": True, "reason": "no-active-profile"}

        current = current_dict
        comparison = self.compare_profiles(candidate, current)

        if comparison["winner"] == "candidate":
            await self._policy_client.upsert_profile(candidate, activate=True)
            comparison["activated"] = True
        else:
            comparison["activated"] = False

        return comparison

    def compare_profiles(
        self,
        candidate: DecisionProfile,
        current_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform simple A/B comparison using performance metrics."""
        candidate_metrics = candidate.performance_metrics or {}
        current_metrics = current_profile.get("performance_metrics", {})

        candidate_score = float(candidate_metrics.get("success_rate", 0.0))
        candidate_validation = float(candidate_metrics.get("validation_score", candidate.confidence_score))
        current_score = float(current_metrics.get("success_rate", 0.0))
        current_validation = float(current_metrics.get("validation_score", 0.0))

        score_delta = candidate_score - current_score
        validation_delta = candidate_validation - current_validation

        winner = "candidate" if score_delta > self._threshold else "current"

        return {
            "winner": winner,
            "candidate_score": candidate_score,
            "current_score": current_score,
            "score_delta": score_delta,
            "candidate_validation": candidate_validation,
            "current_validation": current_validation,
            "validation_delta": validation_delta,
        }
