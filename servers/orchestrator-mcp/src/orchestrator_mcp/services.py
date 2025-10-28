"""Core orchestration service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from .clients import LearningClient, PolicyClient, SupervisorClient

logger = structlog.get_logger(__name__)


class OrchestratorService:
    """Thin orchestrator for executing MCP plans."""

    def __init__(
        self,
        supervisor_client: SupervisorClient,
        policy_client: PolicyClient,
        learning_client: LearningClient,
    ):
        self._supervisor_client = supervisor_client
        self._policy_client = policy_client
        self._learning_client = learning_client

    async def execute_plan(self, plan: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        logger.info("orchestrator.execute_plan.start", dry_run=dry_run)
        executed_steps: List[Dict[str, Any]] = []

        for idx, step in enumerate(plan):
            step_id = f"step_{idx}"
            action = step.get("action")
            logger.info("orchestrator.step.start", step_id=step_id, action=action)

            if dry_run:
                executed_steps.append({"step": step, "status": "skipped"})
                continue

            try:
                result = await self._perform_action(step)
                executed_steps.append({"step": step, "status": "completed", "result": result})
            except Exception as exc:
                logger.error("orchestrator.step.error", step_id=step_id, error=str(exc))
                rollback_report = await self._rollback(executed_steps)
                return {
                    "success": False,
                    "error": str(exc),
                    "executed_steps": executed_steps,
                    "rollback": rollback_report,
                }

        logger.info("orchestrator.execute_plan.completed")
        return {"success": True, "executed_steps": executed_steps}

    async def _perform_action(self, step: Dict[str, Any]) -> Dict[str, Any]:
        action = step.get("action")
        if action == "check_policy_profile":
            profile_id = step["params"]["profile_id"]
            return await self._policy_client.check_profile(profile_id)
        if action == "fetch_supervisor_health":
            return await self._supervisor_client.fetch_health()
        if action == "trigger_learning_online":
            return await self._learning_client.trigger_online_learning()
        if action == "list_learning_policy_profiles":
            return await self._learning_client.list_policy_profiles()
        if action == "activate_policy_profile":
            profile_id = step["params"]["profile_id"]
            return await self._policy_client.activate_profile(profile_id)
        if action == "list_policy_profiles":
            active_only = step.get("params", {}).get("active_only", False)
            return await self._policy_client.list_profiles(active_only=active_only)
        if action == "configure_policy_experiment":
            params = step["params"]
            return await self._policy_client.configure_experiment(
                profile_id=params["profile_id"],
                name=params["experiment_name"],
                weight=params.get("weight", 0.5),
                metadata=params.get("metadata"),
            )
        if action == "list_policy_experiments":
            return await self._policy_client.list_experiments()
        return {"info": f"No handler for action {action}"}

    async def _rollback(self, executed_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("orchestrator.rollback.start", steps=len(executed_steps))
        # For MVP just log rollback information (no side effects)
        for item in reversed(executed_steps):
            logger.info("orchestrator.rollback.step", step=item["step"])
        return {"rolled_back_steps": len(executed_steps), "timestamp": datetime.utcnow().isoformat()}

    async def dry_run_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate plan without executing actions."""
        return await self.execute_plan(plan, dry_run=True)
