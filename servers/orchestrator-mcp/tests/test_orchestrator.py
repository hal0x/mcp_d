"""Tests for OrchestratorService."""

import pytest

from orchestrator_mcp.services import OrchestratorService


class DummySupervisor:
    async def fetch_health(self):
        return {"status": "ok"}


class DummyPolicy:
    async def check_profile(self, profile_id: str):
        return {"profile_id": profile_id, "active": True}

    async def activate_profile(self, profile_id: str):
        return {"profile_id": profile_id, "activated": True}

    async def configure_experiment(self, profile_id: str, name: str, weight: float, metadata=None):
        return {"profile_id": profile_id, "experiment": name, "weight": weight}

    async def list_experiments(self):
        return {"experiments": {}}

    async def list_profiles(self, active_only: bool = False):
        return {"profiles": [], "active_only": active_only}


class DummyLearning:
    async def trigger_online_learning(self):
        return {"triggered": True}

    async def list_policy_profiles(self):
        return {"profiles": []}


@pytest.mark.asyncio
async def test_execute_plan_success():
    service = OrchestratorService(DummySupervisor(), DummyPolicy(), DummyLearning())
    plan = [
        {"action": "fetch_supervisor_health"},
        {"action": "check_policy_profile", "params": {"profile_id": "demo"}},
        {"action": "trigger_learning_online"},
    ]
    result = await service.execute_plan(plan)
    assert result["success"] is True
    assert len(result["executed_steps"]) == 3


@pytest.mark.asyncio
async def test_dry_run():
    service = OrchestratorService(DummySupervisor(), DummyPolicy(), DummyLearning())
    plan = [{"action": "unknown"}]
    result = await service.dry_run_plan(plan)
    assert result["success"] is True
    assert result["executed_steps"][0]["status"] == "skipped"


@pytest.mark.asyncio
async def test_execution_failure_triggers_rollback():
    class FailingPolicy(DummyPolicy):
        async def activate_profile(self, profile_id: str):
            raise RuntimeError("activation failed")

    service = OrchestratorService(DummySupervisor(), FailingPolicy(), DummyLearning())
    plan = [
        {"action": "fetch_supervisor_health"},
        {"action": "activate_policy_profile", "params": {"profile_id": "demo"}},
    ]
    result = await service.execute_plan(plan)
    assert result["success"] is False
    assert result["rollback"]["rolled_back_steps"] == 1
