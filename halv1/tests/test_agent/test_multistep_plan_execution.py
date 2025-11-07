from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent.multistep import MultiStepPlanMixin
from planner import Plan, PlanStep
from planner.dag_executor import ArtifactCache, FileCache
from tools import Tool
from tools.registry import ArtifactDict, ToolRegistry


@dataclass
class _DockerExecutorStub:
    def execute_multi_step(self, _codes: list[str]) -> None:  # pragma: no cover - defensive
        raise AssertionError("docker executor should not be invoked")


class _Harness(MultiStepPlanMixin):
    def __init__(self, registry: ToolRegistry, cache: ArtifactCache) -> None:
        self.docker_executor = _DockerExecutorStub()
        self.registry = registry
        self.cache = cache


def test_plan_with_external_tool_uses_real_handler(tmp_path) -> None:
    registry = ToolRegistry()
    cache = ArtifactCache(FileCache(tmp_path))
    calls = {"count": 0}

    async def search_handler(step: PlanStep) -> ArtifactDict:
        calls["count"] += 1
        return {"stdout": f"real search: {step.content}", "stderr": "", "files": {}}

    registry.register(Tool.SEARCH, search_handler)

    plan = Plan(steps=[PlanStep(tool=Tool.SEARCH, content="python news")], context=[])
    harness = _Harness(registry, cache)

    executed, errors = asyncio.run(harness._execute_multi_step_plan(plan))

    assert errors == []
    assert executed[0][1]["stdout"] == "real search: python news"
    assert calls["count"] == 1
