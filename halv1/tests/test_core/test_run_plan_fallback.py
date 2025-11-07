from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from planner import Plan, PlanStep
from tools import Tool
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_local_run_plan_fallback(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "planner.dag_executor":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("agent.core", None)
    core = importlib.import_module("agent.core")

    registry = ToolRegistry()

    async def handler(step: PlanStep):
        return {"stdout": str(eval(step.content))}

    registry.register(Tool.CODE, handler)
    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="1 + 1")], context=[])
    ctx = {"registry": registry}
    cache = core.ArtifactCache()

    results, errors = await core.run_plan(plan, ctx, cache)
    assert not errors
    assert results[0][1]["stdout"] == "2"
