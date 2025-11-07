"""Tests for planner.dag_executor module."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from executor import ExecutionResult
from planner import ConditionExpr, ExecutionMode, Plan, PlanStep
from planner.dag_executor import (
    Artifact,
    ArtifactCache,
    FileCache,
    cache_key,
    run_partial,
    run_plan,
    run_step,
)
from tools import Tool
from tools.registry import ArtifactDict, ToolRegistry


async def _handler(step: PlanStep) -> ArtifactDict:
    return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}


_handler.__version__ = "1"  # type: ignore[attr-defined]


def _postcondition(ctx: dict[str, Any]) -> bool:
    outputs = ctx["outputs"]
    return "test-out" in outputs.get("stdout", "")


def _is_ready(ctx: dict[str, Any]) -> bool:
    return bool(ctx.get("ready"))


class CountingFileCache(FileCache):
    """File cache backend that counts ``save`` calls."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.save_calls = 0

    def save(self, key: str, artifact: Artifact) -> None:
        self.save_calls += 1
        super().save(key, artifact)


def test_run_plan_executes_and_caches(tmp_path: Path) -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def handler(step: PlanStep) -> ArtifactDict:
        calls.append(step.content)
        return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}

    handler.__version__ = "1"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, cast(Any, handler))

    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="first"),
            PlanStep(tool=Tool.CODE, content="second", depends_on=[0]),
        ],
        context=[],
    )

    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert calls == ["first", "second"]
    assert results == [
        (0, {"stdout": "first-out", "stderr": "", "files": {}}),
        (1, {"stdout": "second-out", "stderr": "", "files": {}}),
    ]
    assert not errors

    key = cache_key(plan.steps[1], ctx)
    loaded = cache.load(key)
    assert loaded == {"stdout": "second-out", "stderr": "", "files": {}}


def test_step_with_bad_postcondition_not_cached(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    cache = ArtifactCache(FileCache(tmp_path))

    step = PlanStep(tool=Tool.CODE, content="test", postconditions=["done"])
    plan = Plan(steps=[step], context=[])
    ctx = {"registry": registry}

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert results == []
    assert len(errors) == 1 and isinstance(errors[0][1], RuntimeError)
    key = cache_key(step, ctx)
    assert cache.load(key) is None


def test_run_plan_uses_cache(tmp_path: Path) -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def handler(step: PlanStep) -> ArtifactDict:
        calls.append(step.content)
        return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}

    handler.__version__ = "1"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, cast(Any, handler))

    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="first")], context=[])
    cache = ArtifactCache(FileCache(tmp_path))

    ctx = {"registry": registry}
    asyncio.run(run_plan(plan, ctx, cache))

    ctx2 = {"registry": registry}
    asyncio.run(run_plan(plan, ctx2, cache))

    assert calls == ["first"]


def test_file_restored_from_cache(tmp_path: Path) -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def handler(step: PlanStep) -> ExecutionResult:
        calls.append(step.content)
        return ExecutionResult("", "", {"data.txt": b"hello"}, 0)

    registry.register(Tool.CODE, cast(Any, handler))

    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="gen")], context=[])
    cache = ArtifactCache(FileCache(tmp_path))

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = {"registry": registry}
        asyncio.run(run_plan(plan, ctx, cache))
        path = tmp_path / "data.txt"
        assert path.read_text() == "hello"
        path.unlink()

        ctx2 = {"registry": registry}
        asyncio.run(run_plan(plan, ctx2, cache))

        assert calls == ["gen"]
        assert path.read_text() == "hello"
    finally:
        os.chdir(cwd)


def test_run_plan_saves_once(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    backend = CountingFileCache(tmp_path)
    cache = ArtifactCache(backend)
    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="first")], context=[])
    ctx = {"registry": registry}

    asyncio.run(run_plan(plan, ctx, cache))

    assert backend.save_calls == 1


def test_cache_key_changes_with_policy(tmp_path: Path) -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def handler(step: PlanStep) -> ArtifactDict:
        calls.append(step.content)
        return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}

    handler.__version__ = "1"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, cast(Any, handler))

    cache = ArtifactCache(FileCache(tmp_path))
    plan_with_policy = Plan(
        steps=[PlanStep(tool=Tool.CODE, content="first", policy=ExecutionMode.AUTO)],
        context=[],
    )
    plan_without_policy = Plan(
        steps=[PlanStep(tool=Tool.CODE, content="first")],
        context=[],
    )

    ctx = {"registry": registry}
    asyncio.run(run_plan(plan_with_policy, ctx, cache))

    ctx2 = {"registry": registry}
    asyncio.run(run_plan(plan_without_policy, ctx2, cache))

    assert calls == ["first", "first"]


def test_run_plan_precondition_failure(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                preconditions=["ready"],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert results == []
    assert len(errors) == 1
    assert errors[0][0] == 0
    assert isinstance(errors[0][1], RuntimeError)
    assert "Preconditions not met" in str(errors[0][1])


def test_run_plan_callable_precondition(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                preconditions=[_is_ready],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry, "ready": True}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert results == [(0, {"stdout": "test-out", "stderr": "", "files": {}})]
    assert not errors


def test_run_plan_postcondition_failure(tmp_path: Path) -> None:
    registry = ToolRegistry()

    async def bad_handler(step: PlanStep) -> ArtifactDict:
        return {"stdout": "nope", "stderr": "", "files": {}}

    registry.register(Tool.CODE, bad_handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                postconditions=["done"],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert results == []
    assert len(errors) == 1
    assert errors[0][0] == 0
    assert isinstance(errors[0][1], RuntimeError)
    assert "Postconditions failed" in str(errors[0][1])


def test_run_plan_callable_postcondition(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                postconditions=[_postcondition],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache))

    assert results == [(0, {"stdout": "test-out", "stderr": "", "files": {}})]
    assert not errors


def test_run_plan_cleanup_removes_old_artifacts(tmp_path: Path) -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def handler(step: PlanStep) -> ArtifactDict:
        calls.append(step.content)
        return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}

    handler.__version__ = "1"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, cast(Any, handler))

    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="first")], context=[])
    cache = ArtifactCache(FileCache(tmp_path))
    ctx = {"registry": registry}

    asyncio.run(run_plan(plan, ctx, cache))
    key = cache_key(plan.steps[0], ctx)
    path = tmp_path / key / "artifact.json"
    old_time = datetime.now() - timedelta(days=2)
    os.utime(path, (old_time.timestamp(), old_time.timestamp()))

    asyncio.run(run_plan(plan, ctx, cache, cleanup_ttl=timedelta(days=1)))

    assert calls == ["first", "first"]
    assert path.exists()
    assert path.stat().st_mtime > old_time.timestamp()


def test_run_plan_continue_on_error_and_run_partial(tmp_path: Path) -> None:
    registry = ToolRegistry()

    async def handler(step: PlanStep) -> ArtifactDict:
        if step.content == "bad":
            raise RuntimeError("boom")
        return {"stdout": f"{step.content}-out", "stderr": "", "files": {}}

    handler.__version__ = "1"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, cast(Any, handler))

    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="root", id="0"),
            PlanStep(tool=Tool.CODE, content="bad", depends_on=[0], id="1"),
            PlanStep(tool=Tool.CODE, content="good", depends_on=[0], id="2"),
            PlanStep(tool=Tool.CODE, content="after", depends_on=[1], id="3"),
        ],
        context=[],
    )
    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))

    results, errors = asyncio.run(run_plan(plan, ctx, cache, continue_on_error=True))

    assert results == [
        (0, {"stdout": "root-out", "stderr": "", "files": {}}),
        (2, {"stdout": "good-out", "stderr": "", "files": {}}),
    ]
    assert len(errors) == 2
    assert errors[0][0] == 1 and isinstance(errors[0][1], RuntimeError)
    assert errors[1][0] == 3
    assert "Dependency failed" in str(errors[1][1])

    async def fixed(step: PlanStep) -> ArtifactDict:
        return {"stdout": f"{step.content}-fixed", "stderr": "", "files": {}}

    fixed.__version__ = "2"  # type: ignore[attr-defined]
    registry.register(Tool.CODE, fixed)

    results2, errors2 = asyncio.run(run_partial(plan, ["1"], ctx, cache))

    assert errors2 == []
    assert results2 == [
        (1, {"stdout": "bad-fixed", "stderr": "", "files": {}}),
        (3, {"stdout": "after-fixed", "stderr": "", "files": {}}),
    ]


def test_run_partial_saves_once(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    backend = CountingFileCache(tmp_path)
    cache = ArtifactCache(backend)
    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="first")], context=[])
    ctx = {"registry": registry}

    asyncio.run(run_partial(plan, [0], ctx, cache))

    assert backend.save_calls == 1


def test_run_step_checks_conditions(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    cache = ArtifactCache(FileCache(tmp_path))

    step_pre = PlanStep(tool=Tool.CODE, content="test", preconditions=["ready"])
    ctx = {"registry": registry}
    with pytest.raises(RuntimeError, match="Preconditions not met"):
        asyncio.run(run_step(step_pre, ctx, cache))

    step_post_fail = PlanStep(tool=Tool.CODE, content="test", postconditions=["done"])
    ctx2 = {"registry": registry}
    with pytest.raises(RuntimeError, match="Postconditions failed"):
        asyncio.run(run_step(step_post_fail, ctx2, cache))

    step_ok = PlanStep(
        tool=Tool.CODE,
        content="test",
        postconditions=[_postcondition],
    )
    ctx3 = {"registry": registry}
    outputs = asyncio.run(run_step(step_ok, ctx3, cache))

    assert "outputs" not in ctx3
    assert outputs == {"stdout": "test-out", "stderr": "", "files": {}}


def test_run_step_saves_to_default_cache() -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    cache = ArtifactCache()

    step = PlanStep(tool=Tool.CODE, content="direct-cache")
    ctx = {"registry": registry}

    key = cache_key(step, ctx)
    path = Path("db/artifacts") / key / "artifact.json"
    if path.exists():
        path.unlink()

    asyncio.run(run_step(step, ctx, cache))

    assert path.exists()
    path.unlink()


def test_cleanup_removes_old_artifacts(tmp_path: Path) -> None:
    cache = ArtifactCache(FileCache(tmp_path))
    artifact: Artifact = {"stdout": "x", "stderr": "", "files": {}}

    cache.save("old", artifact)
    old_path = tmp_path / "old" / "artifact.json"
    old_time = datetime.now() - timedelta(days=2)
    os.utime(old_path, (old_time.timestamp(), old_time.timestamp()))

    cache.save("new", artifact)

    cache.cleanup(ttl=timedelta(days=1))

    assert not old_path.exists()
    assert (tmp_path / "new" / "artifact.json").exists()


def test_run_plan_expr_conditions(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                preconditions=[ConditionExpr("ready.flag == `true`")],
                postconditions=[ConditionExpr("contains(outputs.stdout, 'test-out')")],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry, "ready": {"flag": True}}
    cache = ArtifactCache(FileCache(tmp_path))
    results, errors = asyncio.run(run_plan(plan, ctx, cache))
    assert results == [(0, {"stdout": "test-out", "stderr": "", "files": {}})]
    assert not errors


def test_run_plan_expr_postcondition_failure(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(Tool.CODE, _handler)
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="test",
                postconditions=[ConditionExpr("contains(outputs.stdout, 'nope')")],
            )
        ],
        context=[],
    )
    ctx = {"registry": registry}
    cache = ArtifactCache(FileCache(tmp_path))
    results, errors = asyncio.run(run_plan(plan, ctx, cache))
    assert results == []
    assert len(errors) == 1 and isinstance(errors[0][1], RuntimeError)
