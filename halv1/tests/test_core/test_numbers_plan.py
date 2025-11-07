from pathlib import Path
from typing import Any

import pytest

from planner import Plan, PlanStep
from planner.dag_executor import ArtifactCache, FileCache, run_plan
from tools import Tool
from tools.registry import ArtifactDict, ToolRegistry


@pytest.mark.asyncio
async def test_numbers_workflow(tmp_path: Path) -> None:
    registry = ToolRegistry()
    path = tmp_path / "numbers.txt"
    path.write_text("\n".join(str(i) for i in range(1, 11)))

    async def handler(step: PlanStep) -> ArtifactDict:
        if step.content == "parse":
            file_content = path.read_text()
            numbers = [int(line) for line in file_content.splitlines() if line]
            return {
                "stdout": "",
                "stderr": "",
                "files": {},
                "file_content": file_content,
                "numbers": numbers,
            }
        if step.content == "sum":
            total = sum(int(line) for line in path.read_text().splitlines() if line)
            return {"stdout": str(total), "stderr": "", "files": {}, "total": total}
        raise ValueError("unknown step")

    registry.register(Tool.CODE, handler)
    cache = ArtifactCache(FileCache(tmp_path))

    def parse_post(ctx: dict[str, Any]) -> bool:
        nums = ctx.get("numbers")
        return (
            isinstance(nums, list)
            and all(isinstance(n, int) for n in nums)
            and len(nums) == 10
        )

    def sum_post(ctx: dict[str, Any]) -> bool:
        numbers = ctx.get("numbers") or []
        total = ctx.get("total")
        return isinstance(total, int) and total == sum(numbers) and total > max(numbers)

    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="parse", postconditions=[parse_post]),
            PlanStep(
                tool=Tool.CODE,
                content="sum",
                depends_on=["0"],
                postconditions=[sum_post],
            ),
        ],
        context=[],
    )

    ctx = {"registry": registry}
    results, errors = await run_plan(plan, ctx, cache)
    assert not errors
    parse_artifact = results[0][1]
    sum_artifact = results[1][1]
    assert parse_artifact["numbers"] == list(range(1, 11))
    assert sum_artifact["total"] == 55

    # idempotent run
    calls = {"count": 0}

    async def counting_handler(step: PlanStep) -> ArtifactDict:
        calls["count"] += 1
        return await handler(step)

    registry.register(Tool.CODE, counting_handler)
    ctx2 = {"registry": registry}
    await run_plan(plan, ctx2, cache)
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_run_plan_invalid_reference(tmp_path: Path) -> None:
    registry = ToolRegistry()

    async def handler(step: PlanStep) -> ArtifactDict:
        return {"stdout": "ok", "stderr": "", "files": {}}

    registry.register(Tool.CODE, handler)
    cache = ArtifactCache(FileCache(tmp_path))
    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="a", id="s1", outputs={"val": "stdout"}),
            PlanStep(
                tool=Tool.CODE,
                content="b",
                id="s2",
                inputs={"x": "<from:s1.missing>"},
            ),
        ],
        context=[],
    )
    ctx = {"registry": registry}
    with pytest.raises(ValueError):
        await run_plan(plan, ctx, cache)


def _make_plan(path: Path) -> tuple[Plan, ToolRegistry, ArtifactCache]:
    registry = ToolRegistry()
    cache = ArtifactCache(FileCache(path))

    async def handler(step: PlanStep) -> ArtifactDict:
        file_path = path / "numbers.txt"
        if step.content == "parse":
            file_content = file_path.read_text()
            numbers = [int(line) for line in file_content.splitlines() if line]
            return {
                "stdout": "",
                "stderr": "",
                "files": {},
                "file_content": file_content,
                "numbers": numbers,
            }
        if step.content == "sum":
            total = sum(
                int(line) for line in file_path.read_text().splitlines() if line
            )
            return {"stdout": str(total), "stderr": "", "files": {}, "total": total}
        raise ValueError("unknown")

    registry.register(Tool.CODE, handler)

    def parse_post(ctx: dict[str, Any]) -> bool:
        nums = ctx.get("numbers")
        return (
            isinstance(nums, list)
            and all(isinstance(n, int) for n in nums)
            and len(nums) == 10
        )

    def sum_post(ctx: dict[str, Any]) -> bool:
        numbers = ctx.get("numbers") or []
        total = ctx.get("total")
        return isinstance(total, int) and total == sum(numbers) and total > max(numbers)

    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="parse", postconditions=[parse_post]),
            PlanStep(
                tool=Tool.CODE,
                content="sum",
                depends_on=["0"],
                postconditions=[sum_post],
            ),
        ],
        context=[],
    )
    return plan, registry, cache


@pytest.mark.asyncio
async def test_missing_file(tmp_path: Path) -> None:
    plan, registry, cache = _make_plan(tmp_path)
    ctx = {"registry": registry}
    _, errors = await run_plan(plan, ctx, cache)
    assert errors and isinstance(errors[0][1], Exception)


@pytest.mark.asyncio
async def test_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "numbers.txt"
    file_path.write_text("")
    plan, registry, cache = _make_plan(tmp_path)
    ctx = {"registry": registry}
    _, errors = await run_plan(plan, ctx, cache)
    assert errors and "Postconditions failed" in str(errors[0][1])


@pytest.mark.asyncio
async def test_bad_line(tmp_path: Path) -> None:
    file_path = tmp_path / "numbers.txt"
    file_path.write_text("1\n2\na")
    plan, registry, cache = _make_plan(tmp_path)
    ctx = {"registry": registry}
    _, errors = await run_plan(plan, ctx, cache)
    assert errors and isinstance(errors[0][1], Exception)
