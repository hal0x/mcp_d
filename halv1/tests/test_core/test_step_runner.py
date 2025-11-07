from __future__ import annotations

import base64
from pathlib import Path
from typing import List, cast

import pytest

from core.step_runner import run_step
from executor import CodeExecutor, ExecutionResult, ToolPolicy
from internet import SearchClient
from planner import PlanStep, Tool


class DummyExecutor(CodeExecutor):
    def __init__(self, result: ExecutionResult) -> None:
        self.result = result

    def execute(self, code: str, policy: ToolPolicy | None = None) -> ExecutionResult:
        return self.result


class DummySearch:
    def __init__(
        self, fetch: str | None = None, results: List[str] | None = None
    ) -> None:
        self.fetch = fetch
        self.results = results or []

    async def fetch_async(self, url: str) -> str:
        return self.fetch or ""

    async def search_and_summarize(self, query: str) -> List[str]:
        return self.results


@pytest.mark.asyncio
async def test_run_step_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exec_result = ExecutionResult(
        stdout="out", stderr="err", files={"a.txt": b"data"}, returncode=0
    )
    executor = DummyExecutor(exec_result)
    search = cast(SearchClient, DummySearch())
    monkeypatch.chdir(tmp_path)
    step = PlanStep(tool=Tool.CODE, content="print('hi')")
    artifact = await run_step(step, executor, search)
    assert artifact == {
        "stdout": "out",
        "stderr": "err",
        "files": {"a.txt": base64.b64encode(b"data").decode()},
    }
    assert (tmp_path / "a.txt").read_bytes() == b"data"


@pytest.mark.asyncio
async def test_run_step_search_query() -> None:
    executor = DummyExecutor(
        ExecutionResult(stdout="", stderr="", files={}, returncode=0)
    )
    search = cast(SearchClient, DummySearch(results=["r1", "r2"]))
    step = PlanStep(tool=Tool.SEARCH, content="query")
    artifact = await run_step(step, executor, search)
    assert artifact == {"stdout": "r1\nr2", "stderr": "", "files": {}}


@pytest.mark.asyncio
async def test_run_step_search_url() -> None:
    executor = DummyExecutor(
        ExecutionResult(stdout="", stderr="", files={}, returncode=0)
    )
    search = cast(SearchClient, DummySearch(fetch="page"))
    step = PlanStep(tool=Tool.SEARCH, content="http://example.com")
    artifact = await run_step(step, executor, search)
    assert artifact == {"stdout": "page", "stderr": "", "files": {}}
