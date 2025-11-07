"""Simple agent orchestrating planning and execution."""

from __future__ import annotations

import asyncio
import logging
from typing import List

from executor import CodeExecutor, ExecutionResult
from internet import SearchClient
from memory import MemoryServiceAdapter
from planner import Plan, PlanStep, TaskPlanner, Tool
from utils.artifacts import write_artifact_files
from utils.performance import measure_time_async, measure_context_async

logger = logging.getLogger(__name__)


class Agent:
    """Coordinates request handling using a planner and executor."""

    def __init__(
        self,
        planner: TaskPlanner,
        executor: CodeExecutor,
        memory: MemoryServiceAdapter,
        search: SearchClient | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.search = search or SearchClient()

    @measure_time_async("agent_run_step")
    async def _run_step(self, step: PlanStep, max_retries: int) -> List[str]:
        tool = step.tool if isinstance(step.tool, Tool) else Tool(step.tool)
        results: List[str] = []
        for attempt in range(1, max_retries + 1):
            try:
                if tool is Tool.CODE:
                    async with measure_context_async("code_execution"):
                        exec_res: ExecutionResult = self.executor.execute(step.content)
                    if exec_res.stdout:
                        results.append(exec_res.stdout)
                    if exec_res.stderr:
                        results.append(exec_res.stderr)
                    async with measure_context_async("write_artifacts"):
                        write_artifact_files(exec_res.files)
                    results.extend(f"wrote {name}" for name in exec_res.files)
                elif tool is Tool.SEARCH:
                    if step.content.startswith("http://") or step.content.startswith(
                        "https://"
                    ):
                        async with measure_context_async("http_fetch"):
                            content = await self.search.fetch_async(step.content)
                        results = [content]
                    else:
                        async with measure_context_async("search_query"):
                            results = await self.search.search_and_summarize(step.content)
                break
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("step failed on attempt %s/%s", attempt, max_retries)
                if attempt == max_retries:
                    results = [str(exc)]
        return results

    @measure_time_async("agent_handle_request")
    async def handle_request(
        self, request: str, max_iterations: int = 5, max_retries: int = 3
    ) -> List[str]:
        """Process ``request`` through iterative planning and execution.

        Each plan is executed step-by-step. Results are stored in memory and used
        to refine the request via :meth:`planner.refine`. The loop terminates
        when the planner signals completion or ``max_iterations`` is reached.
        """

        current_request = request
        context: List[str] = []
        for iteration in range(max_iterations):
            async with measure_context_async(f"planning_iteration_{iteration}"):
                plan: Plan = self.planner.plan(
                    current_request, context=context, previous_results=self.memory.recall()
                )
            step_results: List[str] = []
            for step_idx, step in enumerate(plan.steps):
                async with measure_context_async(f"step_execution_{step_idx}_{step.tool}"):
                    results: List[str] = []
                    for attempt in range(1, max_retries + 1):
                        try:
                            results = await self._run_step(step, max_retries)
                            break
                        except Exception as exc:  # pragma: no cover - runtime safety
                            logger.exception(
                                "step failed on attempt %s/%s", attempt, max_retries
                            )
                            if attempt == max_retries:
                                results = [str(exc)]
                    for res in results:
                        self.memory.remember(res)
                        step_results.append(res)
            context = list(plan.context)
            new_request = self.planner.refine(current_request, step_results)
            if new_request is None:
                break
            current_request = new_request
        return self.memory.recall()

    def handle_request_sync(
        self, request: str, max_iterations: int = 5, max_retries: int = 3
    ) -> List[str]:
        return asyncio.run(self.handle_request(request, max_iterations, max_retries))
