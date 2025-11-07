from __future__ import annotations

from typing import Any

from executor import CodeExecutor, ExecutionResult, ToolPolicy
from planner import PlanStep, Tool
from tools.registry import ArtifactDict
from utils.artifacts import write_artifact_files


async def run_step(
    step: PlanStep,
    executor: CodeExecutor,
    search: "SearchClient",
    ctx: dict[str, Any] | None = None,
) -> ArtifactDict:
    """Execute ``step`` using ``executor`` or ``search`` and return an artifact.

    Any files produced during execution are persisted via
    :func:`utils.artifacts.write_artifact_files` and referenced in the returned
    artifact.
    """

    tool = step.tool if isinstance(step.tool, Tool) else Tool(step.tool)

    if tool is Tool.CODE:
        policy = step.policy if isinstance(step.policy, ToolPolicy) else None
        exec_res: ExecutionResult = executor.execute(step.content, policy=policy)
        files_map = write_artifact_files(exec_res.files)
        artifact: ArtifactDict = {
            "stdout": exec_res.stdout,
            "stderr": exec_res.stderr,
            "files": files_map,
        }
    elif tool is Tool.SEARCH:
        if step.content.startswith("http://") or step.content.startswith("https://"):
            content = await search.fetch_async(step.content)
            artifact = {"stdout": content, "stderr": "", "files": {}}
        else:
            results = await search.search_and_summarize(step.content)
            artifact = {"stdout": "\n".join(results), "stderr": "", "files": {}}
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported tool: {tool}")

    if ctx is not None:
        for field in ("file_content", "numbers", "total"):
            if field in artifact:
                ctx[field] = artifact[field]

    return artifact
