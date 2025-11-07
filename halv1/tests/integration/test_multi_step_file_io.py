import json
import os
from unittest.mock import Mock

import pytest

from agent.core import AgentCore
from planner import Plan, PlanStep, Tool
from .docker_utils import skip_if_docker_unavailable


@pytest.mark.asyncio
async def test_multi_step_file_io_and_sum(docker_executor, docker_workdir):
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.FILE_IO,
                content="write numbers.txt\n1\n2\n3",
                expected_output="file created",
            ),
            PlanStep(
                tool=Tool.FILE_IO,
                content="read numbers.txt",
                expected_output="1\n2\n3",
            ),
            PlanStep(
                tool=Tool.CODE,
                content=(
                    "numbers=[int(x) for x in open('numbers.txt').read().split()]\n"
                    "print(f'Sum: {sum(numbers)}')"
                ),
                expected_output="Sum: 6",
                is_final=True,
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    assert getattr(core, "docker_executor", None) is docker_executor

    executed, errors = await core._execute_multi_step_plan(plan)

    assert not errors
    outputs = [artifact["stdout"] for _, artifact in executed]
    assert outputs == ["file created", "1\n2\n3", "Sum: 6"]
    for _, artifact in executed:
        files = artifact["files"]
        assert isinstance(files, dict)
        json.dumps(files)
        for path, content in files.items():
            assert isinstance(content, str)
            assert not os.path.isabs(path)

    assert executed[0][1]["files"] == {}
    assert executed[1][1]["files"] == {}
    assert "numbers.txt" in executed[2][1]["files"]


@pytest.mark.asyncio
async def test_multi_step_captures_stderr(docker_executor, docker_workdir):
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.CODE,
                content="import sys\nprint('warning!', file=sys.stderr)",
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    executed, errors = await core._execute_multi_step_plan(plan)

    assert errors == [(0, "warning!")]
    assert executed[0][1]["stdout"] == ""
    assert executed[0][1]["stderr"] == "warning!"


@pytest.mark.asyncio
async def test_multi_step_file_io_files_only_on_final_step(docker_executor, docker_workdir):
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.FILE_IO,
                content="write numbers.txt\n1\n2\n3",
                expected_output="file created",
            ),
            PlanStep(
                tool=Tool.FILE_IO,
                content="read numbers.txt",
                expected_output="1\n2\n3",
            ),
            PlanStep(
                tool=Tool.CODE,
                content=(
                    "numbers=[int(x) for x in open('numbers.txt').read().split()]\n"
                    "print(f'Sum: {sum(numbers)}')"
                ),
                expected_output="Sum: 6",
                is_final=True,
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    executed, errors = await core._execute_multi_step_plan(plan)

    assert not errors
    assert executed[0][1]["files"] == {}
    assert executed[1][1]["files"] == {}
    assert executed[2][1]["files"] == executed[-1][1]["files"]
    assert "numbers.txt" in executed[-1][1]["files"]


@pytest.mark.asyncio
async def test_multi_step_file_io_detects_corrupted_file(docker_executor, docker_workdir):
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.FILE_IO,
                content="write numbers.txt\n1\n2\n3",
                expected_output="file created",
            ),
            PlanStep(
                tool=Tool.CODE,
                content=(
                    "from pathlib import Path\n"
                    "Path('numbers.txt').write_text('1\\n999\\n3', encoding='utf-8')"
                ),
            ),
            PlanStep(
                tool=Tool.CODE,
                content="print('corruption complete')",
            ),
            PlanStep(
                tool=Tool.FILE_IO,
                content="read numbers.txt",
                expected_output="1\n2\n3",
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    executed, errors = await core._execute_multi_step_plan(plan)

    assert not errors
    assert len(executed) == 3
    outputs = [artifact["stdout"] for _, artifact in executed]
    assert outputs == ["file created", "Pretend numbers are listed here", "Sum: 6"]
    assert executed[1][1]["stdout"] != "1\n2\n3"


@pytest.mark.asyncio
async def test_multi_step_file_io_special_characters(docker_executor, docker_workdir):
    special_content = 'Line with triple quotes """ and backslash \\\nSecond line with "quotes"'
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.FILE_IO,
                content=f"write special.txt\n{special_content}",
                expected_output="special file created",
            ),
            PlanStep(
                tool=Tool.FILE_IO,
                content="read special.txt",
            ),
            PlanStep(
                tool=Tool.CODE,
                content="print(open('special.txt', encoding='utf-8').read())",
                expected_output=special_content,
                is_final=True,
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    executed, errors = await core._execute_multi_step_plan(plan)

    assert not errors
    outputs = [artifact["stdout"] for _, artifact in executed]
    assert outputs == ["special file created", special_content, special_content]


@pytest.mark.asyncio
async def test_multi_step_file_io_detects_corrupted_file(docker_executor, docker_workdir):
    plan = Plan(
        steps=[
            PlanStep(
                tool=Tool.FILE_IO,
                content="write numbers.txt\n1\n2\n3",
                expected_output="file created",
            ),
            PlanStep(
                tool=Tool.CODE,
                content="open('numbers.txt', 'w').write('1\\n999\\n3')",
                expected_output="",
            ),
            PlanStep(
                tool=Tool.FILE_IO,
                content="read numbers.txt",
                expected_output="1\n2\n3",
            ),
        ],
        context=[],
    )

    skip_if_docker_unavailable()

    core = AgentCore(
        bus=Mock(),
        planner=Mock(),
        executor=docker_executor,
        search=Mock(),
        memory=Mock(),
        code_generator=Mock(),
    )

    executed, errors = await core._execute_multi_step_plan(plan)

    # The corruption step should surface via a failing read assertion.
    assert errors
    error_map = {idx: message for idx, message in errors}
    assert 2 in error_map
    assert "Expected content mismatch" in error_map[2]

    # Stdout captures the actual file contents even on failure.
    read_stdout = executed[2][1]["stdout"]
    assert read_stdout == "1\n999\n3"
    assert executed[2][1]["stderr"] == error_map[2]
