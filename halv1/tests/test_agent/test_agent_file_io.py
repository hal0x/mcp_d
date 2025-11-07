from pathlib import Path
from types import MethodType

import pytest

from agent import AgentCore
from planner import PlanStep, Tool


def make_core() -> AgentCore:
    core = object.__new__(AgentCore)
    core._read_file = MethodType(AgentCore._read_file, core)  # type: ignore[method-assign]
    core._write_file = MethodType(AgentCore._write_file, core)  # type: ignore[method-assign]
    return core


def test_write_then_read(tmp_path: Path) -> None:
    core = make_core()
    path = tmp_path / "data.txt"

    write_step = PlanStep(tool=Tool.FILE_IO, content=f"write {path}\ntext")
    result = core._execute_file_io(write_step)
    assert result["stdout"] == f"wrote {path}"
    assert path.read_text() == "text"

    read_step = PlanStep(tool=Tool.FILE_IO, content=f"read {path}")
    result = core._execute_file_io(read_step)
    assert result["stdout"] == "text"


def test_unknown_verb(tmp_path: Path) -> None:
    core = make_core()
    path = tmp_path / "data.txt"
    step = PlanStep(tool=Tool.FILE_IO, content=f"delete {path}")
    with pytest.raises(ValueError):
        core._execute_file_io(step)
