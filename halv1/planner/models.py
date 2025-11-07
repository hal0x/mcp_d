"""Pydantic models for planner schemas.

Separated from planner.task_planner to reduce file size and isolate schema concerns.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field

from tools import Tool


class ConditionExprModel(BaseModel):
    """Pydantic representation of an expression condition."""

    model_config = ConfigDict(extra="forbid")

    expr: str
    engine: str = "jmespath"


class PlanStepModel(BaseModel):
    """Pydantic representation of a plan step from the LLM."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    tool: Tool
    inputs: dict[str, Any] = Field(default_factory=dict)
    content: str
    preconditions: list[str | ConditionExprModel] = Field(default_factory=list)
    postconditions: list[str | ConditionExprModel] = Field(default_factory=list)
    depends_on: List[int] = Field(default_factory=list)
    completion: str | None = None
    # Store policy as raw string when provided (e.g., "auto")
    policy: str | None = None
    # Allow empty expected_output and do not enforce min length
    expected_output: str = Field(
        default="",
        description="Ожидаемый результат выполнения шага",
    )
    is_final: bool = Field(default=False, description="Является ли этот шаг финальным")


class PlanModel(BaseModel):
    """Pydantic representation of a full plan."""

    model_config = ConfigDict(extra="ignore")

    steps: List[PlanStepModel] = Field(default_factory=list)
    task_completion_criteria: str = Field(default="", description="Критерии завершения всей задачи")
    requires_all_steps: bool = Field(default=True, description="Требует ли задача выполнения всех шагов")

