from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, TypedDict, Union

from pydantic import BaseModel

from .args import CodeArgs, FileIOArgs, HttpArgs, SearchArgs, ShellArgs
from .types import Tool

if TYPE_CHECKING:
    from planner import PlanStep


class ArtifactDict(TypedDict, total=False):
    stdout: str
    stderr: str
    files: Dict[str, str]
    file_content: str
    numbers: List[int]
    total: int


Handler = Callable[["PlanStep"], Union[ArtifactDict, Awaitable[ArtifactDict]]]


class ToolRegistry:
    """Registry mapping tools to execution handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[Tool, Handler] = {}
        self._models: Dict[Tool, type[BaseModel]] = {
            Tool.CODE: CodeArgs,
            Tool.SEARCH: SearchArgs,
            Tool.FILE_IO: FileIOArgs,
            Tool.SHELL: ShellArgs,
            Tool.HTTP: HttpArgs,
        }

    def register(
        self,
        tool: Tool,
        handler: Handler,
        *,
        model: type[BaseModel] | None = None,
    ) -> None:
        """Register ``handler`` for ``tool``."""
        self._handlers[tool] = handler
        if model is not None:
            self._models[tool] = model

    def get(self, tool: Tool) -> Handler:
        """Return handler registered for ``tool``."""
        return self._handlers[tool]

    def get_model(self, tool: Tool) -> type[BaseModel] | None:
        """Return argument model registered for ``tool`` if available."""
        return self._models.get(tool)

    def try_get(self, tool: Tool) -> Handler | None:
        """Return handler registered for ``tool`` or ``None`` if missing."""
        return self._handlers.get(tool)


def register_builtin_handlers(
    registry: "ToolRegistry",
    *,
    code: Handler,
    search: Handler,
    file_io: Handler,
    shell: Handler,
    http: Handler,
) -> None:
    """Convenience helper to register common tool handlers."""
    registry.register(Tool.CODE, code)
    registry.register(Tool.SEARCH, search)
    registry.register(Tool.FILE_IO, file_io)
    registry.register(Tool.SHELL, shell)
    registry.register(Tool.HTTP, http)
