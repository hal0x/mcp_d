from __future__ import annotations

from pydantic import BaseModel
from ..services import example_service
from mcp.server.fastmcp import FastMCP


# Tools are registered when this module is imported by server.py
_mcp = None  # replaced by FastMCP instance during import by server


class EchoArgs(BaseModel):
    message: str


def bind(mcp: FastMCP) -> None:
    global _mcp
    _mcp = mcp

    @_mcp.tool()
    def example_echo(args: EchoArgs) -> dict:
        """Echo the provided message back to the caller."""
        return {"echo": example_service.echo(args.message)}


# Expose symbol for import side-effect in server.py
def example_echo(_: EchoArgs) -> dict:  # type: ignore[override]
    raise RuntimeError("example_echo should be registered via tools.bind(mcp)")
