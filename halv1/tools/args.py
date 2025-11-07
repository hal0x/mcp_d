from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CodeArgs(BaseModel):
    code: str


class SearchArgs(BaseModel):
    query: str


class FileIOArgs(BaseModel):
    operation: Literal["read", "write"]
    path: str
    content: str | None = None


class ShellArgs(BaseModel):
    command: str


class HttpArgs(BaseModel):
    method: Literal["GET", "POST"]
    url: str
    body: str | None = None
