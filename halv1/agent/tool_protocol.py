from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Dict, Sequence, cast

from pydantic import BaseModel, Field, ValidationError

from core.utils.json_io import parse_llm_json
from llm.base_client import LLMClient
from llm.utils import unwrap_response
from planner import PlanStep
from tools import Tool
from tools.registry import ToolRegistry


class ToolCall(BaseModel):
    """Structured request for a tool invocation from the LLM."""

    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    """Final response produced by the LLM when no more tools are needed."""

    final_answer: str


def _model_validate(model: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """Validate ``data`` using ``model`` supporting Pydantic v1/v2."""

    if hasattr(model, "model_validate"):
        return model.model_validate(data)
    return model.parse_obj(data)


def try_parse_tool_call(text: str) -> ToolCall | FinalAnswer | None:
    """Attempt to parse ``text`` into :class:`ToolCall` or :class:`FinalAnswer`.

    Returns ``None`` if parsing fails or the structure is invalid.
    """

    data = parse_llm_json(text)
    if data is None:
        return None
    for model in (ToolCall, FinalAnswer):
        try:
            return _model_validate(model, data)  # type: ignore[return-value]
        except ValidationError:
            continue
    return None


def agent_step(
    llm: LLMClient,
    registry: ToolRegistry,
    prompt: str,
    *,
    max_retries: int = 3,
    max_steps: int = 5,
) -> str:
    """Run an interactive agent loop until a final answer is produced.

    The LLM is expected to respond in JSON either as a ``ToolCall`` or a
    ``FinalAnswer``.  If a response cannot be parsed, the prompt is augmented
    with clarification and the request is retried up to ``max_retries`` times
    per step.  Tool handlers are looked up in ``registry``.
    """

    messages: Sequence[str] = [prompt]
    for _ in range(max_steps):
        current_prompt = "\n".join(messages)
        for _ in range(max_retries):
            reply = llm.generate(current_prompt)
            text_reply, _ = unwrap_response(reply)
            parsed = try_parse_tool_call(text_reply)
            if parsed is None:
                messages = list(messages) + [
                    'Respond in JSON as {"tool": "<tool>", "arguments": {...}} '
                    'or {"final_answer": "<text>"}.',
                ]
                current_prompt = "\n".join(messages)
                continue
            if isinstance(parsed, FinalAnswer):
                validated = cast(
                    FinalAnswer, _model_validate(FinalAnswer, parsed.model_dump())
                )
                return validated.final_answer
            try:
                tool_enum = Tool(parsed.tool)
                args_input = parsed.arguments
            except ValueError:
                raise ValueError(f"Unknown tool: {parsed.tool}")
            model = registry.get_model(tool_enum)
            args_model: Any = None
            if model is not None:
                try:
                    args_model = _model_validate(model, args_input)
                except ValidationError as exc:
                    messages = list(messages) + [
                        f"Invalid arguments for tool {parsed.tool}: {exc}"
                    ]
                    current_prompt = "\n".join(messages)
                    continue
            content: str
            if args_model is not None:
                if tool_enum is Tool.CODE:
                    content = args_model.code
                elif tool_enum is Tool.SEARCH:
                    content = args_model.query
                elif tool_enum is Tool.FILE_IO:
                    if args_model.operation == "read":
                        content = f"read {args_model.path}"
                    else:
                        body = args_model.content or ""
                        content = f"write {args_model.path}\n{body}"
                elif tool_enum is Tool.SHELL:
                    content = args_model.command
                elif tool_enum is Tool.HTTP:
                    first = f"{args_model.method} {args_model.url}"
                    body = args_model.body or ""
                    content = f"{first}\n{body}"
                else:  # pragma: no cover - fallback
                    content = json.dumps(args_model.model_dump())
            else:
                content = json.dumps(parsed.arguments)
            step = PlanStep(tool=tool_enum, content=content)
            handler = registry.get(tool_enum)
            result = handler(step)
            if inspect.isawaitable(result):
                result = asyncio.run(result)  # type: ignore[arg-type]
            if not isinstance(result, list):
                raise ValueError("Tool handlers must return a list of strings")
            messages = list(messages) + ["\n".join(result)]
            break
        else:
            raise ValueError("LLM failed to produce a valid response")
    raise ValueError("Maximum number of steps exceeded without final answer")
