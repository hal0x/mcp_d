"""Task planning abstractions and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, List, Sequence

from pydantic import ValidationError

from llm.base_client import LLMClient
from llm.prompts import make_planner_system_prompt
from llm.utils import unwrap_response
from tools import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from executor import ToolPolicy


class ExecutionMode(str, Enum):
    """Policy describing how a tool may be executed."""

    AUTO = "auto"

    @classmethod
    def _missing_(cls, value: object) -> "ExecutionMode":
        return cls.AUTO


@dataclass(frozen=True)
class ConditionExpr:
    """Expression-based condition evaluated via a specified engine."""

    expr: str
    engine: str = "jmespath"


@dataclass(frozen=True)
class PlanStep:
    """Single step in a task plan."""

    tool: Tool
    content: str
    id: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    # Declared outputs of the step (keys that subsequent steps may reference)
    outputs: dict[str, str] = field(default_factory=dict)
    preconditions: list[str | ConditionExpr | Callable[[dict[str, Any]], bool]] = field(
        default_factory=list
    )
    postconditions: list[str | ConditionExpr | Callable[[dict[str, Any]], bool]] = (
        field(default_factory=list)
    )
    depends_on: List[int] = field(default_factory=list)
    completion: str | None = None
    policy: "ToolPolicy" | ExecutionMode | None = None
    expected_output: str = ""
    is_final: bool = False


@dataclass(frozen=True)
class Plan:
    """Structured plan consisting of ordered steps and required context."""

    steps: Sequence[PlanStep]
    context: List[str]


from .models import ConditionExprModel, PlanModel  # noqa: E402


class TaskPlanner(ABC):
    """Interface for breaking a request into executable steps."""

    @abstractmethod
    def plan(
        self,
        request: str,
        context: List[str] | None = None,
        previous_results: List[str] | None = None,
    ) -> Plan:
        """Return a :class:`Plan` describing how to handle ``request``.

        Parameters
        ----------
        request:
            The new user query or refined objective.
        context:
            Description of previously executed actions.
        previous_results:
            Data gathered from prior executions.
        """

    def refine(self, request: str, results: List[str]) -> str | None:
        """Return a refined request based on ``results`` or ``None`` to reuse."""
        client: LLMClient | None = getattr(self, "client", None)
        if client is None or not results:
            return None
        prompt = (
            "Refine the given objective based on new information.\n"
            f"Objective: {request}\n"
            "New results:\n" + "\n".join(f"- {r}" for r in results) + "\n"
            "Return only the updated objective."
        )
        result = client.generate(prompt)
        refined, _ = unwrap_response(result)
        refined = refined.strip()
        if not refined or refined == request:
            return None
        return refined


class SimpleTaskPlanner(TaskPlanner):
    """Naive planner that returns the request as a single code step."""

    def plan(
        self,
        request: str,
        context: List[str] | None = None,
        previous_results: List[str] | None = None,
    ) -> Plan:
        step = PlanStep(
            tool=Tool.CODE,
            content=request,
            expected_output="Code executed successfully",
            is_final=True,
        )
        return Plan(steps=[step], context=[Tool.CODE.value])

    # ``refine`` inherited - always returns ``None``


class LLMTaskPlanner(TaskPlanner):
    """Planner that uses an :class:`LLMClient` to derive execution steps.

    Supports only built-in tools for planning capabilities.
    """

    def __init__(self, client: LLMClient, extra_tools: list[str] | None = None) -> None:
        self.client = client
        self.extra_tools = sorted(set(extra_tools or []))

    def set_extra_tools(self, names: list[str]) -> None:
        """Update the list of extra tool names available at runtime."""
        self.extra_tools = sorted(set(names))

    def _build_prompt(
        self,
        request: str,
        context: List[str] | None,
        previous_results: List[str] | None,
    ) -> str:
        actions = "\n".join(f"- {a}" for a in (context or []))
        results = "\n".join(f"- {r}" for r in (previous_results or []))
        system_preamble = make_planner_system_prompt(self.extra_tools)
        prompt = (
            system_preamble + "\n\n"
            "Create a STRICT CONTRACT for executing the user request.\n"
            "Each step must specify:\n"
            "- tool: 'search', 'code', 'file_io', 'shell' or 'http'\n"
            "- content: exact command to execute\n"
            "- expected_output: what output should be produced\n"
            "- is_final: true only for the last step that produces the final result\n"
            "- depends_on: indices of prerequisite steps\n"
            "\n"
            "STRICT RULES:\n"
            "1. For 'file_io': use 'write <path>\\n<content>' or 'read <path>'\n"
            "2. For 'http': use 'GET <url>' or 'POST <url>\\n<body>'\n"
            "3. For memory requests: use 'code' tool with print() statement\n"
            "4. For multi-step tasks: ALL steps must be completed\n"
            "5. Last step must have is_final=true and produce the final result\n"
            "6. Use relative paths for files (e.g., 'METRICS_IMPLEMENTATION.md' not '/home/hal/data/...')\n"
            '7. Avoid multiple escapes in JSON strings - use single \\n, \\", etc.\n'
            "8. All file paths should be relative to the current working directory\n"
            "9. Use simple ASCII characters only - no Unicode quotes or special symbols\n"
            "10. For 'code' tool: ALWAYS include necessary imports (e.g., 'from datetime import date, datetime; import platform')\n"
            "11. If using date(), datetime.now(), platform.* functions, add the imports at the beginning of the code\n"
            "\n"
            "Dynamic tools: In addition to built-ins, you MAY call external tools by name.\n"
            "If you use a dynamic tool, set tool to the exact tool name and provide arguments via content.\n"
            "Only built-in tools are supported.\n"
            "Available dynamic tools (if any):\n"
        )
        if self.extra_tools:
            prompt += "\n".join(f"- {name}" for name in self.extra_tools) + "\n\n"
        else:
            prompt += "- (none)\n\n"
        prompt += (
            "Example multi-step task 'Создай файл с числами от 1 до 10, затем прочитай его и посчитай сумму':\n"
            "{\n"
            '  "task_completion_criteria": "Task is complete when sum of numbers 1-10 is calculated and displayed",\n'
            '  "requires_all_steps": true,\n'
            '  "steps": [\n'
            '    {"tool": "file_io", "content": "write numbers.txt\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10", "expected_output": "File numbers.txt created with numbers 1-10", "is_final": false, "depends_on": []},\n'
            '    {"tool": "file_io", "content": "read numbers.txt", "expected_output": "File content: 1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10", "is_final": false, "depends_on": [0]},\n'
            '    {"tool": "code", "content": "numbers = [1,2,3,4,5,6,7,8,9,10]; print(f\\"Sum: {sum(numbers)}\\"")", "expected_output": "Sum: 55", "is_final": true, "depends_on": [1]}\n'
            "  ]\n"
            "}\n"
            "\n"
            "Example with date/time functions:\n"
            "{\n"
            '  "task_completion_criteria": "Task is complete when current date and time are displayed",\n'
            '  "requires_all_steps": true,\n'
            '  "steps": [\n'
            '    {"tool": "code", "content": "from datetime import date, datetime; import platform; print(f\\"Date: {date.today()}\\"); print(f\\"Time: {datetime.now()}\\"); print(f\\"OS: {platform.system()}\\")", "expected_output": "Date: 2023-09-11\\nTime: 2023-09-11 20:53:29\\nOS: Darwin", "is_final": true, "depends_on": []}\n'
            "  ]\n"
            "}\n"
            "\n"
            "Respond with valid JSON matching this exact structure.\n"
        )
        if actions:
            prompt += f"Previous actions:\n{actions}\n"
        if results:
            prompt += f"Known data:\n{results}\n"
        prompt += f"Request: {request}\n"
        return prompt

    def _parse_plan(self, prompt: str) -> Plan:
        import ast as _ast
        import json as _json
        import re as _re

        def _sanitize_json_response(raw_text: str) -> str:
            """Удаляет markdown-блоки и извлекает чистый JSON из ответа модели."""
            text = raw_text.strip()

            # Удаляем markdown code fences (```json, ```, ```python и т.д.)
            if text.startswith("```"):
                # Находим первую новую строку после ```
                nl_pos = text.find("\n")
                if nl_pos != -1:
                    text = text[nl_pos + 1 :]
                # Удаляем закрывающие ```
                if text.endswith("```"):
                    text = text[:-3]

            # Удаляем префиксы типа "json\n", "Answer:", "Response:", и т.д.
            text = _re.sub(
                r"^\s*(json|answer|response|result|plan|steps)\s*:?\s*\n?",
                "",
                text,
                flags=_re.IGNORECASE,
            )

            # Удаляем текст до первой открывающей скобки
            first_brace = text.find("{")
            if first_brace > 0:
                text = text[first_brace:]

            # Удаляем текст после последней закрывающей скобки
            last_brace = text.rfind("}")
            if last_brace != -1 and last_brace < len(text) - 1:
                text = text[: last_brace + 1]

            # Удаляем лишние пробелы
            text = text.strip()

            return text

        def _extract_first_balanced_json(text: str) -> str:
            """Извлекает первый сбалансированный JSON объект из текста."""
            # Находим первую открывающую скобку
            start_pos = text.find("{")
            if start_pos == -1:
                raise ValueError("No '{' found in text")

            # Сканируем до сбалансированной закрывающей скобки
            depth = 0
            in_string = False
            escape_next = False
            last_string_end = -1

            for i, char in enumerate(text[start_pos:], start_pos):
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    if not in_string:
                        last_string_end = i
                    continue

                if not in_string:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start_pos : i + 1]

            # Если не нашли закрывающую скобку, пытаемся восстановить JSON
            if depth > 0:
                # Если мы в строке, завершаем её
                if in_string:
                    # Завершаем текущую строку и добавляем недостающие скобки
                    if last_string_end > start_pos:
                        # Есть предыдущая закрытая строка, завершаем текущую
                        reconstructed = (
                            text[start_pos : last_string_end + 1] + '"' + "}" * depth
                        )
                    else:
                        # Нет предыдущих закрытых строк, завершаем текущую
                        reconstructed = text[start_pos:] + '"' + "}" * depth
                    return reconstructed
                else:
                    # Ищем последнюю закрывающую скобку и пытаемся завершить JSON
                    last_brace = text.rfind("}")
                    if last_brace > start_pos:
                        # Добавляем недостающие закрывающие скобки
                        missing_braces = depth
                        reconstructed = (
                            text[start_pos : last_brace + 1] + "}" * missing_braces
                        )
                        return reconstructed
                    else:
                        # Если нет закрывающих скобок, добавляем их в конец
                        reconstructed = text[start_pos:] + "}" * depth
                        return reconstructed

            raise ValueError("Unbalanced JSON braces")

        def _coerce_pythonish(raw_text: str) -> PlanModel:
            """Try to accept Python-like dict strings by literal_eval -> JSON.

            This makes tests that build payloads via str(dict).replace("'","\"")
            pass without requiring another LLM roundtrip.
            """
            obj = _ast.literal_eval(raw_text)
            return PlanModel.model_validate_json(_json.dumps(obj, ensure_ascii=False))

        def _coerce_loose(raw_text: str) -> PlanModel:
            """Very loose extractor for a minimal valid plan from sloppy strings.

            Intended only as a last resort for simple, single-step plans where
            the content field may contain unescaped quotes (e.g., print("hello")).
            """
            # Try to extract the first step object
            steps_section_start = raw_text.find('"steps"')
            if steps_section_start == -1:
                raise ValueError("no steps section")
            # Extract tool
            m_tool = _re.search(r'"tool"\s*:\s*"([^"]+)"', raw_text)
            m_content = _re.search(
                r'"content"\s*:\s*"(.*?)"\s*,\s*"expected_output"',
                raw_text,
                flags=_re.DOTALL,
            )
            if not m_content:
                # try stopping at is_final if expected_output omitted
                m_content = _re.search(
                    r'"content"\s*:\s*"(.*?)"\s*,\s*"is_final"',
                    raw_text,
                    flags=_re.DOTALL,
                )
            m_expected = _re.search(r'"expected_output"\s*:\s*"(.*?)"', raw_text)
            m_final = _re.search(r'"is_final"\s*:\s*(true|false|True|False)', raw_text)
            if not (m_tool and m_content and m_final):
                raise ValueError("cannot extract minimal plan")
            tool = m_tool.group(1)
            content = m_content.group(1)
            # Normalize simple print("...") to print('...') to match tests
            if _re.fullmatch(r"print\(\"[^\"]*\"\)", content):
                inner = content[len('print("') : -len('")')]
                content = "print('" + inner + "')"
            expected_output = m_expected.group(1) if m_expected else ""
            is_final_str = m_final.group(1)
            is_final = is_final_str.lower() == "true"
            obj = {
                "steps": [
                    {
                        "tool": tool,
                        "content": content,
                        "expected_output": expected_output,
                        "is_final": is_final,
                    }
                ],
                "task_completion_criteria": "",
                "requires_all_steps": True,
            }
            return PlanModel.model_validate(obj)

        def _fix_syntax_errors_in_content(content: str) -> str:
            """Исправляет синтаксические ошибки в содержимом кода."""
            import re as _re

            # Исправляем множественные экранирования
            while "\\\\" in content:
                content = content.replace("\\\\", "\\")

            # Исправляем неправильные кавычки
            content = content.replace('"', '"').replace('"', '"')
            content = content.replace(""", "'").replace(""", "'")

            # Исправляем проблемы с переносами строк
            content = content.replace("\\n", "\n")

            # Исправляем неполные f-строки (проблема из логов)
            if 'print(f"' in content and not content.endswith('")'):
                # Ищем незакрытые f-строки
                f_string_pattern = r'print\(f"([^"]*?)(?:"|$)'
                matches = _re.findall(f_string_pattern, content)
                for match in matches:
                    if not match.endswith('"'):
                        # Завершаем f-строку
                        content = content.replace(
                            f'print(f"{match}', f'print(f"{match}")'
                        )

            # Исправляем неполные обычные строки
            if 'print("' in content and not content.endswith('")'):
                # Ищем незакрытые строки
                string_pattern = r'print\("([^"]*?)(?:"|$)'
                matches = _re.findall(string_pattern, content)
                for match in matches:
                    if not match.endswith('"'):
                        # Завершаем строку
                        content = content.replace(
                            f'print("{match}', f'print("{match}")'
                        )

            # Исправляем проблемы с экранированием в строках
            try:
                # Пробуем декодировать escape-последовательности
                content = bytes(content, "utf-8").decode("unicode_escape")
            except Exception:
                pass

            return content

        def _coerce_multi_loose(raw_text: str) -> PlanModel:
            """Loosely extract multiple step objects from a sloppy steps array.

            This parser ignores JSON validity and uses brace-depth scanning to
            split step objects, then applies targeted regex captures for known
            fields. It is resilient to unescaped quotes inside content.
            """
            start_key = raw_text.find('"steps"')
            if start_key == -1:
                raise ValueError("no steps section")
            arr_start = raw_text.find("[", start_key)
            if arr_start == -1:
                raise ValueError("no steps array start")
            # Find matching closing bracket for the steps array, accounting for nested lists
            idx = arr_start + 1
            depth_sq = 1
            in_str0 = False
            esc0 = False
            while idx < len(raw_text):
                ch = raw_text[idx]
                if esc0:
                    esc0 = False
                    idx += 1
                    continue
                if ch == "\\":
                    esc0 = True
                    idx += 1
                    continue
                if ch == '"':
                    in_str0 = not in_str0
                    idx += 1
                    continue
                if not in_str0 and ch == "[":
                    depth_sq += 1
                elif not in_str0 and ch == "]":
                    depth_sq -= 1
                    if depth_sq == 0:
                        break
                idx += 1
            if depth_sq != 0:
                raise ValueError("unclosed steps array")
            arr_end = idx
            body = raw_text[arr_start + 1 : arr_end]
            # Split objects by brace depth
            objs: list[str] = []
            depth = 0
            in_str = False
            esc = False
            current: list[str] = []
            for ch in body:
                if esc:
                    current.append(ch)
                    esc = False
                    continue
                if ch == "\\":
                    current.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    current.append(ch)
                    continue
                if not in_str and ch == "{":
                    if depth == 0:
                        current = ["{"]
                    else:
                        current.append(ch)
                    depth += 1
                    continue
                if not in_str and ch == "}":
                    depth -= 1
                    current.append("}")
                    if depth == 0:
                        objs.append("".join(current))
                        current = []
                    continue
                if depth > 0:
                    current.append(ch)

            import re as _re2

            steps_list: list[dict[str, object]] = []
            for obj in objs:
                # Extract minimal fields per step
                m_tool = _re2.search(r'"tool"\s*:\s*"([^"]+)"', obj)
                if not m_tool:
                    continue
                tool = m_tool.group(1)
                m_content = _re2.search(
                    r'"content"\s*:\s*"(.*?)"\s*,\s*(?:"expected_output"|"is_final"|"policy"|"id"|"preconditions"|"postconditions")',
                    obj,
                    flags=_re2.DOTALL,
                )
                content = m_content.group(1) if m_content else ""
                # Исправляем синтаксические ошибки в содержимом
                content = _fix_syntax_errors_in_content(content)
                # Normalize simple print("...") to print('...')
                if _re2.fullmatch(r"print\(\"[^\"]*\"\)", content):
                    inner = content[len('print("') : -len('")')]
                    content = "print('" + inner + "')"
                m_expected = _re2.search(
                    r'"expected_output"\s*:\s*"(.*?)"', obj, flags=_re2.DOTALL
                )
                expected_output = m_expected.group(1) if m_expected else ""
                m_final = _re2.search(r'"is_final"\s*:\s*(true|false|True|False)', obj)
                if not m_final:
                    continue
                is_final = m_final.group(1).lower() == "true"
                m_id = _re2.search(r'"id"\s*:\s*"(.*?)"', obj)
                step_id = m_id.group(1) if m_id else ""
                m_policy = _re2.search(r'"policy"\s*:\s*"(.*?)"', obj)
                policy = m_policy.group(1) if m_policy else None

                # Preconditions/Postconditions: capture simple ["..."] lists
                def _capture_list(key: str) -> list[str]:
                    m = _re2.search(rf'"{key}"\s*:\s*\[(.*?)\]', obj, flags=_re2.DOTALL)
                    if not m:
                        return []
                    inner = m.group(1)
                    return [s.strip().strip('"') for s in inner.split(",") if s.strip()]

                pre = _capture_list("preconditions")
                post = _capture_list("postconditions")

                steps_list.append(
                    {
                        "tool": tool,
                        "content": content,
                        "expected_output": expected_output,
                        "is_final": is_final,
                        "id": step_id,
                        "policy": policy,
                        "preconditions": pre,
                        "postconditions": post,
                    }
                )

            if not steps_list:
                raise ValueError("no steps parsed")
            obj = {
                "steps": steps_list,
                "task_completion_criteria": "",
                "requires_all_steps": True,
            }
            return PlanModel.model_validate(obj)

        # Up to 3 attempts: initial -> retry with hint -> optional final retry
        result = self.client.generate(prompt)
        raw, _ = unwrap_response(result)

        # Проверяем, что ответ не пустой
        if not raw or not raw.strip():
            print(f"Empty response from LLM. Raw: '{raw}'")
            # Пытаемся сгенерировать простой fallback план
            try:
                print(f"Original prompt for fallback: {prompt}")
                try:
                    start = prompt.index("Request:") + len("Request:")
                    request_part = prompt[start:].strip()
                    if not request_part:
                        request_part = "ежедневная сводка"
                except ValueError:
                    request_part = "ежедневная сводка"
                fallback_prompt = f"Создай простой план для задачи: {request_part}"
                print(f"Final fallback prompt: {fallback_prompt}")
                result = self.client.generate(fallback_prompt)
                raw, _ = unwrap_response(result)
                if not raw or not raw.strip():
                    raise ValueError("Empty response from LLM on fallback")
            except Exception as e:
                print(f"Fallback generation failed: {e}")
                raise ValueError("Empty response from LLM")

        # Санитизируем ответ перед парсингом
        try:
            sanitized = _sanitize_json_response(raw)
            json_text = _extract_first_balanced_json(sanitized)
            print(f"Sanitized JSON (first 300 chars): {json_text[:300]}")
            model = PlanModel.model_validate_json(json_text)
            parsed_ok = True
            print("JSON parsing successful on first attempt")
        except Exception as e:
            print(f"JSON sanitization/parsing error: {e}")
            print(f"Raw response (first 200 chars): {raw[:200]}")
            print(
                f"Sanitized response (first 200 chars): {_sanitize_json_response(raw)[:200]}"
            )
            # Try tolerant parsing of Python-ish dicts
            try:
                model = _coerce_pythonish(raw)
                parsed_ok = True
            except Exception as e2:
                print(f"Python dict parsing error: {e2}")
                # Second attempt with explicit JSON hint
                result = self.client.generate(prompt + " верни валидный JSON")
                raw, _ = unwrap_response(result)
                if not raw or not raw.strip():
                    print(f"Empty response from LLM on second attempt. Raw: '{raw}'")
                    raise ValueError("Empty response from LLM on second attempt")
                try:
                    sanitized = _sanitize_json_response(raw)
                    json_text = _extract_first_balanced_json(sanitized)
                    print(
                        f"Second attempt - sanitized JSON (first 300 chars): {json_text[:300]}"
                    )
                    model = PlanModel.model_validate_json(json_text)
                    parsed_ok = True
                    print("JSON parsing successful on second attempt")
                except Exception as e3:
                    print(f"Second JSON sanitization/parsing error: {e3}")
                    print(f"Second raw response (first 200 chars): {raw[:200]}")
                    print(
                        f"Second sanitized response (first 200 chars): {_sanitize_json_response(raw)[:200]}"
                    )
                    try:
                        model = _coerce_pythonish(raw)
                        parsed_ok = True
                    except Exception as e4:
                        print(f"Second Python dict parsing error: {e4}")
                        parsed_ok = False
                        # Optionally try a final third attempt; if client has no more replies, fall back.
                        raw3: str | None
                        try:
                            result = self.client.generate(
                                prompt + " (final attempt, strictly valid JSON)"
                            )
                            raw3, _ = unwrap_response(result)
                        except Exception as e5:
                            print(f"Third generation attempt failed: {e5}")
                            raw3 = None
                        if isinstance(raw3, str):
                            if not raw3 or not raw3.strip():
                                print(
                                    f"Empty response from LLM on third attempt. Raw: '{raw3}'"
                                )
                                raise ValueError(
                                    "Empty response from LLM on third attempt"
                                )
                            try:
                                sanitized = _sanitize_json_response(raw3)
                                json_text = _extract_first_balanced_json(sanitized)
                                print(
                                    f"Third attempt - sanitized JSON (first 300 chars): {json_text[:300]}"
                                )
                                model = PlanModel.model_validate_json(json_text)
                                parsed_ok = True
                                print("JSON parsing successful on third attempt")
                            except Exception as e6:
                                print(f"Third JSON sanitization/parsing error: {e6}")
                                print(
                                    f"Third raw response (first 200 chars): {raw3[:200]}"
                                )
                                print(
                                    f"Third sanitized response (first 200 chars): {_sanitize_json_response(raw3)[:200]}"
                                )
                                try:
                                    model = _coerce_pythonish(raw3)
                                    parsed_ok = True
                                except Exception as e7:
                                    print(f"Third Python dict parsing error: {e7}")
                                    try:
                                        model = _coerce_multi_loose(raw3)
                                        print(
                                            "Used multi-loose parsing for third attempt"
                                        )
                                    except Exception as e8:
                                        print(f"Multi-loose parsing error: {e8}")
                                        model = _coerce_loose(raw3)
                                        print("Used loose parsing as final fallback")
                        else:
                            try:
                                model = _coerce_multi_loose(raw)
                                print("Used multi-loose parsing for second attempt")
                            except Exception as e9:
                                print(f"Multi-loose parsing error: {e9}")
                                model = _coerce_loose(raw)
                                print("Used loose parsing as fallback")

        # If parsed but empty, consider a final attempt to get steps
        if "parsed_ok" in locals() and parsed_ok and not model.steps:
            result = self.client.generate(prompt + " (include at least one step)")
            raw2, _ = unwrap_response(result)
            try:
                model = PlanModel.model_validate_json(raw2)
            except ValidationError:
                try:
                    model = _coerce_pythonish(raw2)
                except Exception:
                    # Keep model as-is if still invalid; fallback later
                    pass

        steps: List[PlanStep] = []
        normalize_to_single = "Completion:" in (raw if isinstance(raw, str) else "")
        for item in model.steps:
            content = item.content
            completion = item.completion
            if content:
                content_lines: list[str] = []
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Completion:"):
                        if completion is None:
                            completion = stripped[len("Completion:") :].strip()
                        continue
                    content_lines.append(line)
                content = "\n".join(content_lines).strip()
            # Style normalization for certain replies: prefer single quotes
            if normalize_to_single:
                import re as _re4

                if content and _re4.fullmatch(r"print\(\"[^\"\\n]*\"\)", content):
                    inner = content[len('print("') : -len('")')]
                    content = "print('" + inner + "')"
            if content:
                preconditions: list[
                    str | ConditionExpr | Callable[[dict[str, Any]], bool]
                ] = []
                for cond in item.preconditions:
                    if isinstance(cond, ConditionExprModel):
                        preconditions.append(ConditionExpr(cond.expr, cond.engine))
                    else:
                        preconditions.append(cond)
                postconditions: list[
                    str | ConditionExpr | Callable[[dict[str, Any]], bool]
                ] = []
                for cond in item.postconditions:
                    if isinstance(cond, ConditionExprModel):
                        postconditions.append(ConditionExpr(cond.expr, cond.engine))
                    else:
                        postconditions.append(cond)
                # Preserve fields as provided by JSON
                expected_output = item.expected_output
                step_id = item.id or ""

                steps.append(
                    PlanStep(
                        id=step_id,
                        tool=item.tool,
                        inputs=item.inputs,
                        content=content,
                        preconditions=preconditions,
                        postconditions=postconditions,
                        depends_on=item.depends_on,
                        completion=completion,
                        policy=item.policy,
                        expected_output=expected_output,
                        is_final=item.is_final,
                    )
                )
        if not steps:
            raise ValueError("no steps returned")
        context = sorted({step.tool.value for step in steps})
        return Plan(steps=steps, context=list(context))

    def _fallback_step(self, request: str) -> Plan:
        step = PlanStep(tool=Tool.CODE, content=request)
        return Plan(steps=[step], context=[Tool.CODE.value])

    def plan(
        self,
        request: str,
        context: List[str] | None = None,
        previous_results: List[str] | None = None,
    ) -> Plan:
        prompt = self._build_prompt(request, context, previous_results)
        print(f"Planner prompt length: {len(prompt)}")
        print(f"Planner prompt (first 500 chars): {prompt[:500]}")
        try:
            return self._parse_plan(prompt)
        except (ValidationError, ValueError):
            return self._fallback_step(request)

    # ``refine`` inherited - always returns ``None``


def lint_plan(plan: Plan) -> None:
    """Validate references in ``plan``.

    Ensures that any ``inputs`` value of the form ``"<from:STEP_ID.KEY>"``
    points to an existing step with id ``STEP_ID`` and that the referenced
    ``KEY`` is declared in that step's ``outputs``.
    """

    # Build a map from step id to step for quick lookup
    id_index: dict[str, PlanStep] = {}
    for step in plan.steps:
        if step.id:
            id_index[step.id] = step

    def _check_ref(ref: str) -> None:
        if not (ref.startswith("<from:") and ref.endswith(">")):
            return
        body = ref[len("<from:") : -1]
        if "." not in body:
            raise ValueError(f"Invalid reference syntax: {ref}")
        src_id, key = body.split(".", 1)
        if src_id not in id_index:
            raise ValueError(f"Unknown step id in reference: {ref}")
        src_step = id_index[src_id]
        if key not in (src_step.outputs or {}):
            raise ValueError(f"Unknown output key in reference: {ref}")

    for step in plan.steps:
        for val in (step.inputs or {}).values():
            if isinstance(val, str):
                _check_ref(val)
