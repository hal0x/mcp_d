"""Generate executable Python code from natural language descriptions."""

from __future__ import annotations

import ast

from llm.base_client import ConversationHistory, LLMClient
from llm.prompts import make_code_prompt
from llm.utils import unwrap_response

from .code_executor import ExecutionError


class CodeGenerator:
    """Use an :class:`LLMClient` to turn descriptions into Python code."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.conversation_history: ConversationHistory | None = None

    def _is_unsafe_description(self, description: str) -> bool:
        text = description.lower()
        dangerous = [
            "rm -rf",
            ":(){:|:&};:",  # fork bomb
            "del /q",
            "shutdown",
        ]
        return any(p in text for p in dangerous)

    def _assert_code_safe(self, code: str) -> None:
        # Parse a best-effort AST to catch dangerous imports/APIs
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return  # handled in main flow
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in {"os", "subprocess", "socket", "ctypes"}:
                        raise ExecutionError(f"import of {name} is not allowed")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in {"os", "subprocess", "socket", "ctypes"}:
                    raise ExecutionError(f"import of {mod} is not allowed")

    def generate(
        self, description: str, max_attempts: int = 3, *, error_reason: str = ""
    ) -> str:
        """Return Python code for ``description``.

        The underlying LLM is given up to ``max_attempts`` tries to produce
        syntactically valid code. On failure, the syntax error is fed back into
        the prompt for another attempt.

        Raises
        ------
        ExecutionError
            If syntactically valid code cannot be produced.
        """
        if self._is_unsafe_description(description):
            raise ExecutionError("Unsafe description requested")

        prompt = make_code_prompt(description, error_reason)
        last_reason = error_reason
        for attempt in range(max_attempts):
            print(f"Code generation attempt {attempt + 1}/{max_attempts}")
            raw_response = self.client.generate(prompt)
            code, history = unwrap_response(raw_response)
            if history is not None:
                self.conversation_history = history

            code = self._extract_code_from_markdown(code)
            code = self._clean_unicode_chars(code)

            # Дополнительная проверка и исправление кода
            if code and not code.strip():
                print(f"Empty code generated on attempt {attempt + 1}")
                last_reason = "Empty code generated"
                if attempt == max_attempts - 1:
                    break
                prompt = make_code_prompt(description, last_reason)
                continue
            # Safety check for dangerous imports/APIs
            self._assert_code_safe(code)

            try:
                ast.parse(code)
                print(f"Code generation successful on attempt {attempt + 1}")
                return code
            except SyntaxError as exc:
                reason = f"syntax error: {exc}"
                print(f"Syntax error on attempt {attempt + 1}: {reason}")

                # Пытаемся исправить различные типы синтаксических ошибок
                fixed = None
                
                if "unterminated string literal" in reason.lower():
                    fixed = self._close_unterminated_string(code)
                    if fixed is not None:
                        try:
                            ast.parse(fixed)
                            print(f"Fixed unterminated string literal on attempt {attempt + 1}")
                            return fixed
                        except SyntaxError as exc2:
                            reason = f"syntax error: {exc2}"
                            print(f"Still syntax error after fix: {reason}")
                            code = fixed
                
                elif "unindent does not match any outer indentation level" in reason.lower():
                    fixed = self._fix_indentation(code)
                    if fixed is not None:
                        try:
                            ast.parse(fixed)
                            print(f"Fixed indentation error on attempt {attempt + 1}")
                            return fixed
                        except SyntaxError as exc2:
                            reason = f"syntax error: {exc2}"
                            print(f"Still syntax error after indentation fix: {reason}")
                            code = fixed
                
                last_reason = reason or "Invalid code generated"
            if attempt == max_attempts - 1:
                break
            prompt = make_code_prompt(description, last_reason)

        print(f"Code generation failed after {max_attempts} attempts: {last_reason}")
        raise ExecutionError(last_reason or "code generation failed")

    @staticmethod
    def _fix_indentation(code: str) -> str | None:
        """Исправляет ошибки отступов в коде."""
        lines = code.split('\n')
        fixed_lines = []
        indent_stack = [0]  # Стек уровней отступов
        
        for i, line in enumerate(lines):
            if not line.strip():  # Пустая строка
                fixed_lines.append('')
                continue
                
            # Подсчитываем текущий отступ
            current_indent = len(line) - len(line.lstrip())
            
            # Если строка не пустая, но отступ меньше ожидаемого
            if current_indent < indent_stack[-1]:
                # Находим подходящий уровень отступа из стека
                for level in reversed(indent_stack):
                    if current_indent >= level:
                        # Исправляем отступ
                        fixed_line = ' ' * level + line.lstrip()
                        fixed_lines.append(fixed_line)
                        break
                else:
                    # Если не нашли подходящий уровень, используем минимальный
                    fixed_line = ' ' * indent_stack[0] + line.lstrip()
                    fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
                # Обновляем стек отступов
                if line.strip().endswith(':'):
                    # Новая блок - добавляем уровень отступа
                    indent_stack.append(current_indent + 4)
                elif current_indent < indent_stack[-1]:
                    # Выход из блока - убираем уровни отступов
                    while indent_stack and current_indent < indent_stack[-1]:
                        indent_stack.pop()
        
        fixed_code = '\n'.join(fixed_lines)
        
        # Проверяем, что код стал синтаксически корректным
        try:
            ast.parse(fixed_code)
            return fixed_code
        except SyntaxError:
            return None

    @staticmethod
    def _close_unterminated_string(code: str) -> str | None:
        """Close unbalanced quotes in ``code`` if possible."""
        import re

        def _insert_quote(base: str, quote: str) -> str:
            stripped = base.rstrip()
            tail = base[len(stripped) :]
            if stripped and stripped[-1] in ")]}:,":
                return stripped[:-1] + quote + stripped[-1] + tail
            return stripped + quote + tail

        # Сначала попробуем найти незакрытые строки по паттернам
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                fixed_lines.append(line)
                continue

            if "'''" in line or '"""' in line:
                fixed_lines.append(line)
                continue

            if line.count("'") % 2 == 1:
                line = _insert_quote(line, "'")
            if line.count('"') % 2 == 1:
                line = _insert_quote(line, '"')

            fixed_lines.append(line)
        
        fixed = '\n'.join(fixed_lines)
        
        # Дополнительная проверка для многострочных строк
        if fixed.count("'''") % 2 == 1:
            fixed = _insert_quote(fixed, "'''")
        elif fixed.count('"""') % 2 == 1:
            fixed = _insert_quote(fixed, '"""')
        else:
            single = fixed.count("'") - 3 * fixed.count("'''")
            double = fixed.count('"') - 3 * fixed.count('"""')
            if single % 2 == 1:
                fixed = _insert_quote(fixed, "'")
            if double % 2 == 1:
                fixed = _insert_quote(fixed, '"')
        
        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            return None

    # _create_robust_prompt is now centralized in llm.prompts.make_code_prompt
    def _create_robust_prompt(self, description: str, error_reason: str = "") -> str:
        return make_code_prompt(description, error_reason)

    def _extract_code_from_markdown(self, code: str) -> str:
        """Извлекает код из markdown блоков если они есть."""
        code = code.strip()

        # Удаляем блоки ```python ... ```
        if "```python" in code:
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        # Удаляем блоки ``` ... ```
        elif code.startswith("```") and code.endswith("```"):
            lines = code.split("\n")
            if len(lines) > 2:
                code = "\n".join(lines[1:-1])

        return code.strip()

    def _clean_unicode_chars(self, code: str) -> str:
        """Удаляет проблемные Unicode символы и заменяет их ASCII аналогами."""
        # Словарь замен проблемных символов
        replacements = {
            "→": " = ",  # Стрелка на присваивание
            "–": "-",  # Длинное тире на минус
            "—": "-",  # Еще более длинное тире
            "‘": "'",  # Левая одинарная кавычка
            "’": "'",  # Правая одинарная кавычка
            "“": '"',  # Левая двойная кавычка
            "”": '"',  # Правая двойная кавычка
            "«": '"',  # Левая елочка
            "»": '"',  # Правая елочка
            "\u202f": " ",  # Неразрывный узкий пробел
            "\u2011": "-",  # Неразрывный дефис
        }

        # Применяем замены
        for unicode_char, ascii_replacement in replacements.items():
            code = code.replace(unicode_char, ascii_replacement)

        # Удаляем все символы не из ASCII диапазона кроме пробелов и переносов строк
        cleaned = ""
        for char in code:
            if ord(char) < 128 or char in ["\n", "\r", "\t"]:
                cleaned += char
            else:
                # Заменяем неизвестные Unicode символы пробелом
                cleaned += " "

        return cleaned
