from typing import Iterator, List

import pytest

from executor import CodeGenerator, ExecutionError
from llm.base_client import LLMClient


class DummyLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple
        return self.response

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        yield self.response


def test_code_generator_returns_safe_code() -> None:
    llm = DummyLLM("print(1 + 1)")
    generator = CodeGenerator(llm)
    code = generator.generate("add numbers")
    assert code == "print(1 + 1)"


def test_code_generator_blocks_unsafe_output() -> None:
    llm = DummyLLM("import os\nos.system('ls')")
    generator = CodeGenerator(llm)
    with pytest.raises(ExecutionError) as excinfo:
        generator.generate("list files")
    assert "import of os" in str(excinfo.value)


def test_code_generator_blocks_unsafe_description() -> None:
    llm = DummyLLM("print('hi')")
    generator = CodeGenerator(llm)
    with pytest.raises(ExecutionError) as excinfo:
        generator.generate("run rm -rf /")
    assert "Unsafe description" in str(excinfo.value)


def test_code_generator_blocks_malformed_code() -> None:
    llm = DummyLLM("for")
    generator = CodeGenerator(llm)
    with pytest.raises(ExecutionError) as excinfo:
        generator.generate("loop")
    msg = str(excinfo.value)
    assert "syntax error" in msg
    assert "invalid syntax" in msg


class FixingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            return "for"
        return "print('ok')"

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        yield self.generate(prompt)


def test_code_generator_retries_with_feedback() -> None:
    llm = FixingLLM()
    generator = CodeGenerator(llm)
    code = generator.generate("print ok")
    assert code == "print('ok')"
    assert llm.calls == 2
    assert any("invalid syntax" in p for p in llm.prompts[1:])


class AlwaysBadLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return "for"

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        yield self.generate(prompt)


def test_code_generator_stops_after_max_attempts() -> None:
    llm = AlwaysBadLLM()
    generator = CodeGenerator(llm)
    with pytest.raises(ExecutionError):
        generator.generate("loop")
    assert llm.calls == 3
