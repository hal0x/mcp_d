from typing import Iterator

from executor import CodeGenerator
from llm.base_client import LLMClient


class UnterminatedLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple
        self.calls += 1
        return "print('hello)"

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        yield self.generate(prompt)


def test_code_generator_fixes_unterminated_string() -> None:
    llm = UnterminatedLLM()
    generator = CodeGenerator(llm)
    code = generator.generate("greet")
    assert code == "print('hello')"
    assert llm.calls == 1
