from typing import Iterable

from llm.context_aware_client import ContextAwareWrapper


class DummyClient:
    """Client implementing only the old API without context support."""

    def generate(self, prompt: str) -> str:
        return f"echo: {prompt}"

    def stream(self, prompt: str) -> Iterable[str]:
        yield prompt


def test_wrapper_handles_client_without_context_parameter() -> None:
    client = DummyClient()
    wrapper = ContextAwareWrapper(client)

    assert wrapper._supports_history is False

    existing_history = [{"role": "system", "content": "init"}]
    response, history = wrapper.generate("hello", history=existing_history)
    assert response == "echo: hello"
    assert history[:1] == existing_history
    assert history[-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "echo: hello"},
    ]
