from typing import Iterator

from planner import LLMTaskPlanner, PlanStep, Tool


class StubLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str) -> str:
        return self.reply

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        if False:
            yield ""  # unreachable placeholder


def test_llm_planner_parses_steps() -> None:
    response = (
        '{"steps": ['
        ' {"tool": "search", "content": "python\\nCompletion: info gathered"},'
        ' {"tool": "http", "content": "GET http://example.com", "completion": "fetched"},'
        ' {"tool": "code", "content": "result = 2 + 2"}'
        " ] }"
    )
    planner = LLMTaskPlanner(StubLLM(response))
    plan = planner.plan("Find info and compute")
    assert plan.context == ["code", "http", "search"]
    assert plan.steps[0] == PlanStep(
        tool=Tool.SEARCH, content="python", completion="info gathered"
    )
    assert plan.steps[1] == PlanStep(
        tool=Tool.HTTP, content="GET http://example.com", completion="fetched"
    )
    assert plan.steps[2] == PlanStep(tool=Tool.CODE, content="result = 2 + 2")


class RecordingLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.reply

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        if False:
            yield ""  # unreachable placeholder


def test_refine_invokes_model_and_changes_request() -> None:
    llm = RecordingLLM("new goal")
    planner = LLMTaskPlanner(llm)
    refined = planner.refine("initial goal", ["some result"])
    assert refined == "new goal"
    assert (
        llm.prompt is not None
        and "initial goal" in llm.prompt
        and "some result" in llm.prompt
    )


class EchoLLM:
    def generate(self, prompt: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("Objective: "):
                return line.removeprefix("Objective: ").strip()
        return ""

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        if False:
            yield ""  # unreachable placeholder


def test_refine_returns_none_when_request_unchanged() -> None:
    planner = LLMTaskPlanner(EchoLLM())
    refined = planner.refine("keep goal", ["some result"])
    assert refined is None


def test_planner_prompt_mentions_http() -> None:
    llm = RecordingLLM('{"steps": []}')
    planner = LLMTaskPlanner(llm)
    planner.plan("demo")
    assert llm.prompt is not None
    assert "'http'" in llm.prompt
    assert "GET <url>" in llm.prompt
    assert "POST <url>" in llm.prompt


class FallbackLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - unused
        if False:
            yield ""  # unreachable placeholder


def test_planner_retries_on_invalid_json() -> None:
    llm = FallbackLLM(
        [
            "not json",
            '{"steps": [{"tool": "code", "content": "print(1)"}]}',
        ]
    )
    planner = LLMTaskPlanner(llm)
    plan = planner.plan("demo")
    # Может быть выполнена финальная 3-я попытка в некоторых реализациях
    assert 2 <= len(llm.prompts) <= 3
    assert "верни валидный JSON" in llm.prompts[1]
    assert plan.steps[0] == PlanStep(tool=Tool.CODE, content="print(1)")


def test_planner_falls_back_after_two_invalid_json() -> None:
    llm = FallbackLLM(["not json", "still not json"])
    planner = LLMTaskPlanner(llm)
    plan = planner.plan("demo")
    # Может быть выполнена дополнительная попытка, допускаем до 3
    assert 2 <= len(llm.prompts) <= 3
    assert plan.steps == [PlanStep(tool=Tool.CODE, content="demo")]
    assert plan.context == ["code"]


def test_build_prompt_includes_actions_and_results() -> None:
    planner = LLMTaskPlanner(StubLLM(""))
    prompt = planner._build_prompt(
        "req",
        ["act"],
        ["res"],
    )
    assert "Previous actions:\n- act" in prompt
    assert "Known data:\n- res" in prompt
    assert prompt.endswith("Request: req\n")
