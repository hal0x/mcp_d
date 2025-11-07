from planner import LLMTaskPlanner


class EchoLLM:
    def generate(self, prompt: str) -> str:
        return "goal"


def test_refine_returns_none_when_output_unchanged():
    planner = LLMTaskPlanner(EchoLLM())
    assert planner.refine("goal", ["info"]) is None
