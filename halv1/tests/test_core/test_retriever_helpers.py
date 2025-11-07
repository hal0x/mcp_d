from retriever.retriever import Retriever


class DummyIndex:
    """Minimal index stub for composing context."""


class DummyClusterManager:
    """Minimal cluster manager stub."""


def _dummy_retriever() -> Retriever:
    return Retriever(DummyIndex(), DummyClusterManager())


def test_compose_context_formats_cards() -> None:
    retriever = _dummy_retriever()
    cards = [
        {"summary": "s1", "medoid": "m1", "fragments": ["f1", "f2"]},
        {"summary": "s2", "medoid": "m2", "fragments": ["g1"]},
    ]
    context, brief = retriever._compose_context(cards, two_pass=True)
    assert context == "- s1: m1\n  * f1\n  * f2\n\n- s2: m2\n  * g1"
    assert brief == "- s1: m1\n- s2: m2"
