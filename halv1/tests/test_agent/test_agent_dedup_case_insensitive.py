from agent.dedup import deduplicate


def test_dedup_case_insensitive() -> None:
    items = ["Alpha", "alpha", "BETA", "beta"]
    assert deduplicate(items) == ["Alpha", "BETA"]
