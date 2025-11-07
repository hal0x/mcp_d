from agent.dedup import deduplicate


def test_dedup_case_sensitive() -> None:
    items = ["Alpha", "alpha", "BETA", "beta"]
    assert deduplicate(items, case_sensitive=True) == items
