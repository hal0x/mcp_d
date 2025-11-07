from memory import MemoryService


def test_search_empty_query_returns_all_items(tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.write_event("alpha")
    memory.remember("beta")
    memory.remember("gamma")

    assert memory.search("", long_term=False) == ["alpha"]
    assert memory.search("", long_term=True) == ["beta", "gamma"]
    assert memory.search("") == ["alpha", "beta", "gamma"]
