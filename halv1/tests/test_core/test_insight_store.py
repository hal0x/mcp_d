import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.insight_store import InsightStore


def test_malformed_json_loads_empty(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "insights.json"
    path.write_text("{not valid", encoding="utf-8")
    store = InsightStore(str(path))
    assert store.get_clusters() == {}


def test_unreadable_path_raises(tmp_path: pathlib.Path) -> None:
    directory = tmp_path / "insights.json"
    directory.mkdir()
    with pytest.raises(OSError):
        InsightStore(str(directory))
     
def test_get_cluster_returns_copy(tmp_path: pathlib.Path) -> None:
    store = InsightStore(str(tmp_path / "insights.json"))
    store.set_cluster(
        "c1",
        {"centroid": [], "medoid": None, "summary": "hello", "members": []},
    )

    cluster = store.get_cluster("c1")
    assert cluster is not None

    cluster["summary"] = "changed"
    cluster["extra"] = 1
    fresh = store.get_cluster("c1")
    assert fresh is not None
    assert fresh["summary"] == "hello"
    assert "extra" not in fresh

    store.set_cluster("c1", cluster)
    fresh = store.get_cluster("c1")
    assert fresh is not None
    assert fresh["summary"] == "changed"
    assert "extra" in fresh
