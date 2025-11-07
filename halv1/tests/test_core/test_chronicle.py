import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.chronicle import Chronicle
from index.cluster_manager import Cluster, ClusterManager
from index.vector_index import VectorEntry


class DummySummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, texts):
        self.calls.append(list(texts))
        return " | ".join(texts)


def test_chronicle_groups_and_summarizes():
    mgr = ClusterManager()
    # create two clusters with timeline entries across three days
    e1 = VectorEntry("1", "old", [1.0, 0.0], {}, timestamp=0)
    e2 = VectorEntry("2", "new", [1.0, 0.0], {}, timestamp=86400 + 1)
    c1 = Cluster(
        id="c1",
        members=[e1, e2],
        centroid=[1.0, 0.0],
        medoid=e2,
        summary="sumA",
        timeline=[e1, e2],
    )

    e3 = VectorEntry("3", "third", [0.0, 1.0], {}, timestamp=86400 * 2 + 1)
    c2 = Cluster(
        id="c2",
        members=[e3],
        centroid=[0.0, 1.0],
        medoid=e3,
        summary="sumB",
        timeline=[e3],
    )
    mgr.clusters = {"c1": c1, "c2": c2}

    summarizer = DummySummarizer()
    chron = Chronicle(mgr, summarizer)
    text = chron.chronicle(slice_hours=24)

    assert summarizer.calls == [
        ["old", "sumA"],
        ["new", "sumA"],
        ["third", "sumB"],
    ]
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith("old | sumA") and lines[0].startswith(
        "1970-01-01T00:00:00"
    )
    assert lines[1].endswith("new | sumA") and lines[1].startswith(
        "1970-01-02T00:00:00"
    )
    assert lines[2].endswith("third | sumB") and lines[2].startswith(
        "1970-01-03T00:00:00"
    )
