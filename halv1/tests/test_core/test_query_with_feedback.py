import asyncio
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.cluster_manager import ClusterManager
from index.vector_index import VectorEntry
from retriever.query_with_feedback import query_with_feedback
from retriever.retriever import Retriever


class DummyIndex:
    def __init__(self):
        self.queries = []
        self.entries = [VectorEntry("1", "text", [1.0, 0.0], {})]

        class BM25:
            def search(self, query, top_k=25):
                return []

        self.bm25 = BM25()

    async def search(self, query, top_k=25):
        self.queries.append(query)
        return self.entries

    async def embed(self, text):
        return [1.0, 0.0]


class DummyClusterManager:
    def __init__(self):
        self.noise = []
        self.missing = []
        self.clusters = {}

    def get_insights(
        self,
        emb,
        top_k=5,
        cluster_weight=None,
        bm25_score=None,
        w5=0.1,
        w6=0.1,
    ):
        return []

    def mark_noise(self, doc_id):
        self.noise.append(doc_id)

    def log_missing_facts(self, facts):
        self.missing.append(list(facts))


class DummyLLM:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def generate(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return json.dumps(
                {
                    "answer": "a",
                    "missing_facts": ["extra"],
                    "noise_context": ["1"],
                    "relevant_ids": ["1"],
                }
            )
        return json.dumps(
            {
                "answer": "final",
                "missing_facts": [],
                "noise_context": [],
                "relevant_ids": ["1"],
            }
        )


def test_query_with_feedback_parses_json_and_calls_feedback():
    idx = DummyIndex()
    mgr = DummyClusterManager()
    retr = Retriever(idx, mgr)
    llm = DummyLLM()
    result = asyncio.run(query_with_feedback(retr, llm, "q", top_k=1))
    assert result["answer"] == "final"
    assert mgr.noise == ["1"]
    assert mgr.missing[0] == ["extra"]


def test_query_with_feedback_handles_invalid_json() -> None:
    class BadLLM(DummyLLM):
        async def generate(self, prompt):  # type: ignore[override]
            return "not json"

    idx = DummyIndex()
    mgr = DummyClusterManager()
    retr = Retriever(idx, mgr)
    llm = BadLLM()
    result = asyncio.run(query_with_feedback(retr, llm, "q", top_k=1))
    assert result == {
        "answer": "",
        "missing_facts": [],
        "noise_context": [],
        "relevant_ids": [],
    }
    assert mgr.noise == []
    assert mgr.missing == []


def test_mark_noise_reduces_weight():
    mgr = ClusterManager()
    mgr.mark_noise("d1")
    assert mgr.doc_weights["d1"] < 1.0


def test_query_with_feedback_retrieves_again_on_missing_facts():
    class Index(DummyIndex):
        async def search(self, query, top_k=25):
            self.queries.append(query)
            self.entries = [VectorEntry(query, "t", [1.0, 0.0], {})]
            return self.entries

    idx = Index()
    mgr = DummyClusterManager()
    retr = Retriever(idx, mgr)
    llm = DummyLLM()
    asyncio.run(query_with_feedback(retr, llm, "q", top_k=1))
    assert idx.queries[:2] == ["q", "extra"]
    assert llm.calls == 2
