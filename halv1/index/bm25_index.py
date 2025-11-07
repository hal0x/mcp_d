"""Simple BM25 inverted index."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from core.utils.json_io import load_json, save_json

from .preprocess import normalize_text


class BM25Index:
    """A very small BM25 implementation storing ``term -> doc`` mapping.

    The index persists its data to a JSON file. It is intentionally
    lightweight and is designed for the small datasets used in tests.
    """

    def __init__(
        self,
        path: str = "db/index/bm25.json",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.path = Path(path)
        self.k1 = k1
        self.b = b
        self.inverted: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.doc_len: Dict[str, int] = {}
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        data: Dict[str, Any] = load_json(self.path, {})
        self.inverted = defaultdict(
            dict,
            {
                t: {doc: int(cnt) for doc, cnt in docs.items()}
                for t, docs in data.get("inverted", {}).items()
            },
        )
        self.doc_len = {
            doc: int(length) for doc, length in data.get("doc_len", {}).items()
        }

    def _save(self) -> None:
        data = {"inverted": self.inverted, "doc_len": self.doc_len}
        save_json(self.path, data)

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        norm = normalize_text(text)
        return [t for t in norm.split() if t]

    def _add_single(self, doc_id: str, tokens: Iterable[str]) -> None:
        # remove old references
        if doc_id in self.doc_len:
            for term, docs in list(self.inverted.items()):
                if doc_id in docs:
                    del docs[doc_id]
                if not docs:
                    del self.inverted[term]
        tokens_list = list(tokens)
        self.doc_len[doc_id] = len(tokens_list)
        for tok in tokens_list:
            docs = self.inverted[tok]
            docs[doc_id] = docs.get(doc_id, 0) + 1

    def add(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        self._add_single(doc_id, tokens)
        self._save()

    def add_many(self, items: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in items:
            tokens = self._tokenize(text)
            self._add_single(doc_id, tokens)
        self._save()

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 25) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
        N = len(self.doc_len)
        if N == 0 or not tokens:
            return []
        avgdl = sum(self.doc_len.values()) / N
        scores: Dict[str, float] = defaultdict(float)
        for term in tokens:
            docs = self.inverted.get(term)
            if not docs:
                continue
            df = len(docs)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for doc_id, freq in docs.items():
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score = idf * (freq * (self.k1 + 1) / denom)
                scores[doc_id] += score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
