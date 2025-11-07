import pytest

import index.faiss_backend as fb


class _E:
    def __init__(self, emb, id_: str) -> None:
        self.embedding = emb
        self.id = id_


@pytest.mark.skipif(fb.faiss is None, reason="faiss-cpu not installed")
def test_rebuild_skips_empty_and_mismatched_dims() -> None:
    be = fb.FaissBackend()
    entries = [
        _E([1.0, 0.0, 0.0], "a"),
        _E([], "empty"),
        _E([0.0, 1.0, 0.0], "b"),
        _E([1.0, 0.0], "bad_dim"),  # mismatched dimensionality
    ]
    be.rebuild(entries, get_vector=lambda e: e.embedding)
    assert be.index is not None
    # only the 3-D vectors should be kept
    assert len(be.entries) == 2
    # basic search should work with matching dim
    scores = be.search([1.0, 0.0, 0.0], top_k=2, get_id=lambda e: e.id)
    assert set(scores.keys()) <= {"a", "b"}

