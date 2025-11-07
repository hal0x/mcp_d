import pytest

import index.faiss_backend as fb


def test_faiss_backend_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fb, "faiss", None)
    with pytest.raises(RuntimeError, match="faiss-cpu"):
        fb.FaissBackend()
