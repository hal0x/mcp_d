from __future__ import annotations

from typing import Protocol, Sequence, Tuple


class _Predictor(Protocol):
    def predict(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[float]:
        """Return scores for query-document pairs."""


class CrossEncoder:
    """Wrapper around a sentence-transformers style cross encoder.

    Parameters
    ----------
    model:
        Pre-initialized model providing a ``predict`` method. If omitted,
        ``sentence_transformers.CrossEncoder`` is instantiated using
        ``model_name``.
    model_name:
        Name of the model to load when ``model`` is ``None``.
    """

    def __init__(
        self,
        model: _Predictor | None = None,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        if model is not None:
            predictor: _Predictor = model
        else:
            try:
                from sentence_transformers import CrossEncoder as STCrossEncoder
            except Exception as exc:  # pragma: no cover - import-time fallback
                raise RuntimeError(
                    "sentence-transformers is required when no model is provided"
                ) from exc
            predictor = STCrossEncoder(model_name)
        self.model = predictor

    def score(self, query: str, document: str) -> float:
        """Return relevance score for ``query`` and ``document``."""

        scores = self.model.predict([(query, document)])
        return float(scores[0])
