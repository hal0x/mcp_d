from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List

from core.utils.json_io import parse_llm_json
from llm.base_client import LLMClient
from llm.utils import unwrap_response

from .retriever import Retriever


def _as_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    if not val:
        return []
    return [val]


def _build_context(entries: Iterable[Any]) -> str:
    lines = [f"[{e.chunk_id}] {e.text}" for e in entries]
    return "\n".join(lines)


def _parse_response(text: str) -> Dict[str, Any]:
    data = parse_llm_json(text) or {}
    return {
        "answer": data.get("answer", ""),
        "missing_facts": _as_list(data.get("missing_facts")),
        "noise_context": _as_list(data.get("noise_context")),
        "relevant_ids": _as_list(data.get("relevant_ids")),
    }


async def query_with_feedback(
    retriever: Retriever,
    llm_client: LLMClient,
    question: str,
    *,
    top_k: int = 5,
    max_iters: int = 2,
) -> Dict[str, Any]:
    """Answer ``question`` using retrieval augmented generation with feedback.

    The LLM is instructed to return JSON with keys ``answer``, ``missing_facts``,
    ``noise_context`` and ``relevant_ids``.  If ``missing_facts`` is non-empty,
    another retrieval round is performed using the new queries and the model is
    invoked again (up to ``max_iters`` iterations).
    """

    hits: List[Any] = await retriever.query(question, top_k=top_k)
    for _ in range(max_iters):
        context = _build_context(hits)
        prompt = (
            "Answer the question using the provided context. "
            "Respond in JSON with keys: answer, missing_facts, noise_context, relevant_ids.\n"
            f"Question: {question}\nContext:\n{context}"
        )
        resp = llm_client.generate(prompt)
        if inspect.isawaitable(resp):
            resp = await resp
        text, _ = unwrap_response(resp)
        parsed = _parse_response(text)
        # feedback handling
        if parsed["noise_context"]:
            for doc_id in parsed["noise_context"]:
                retriever.cluster_manager.mark_noise(doc_id)
        if parsed["missing_facts"]:
            retriever.cluster_manager.log_missing_facts(parsed["missing_facts"])
            extra_hits: List[Any] = []
            for fact in parsed["missing_facts"]:
                extra_hits.extend(await retriever.query(fact, top_k=top_k))
            hits.extend(extra_hits)
            question = question  # unchanged, but continue loop
            continue
        return parsed
    return parsed
