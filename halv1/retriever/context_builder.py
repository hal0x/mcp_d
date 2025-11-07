"""Utilities for assembling context prompts from cluster insights.

This module exposes :func:`build_prompt` which collects cluster insights
and high-scoring raw facts from a :class:`~index.vector_index.VectorIndex`.
It can optionally operate in a *two-pass* mode returning both a brief
summary and a detailed prompt.

Example
-------
>>> import asyncio
>>> from index.vector_index import VectorIndex
>>> from index.cluster_manager import ClusterManager
>>> prompt = asyncio.run(build_prompt("question", VectorIndex(), ClusterManager()))
>>> isinstance(prompt, str)
True
"""

from __future__ import annotations

from typing import List, Tuple, cast

from index.cluster_manager import ClusterManager
from index.vector_index import VectorIndex
from retriever.insight_utils import build_insight_cards


async def build_prompt(
    query: str,
    index: VectorIndex,
    clusters: ClusterManager,
    *,
    two_pass: bool = False,
) -> str | Tuple[str, str]:
    """Build a context prompt combining cluster insights and raw facts.

    Parameters
    ----------
    query:
        The question or search query.
    index:
        Vector index used for similarity search.
    clusters:
        Cluster manager providing pre-computed insights.
    two_pass:
        When ``True`` the function returns ``(summary, detailed)`` prompts so
        the caller can first use a concise summary and later expand with the
        detailed context.

    Returns
    -------
    str or tuple
        If ``two_pass`` is ``False`` a single detailed prompt string is
        returned.  Otherwise a tuple ``(summary, detailed)`` is produced.
    """

    # gather insight cards and raw facts from the helper
    cards, _ = await build_insight_cards(query, index, clusters)

    # assemble prompts
    detailed_parts: List[str] = []
    for card in cards:
        lines = [f"- {card['summary']}: {card['medoid']}"]
        fragments = cast(List[str], card["fragments"])
        lines.extend(f"  * {f}" for f in fragments)
        detailed_parts.append("\n".join(lines))
    detailed_prompt = "\n\n".join(detailed_parts)

    if two_pass:
        summary_prompt = "\n".join(f"- {c['summary']}: {c['medoid']}" for c in cards)
        return summary_prompt, detailed_prompt
    return detailed_prompt
