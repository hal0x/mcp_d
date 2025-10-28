#!/usr/bin/env python3
"""
Evaluate quality of generated session summaries against source snippets.

For each session (summaries/<chat>/sessions/<session_id>.md), this script:
- Loads the paired snippet jsonl (summaries/<chat>/snippets/<session_id>.jsonl)
- Parses sections from the session markdown (context/discussion/decisions/risks)
- Computes simple quality metrics:
  - Structural completeness (non-empty sections, bullet counts)
  - Compression ratio (summary words vs source words)
  - ROUGE-1/2 precision/recall/F1 (simple n-gram overlap)
  - URL coverage (share of distinct URLs mentioned in summary vs source)
  - Noise flags (suspicious artifacts like fallback paths)
- Outputs an aggregate markdown report at ./summary_quality_report.md
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = ROOT / "summaries"


@dataclass
class SessionMetrics:
    chat: str
    session_id: str
    msgs: int
    src_words: int
    sum_words: int
    compression: float
    rouge1_p: float
    rouge1_r: float
    rouge1_f1: float
    rouge2_p: float
    rouge2_r: float
    rouge2_f1: float
    urls_src: int
    urls_sum: int
    url_recall: float
    context_nonempty: bool
    discussion_bullets: int
    decisions_bullets: int
    risks_bullets: int
    has_noise_artifacts: bool


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return (
        [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if len(tokens) >= n
        else []
    )


def _rouge_prf(ref_text: str, hyp_text: str, n: int = 1) -> tuple[float, float, float]:
    ref_toks = _tokenize_words(ref_text)
    hyp_toks = _tokenize_words(hyp_text)
    ref_ngr = _ngrams(ref_toks, n)
    hyp_ngr = _ngrams(hyp_toks, n)
    if not ref_ngr or not hyp_ngr:
        return (0.0, 0.0, 0.0)
    from collections import Counter

    ref_c = Counter(ref_ngr)
    hyp_c = Counter(hyp_ngr)
    overlap = sum(min(ref_c[k], hyp_c[k]) for k in hyp_c.keys() if k in ref_c)
    p = overlap / max(1, sum(hyp_c.values()))
    r = overlap / max(1, sum(ref_c.values()))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return (p, r, f1)


def _extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", text)
    # normalize trailing punctuation
    urls = [u.rstrip(").,;") for u in urls]
    return urls


def _parse_session_md(md_path: Path) -> dict[str, object]:
    text = md_path.read_text(encoding="utf-8")
    # Split sections by H2 headers
    sections = {
        "context": "",
        "discussion": [],
        "decisions": [],
        "risks": [],
        "all_text": text,
    }
    current = None
    for line in text.splitlines():
        if line.strip().startswith("## "):
            header = line.strip().lower()
            if "контекст" in header or "context" in header:
                current = "context"
                continue
            if "ход дискуссии" in header or "discussion" in header:
                current = "discussion"
                continue
            if "решения" in header or "next steps" in header or "decisions" in header:
                current = "decisions"
                continue
            if "риски" in header or "risks" in header or "open questions" in header:
                current = "risks"
                continue
            if "ссылки" in header or "артефакт" in header or "links" in header:
                # Stop collecting into the previous section
                current = None
                continue
        # collect
        if current == "context":
            sections["context"] += line + "\n"
        elif current in ("discussion", "decisions", "risks"):
            if line.strip().startswith("-"):
                sections[current].append(line.strip())

    # Extract suspicious noise artifacts
    noise = bool(
        re.search(
            r"/chats/.+unknown\.json#msg=|\*\*\*+|\)\)\)?$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )

    return {
        "context": sections["context"].strip(),
        "discussion": sections["discussion"],
        "decisions": sections["decisions"],
        "risks": sections["risks"],
        "all_text": sections["context"]
        + "\n"
        + "\n".join(sections["discussion"] + sections["decisions"] + sections["risks"]),
        "has_noise": noise,
    }


def _load_snippet_jsonl(snippet_path: Path) -> list[dict[str, object]]:
    msgs: list[dict[str, object]] = []
    with snippet_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                # allow partially printed files
                continue
    return msgs


def evaluate_session(chat_dir: Path, session_md: Path) -> SessionMetrics | None:
    session_id = session_md.stem
    chat = chat_dir.name
    snippet = chat_dir / "snippets" / f"{session_id}.jsonl"
    if not snippet.exists():
        return None

    msgs = _load_snippet_jsonl(snippet)
    src_text = "\n".join(m.get("text", "") or "" for m in msgs)
    src_words = len(_tokenize_words(src_text))

    md = _parse_session_md(session_md)
    sum_text = md["all_text"] or ""
    sum_words = len(_tokenize_words(sum_text))

    # structural
    context_nonempty = len(md["context"]) > 20  # threshold to filter placeholders
    discussion_bullets = (
        len(md["discussion"]) if isinstance(md["discussion"], list) else 0
    )
    decisions_bullets = len(md["decisions"]) if isinstance(md["decisions"], list) else 0
    risks_bullets = len(md["risks"]) if isinstance(md["risks"], list) else 0

    # compression
    compression = (sum_words / src_words) if src_words else 0.0

    # ROUGE
    r1p, r1r, r1f = _rouge_prf(src_text, sum_text, n=1)
    r2p, r2r, r2f = _rouge_prf(src_text, sum_text, n=2)

    # URL coverage
    src_urls = set(_extract_urls(src_text))
    sum_urls = set(_extract_urls(sum_text))
    url_recall = (len(sum_urls & src_urls) / len(src_urls)) if src_urls else 0.0

    return SessionMetrics(
        chat=chat,
        session_id=session_id,
        msgs=len(msgs),
        src_words=src_words,
        sum_words=sum_words,
        compression=round(compression, 3),
        rouge1_p=round(r1p, 3),
        rouge1_r=round(r1r, 3),
        rouge1_f1=round(r1f, 3),
        rouge2_p=round(r2p, 3),
        rouge2_r=round(r2r, 3),
        rouge2_f1=round(r2f, 3),
        urls_src=len(src_urls),
        urls_sum=len(sum_urls),
        url_recall=round(url_recall, 3),
        context_nonempty=context_nonempty,
        discussion_bullets=discussion_bullets,
        decisions_bullets=decisions_bullets,
        risks_bullets=risks_bullets,
        has_noise_artifacts=bool(md.get("has_noise")),
    )


def grade(metrics: SessionMetrics) -> str:
    """Assign a coarse quality grade based on heuristics."""
    # Strong signals of poor summary
    if (
        not metrics.context_nonempty
        and metrics.discussion_bullets == 0
        and metrics.decisions_bullets == 0
    ):
        return "poor"
    if metrics.compression < 0.01 or metrics.rouge1_r < 0.05:
        return "poor"
    # Medium if some structure but low coverage
    if metrics.rouge1_r < 0.15 or metrics.discussion_bullets < 2:
        return "fair"
    # Good if decent recall and structure
    if metrics.rouge1_r >= 0.3 and metrics.discussion_bullets >= 3:
        return "good"
    return "fair"


def main() -> None:
    sessions: list[SessionMetrics] = []
    for chat_dir in sorted(SUMMARIES_DIR.iterdir()):
        if not chat_dir.is_dir():
            continue
        sessions_dir = chat_dir / "sessions"
        if not sessions_dir.exists():
            continue
        for md_path in sorted(sessions_dir.glob("*.md")):
            m = evaluate_session(chat_dir, md_path)
            if m:
                sessions.append(m)

    # Aggregate
    total = len(sessions)
    poor = sum(1 for m in sessions if grade(m) == "poor")
    fair = sum(1 for m in sessions if grade(m) == "fair")
    good = sum(1 for m in sessions if grade(m) == "good")

    def avg(attr):
        return sum(getattr(m, attr) for m in sessions) / total if total else 0.0

    report = []
    report.append("# Summary Quality Report")
    report.append("")
    report.append(f"Sessions evaluated: {total}")
    report.append(f"Grades → good: {good}, fair: {fair}, poor: {poor}")
    report.append("")
    report.append("## Averages")
    report.append(f"- Compression: {avg('compression'):.3f}")
    report.append(
        f"- ROUGE-1 F1: {avg('rouge1_f1'):.3f} (P {avg('rouge1_p'):.3f} / R {avg('rouge1_r'):.3f})"
    )
    report.append(
        f"- ROUGE-2 F1: {avg('rouge2_f1'):.3f} (P {avg('rouge2_p'):.3f} / R {avg('rouge2_r'):.3f})"
    )
    report.append(f"- URL recall: {avg('url_recall'):.3f}")
    report.append(f"- Discussion bullets: {avg('discussion_bullets'):.2f}")
    report.append(f"- Decisions bullets: {avg('decisions_bullets'):.2f}")
    report.append(f"- Risks bullets: {avg('risks_bullets'):.2f}")
    report.append("")
    report.append("## Notable Issues")
    noisy = [m for m in sessions if m.has_noise_artifacts]
    empty = [
        m
        for m in sessions
        if (
            not m.context_nonempty
            and m.discussion_bullets == 0
            and m.decisions_bullets == 0
        )
    ]
    report.append(f"- Sessions with noise artifacts: {len(noisy)}")
    report.append(f"- Sessions with empty structure: {len(empty)}")
    report.append("")
    report.append("## Samples (up to 20)")
    for m in sessions[:20]:
        g = grade(m)
        report.append(
            f"- {m.chat}/{m.session_id}: grade={g}, msgs={m.msgs}, comp={m.compression:.3f}, R1F1={m.rouge1_f1:.3f}, "
            f"disc={m.discussion_bullets}, dec={m.decisions_bullets}, noise={m.has_noise_artifacts}"
        )

    out = ROOT / "summary_quality_report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {out}")
    print(f"good={good} fair={fair} poor={poor} (total={total})")


if __name__ == "__main__":
    main()
