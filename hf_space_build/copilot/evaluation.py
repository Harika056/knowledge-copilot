"""Golden-set evaluation: retrieval quality (hit@k, MRR), latency percentiles, and cost per request,
compared across chunking strategies and retrieval modes."""
from __future__ import annotations
import json
import math
from pathlib import Path

from .config import ROLE_ACCESS, Settings
from .embeddings import tokenize
from .pipeline import Copilot
from .retrieval import MODES, STOP
from .store import MemoryStore

STRATEGIES = [("heading", {}), ("fixed", {"size": 60, "overlap": 10})]


def load_golden(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if "question" not in r or "expected_source" not in r:
            raise ValueError(f"golden row needs 'question' and 'expected_source': {r}")
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def evaluate_retrieval(copilot: Copilot, golden: list[dict], mode: str, k: int = 4, role: str = "ADMIN") -> dict:
    allowed, hit1, hitk, rr, lat = ROLE_ACCESS[role], 0, 0, 0.0, []
    import time
    for g in golden:
        t = time.perf_counter()
        res = copilot.retriever.search(g["question"], allowed, mode, k)
        lat.append((time.perf_counter() - t) * 1000)
        rank = next((i for i, h in enumerate(res.hits, 1) if h.chunk.source == g["expected_source"]), None)
        if rank:
            hitk += 1
            rr += 1.0 / rank
            hit1 += rank == 1
    n = len(golden)
    return {"mode": mode, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n,
            "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95), "n": n}


def groundedness(answer: str, passages: list[str]) -> float:
    """PROXY metric: share of answer sentences whose content words mostly appear in the cited passages.
    Use an LLM judge (see the evaluation toolkit project) for a real faithfulness score."""
    import re
    support = set(tokenize(" ".join(passages)))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)) if s.strip()]
    if not sentences:
        return 0.0
    ok = 0
    for s in sentences:
        words = [w for w in tokenize(s) if w not in STOP]
        ok += bool(words) and sum(w in support for w in words) / len(words) >= 0.6
    return ok / len(sentences)


def evaluate_answers(copilot: Copilot, golden: list[dict], role: str = "ADMIN") -> dict:
    lat, cost, toks, ground = [], 0.0, 0, []
    for g in golden:
        a = copilot.ask(g["question"], role)
        lat.append(a.timings_ms["total"])
        cost += a.cost_usd
        toks += a.usage.prompt_tokens + a.usage.completion_tokens
        ground.append(groundedness(a.text, [c["snippet"] for c in a.citations]))
    n = len(golden)
    return {"answer_p50_ms": percentile(lat, 50), "answer_p95_ms": percentile(lat, 95),
            "avg_tokens": toks / n, "cost_per_request_usd": cost / n, "groundedness_proxy": sum(ground) / n}


def run_ablation(embedder, llm, docs_root: str, golden: list[dict], k: int = 4,
                 settings: Settings | None = None, answers: bool = False, strategies=STRATEGIES, modes=MODES) -> list[dict]:
    rows = []
    for name, kw in strategies:
        cp = Copilot(MemoryStore(), embedder, llm, settings or Settings(), k=k)
        chunks = cp.ingest_folder(docs_root, strategy=name, **kw)
        for mode in modes:
            cp.mode = mode
            row = {"chunking": name, "chunks": chunks, **evaluate_retrieval(cp, golden, mode, k)}
            if answers:
                row.update(evaluate_answers(cp, golden))
            rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
