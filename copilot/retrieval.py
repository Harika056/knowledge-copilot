"""Retrieval modes compared in the eval harness: keyword, vector, hybrid (RRF), hybrid + rerank."""
from __future__ import annotations
import time
from dataclasses import dataclass, field

from .embeddings import STOP, Embedder, tokenize
from .models import Hit, Usage
from .store import Store

MODES = ("keyword", "vector", "hybrid", "hybrid+rerank")


def rrf(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: combines rankings without needing comparable scores."""
    score: dict[str, float] = {}
    chunk = {}
    for hits in lists:
        for rank, h in enumerate(hits):
            score[h.chunk.id] = score.get(h.chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunk[h.chunk.id] = h.chunk
    return [Hit(chunk[i], s) for i, s in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


class LexicalReranker:
    """Cheap offline baseline: rewards query-term coverage, bigram matches and heading matches.
    Swap in a cross-encoder or Vertex AI ranking API for real quality; the interface is the same."""

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q = [t for t in tokenize(query) if t not in STOP]
        if not q:
            return hits
        qbi = {f"{a} {b}" for a, b in zip(q, q[1:])}
        out = []
        for h in hits:
            toks = tokenize(h.chunk.text)
            tset = set(toks)
            cover = sum(1 for t in q if t in tset) / len(q)
            bigrams = {f"{a} {b}" for a, b in zip(toks, toks[1:])}
            bi = len(qbi & bigrams) / len(qbi) if qbi else 0.0
            head = set(tokenize(h.chunk.heading))
            hd = sum(1 for t in q if t in head) / len(q)
            out.append(Hit(h.chunk, cover + 0.5 * bi + 0.5 * hd))
        return sorted(out, key=lambda x: x.score, reverse=True)


@dataclass
class Retrieval:
    hits: list[Hit]
    usage: Usage = field(default_factory=Usage)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, reranker=None, candidates: int = 20) -> None:
        self.store, self.embedder = store, embedder
        self.reranker, self.candidates = reranker or LexicalReranker(), candidates

    def search(self, query: str, allowed: frozenset[str], mode: str = "hybrid+rerank", k: int = 4) -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}', choose from {MODES}")
        t, timings, usage, lists = time.perf_counter(), {}, Usage(), []
        if mode in ("vector", "hybrid", "hybrid+rerank"):
            vecs, u = self.embedder.embed([query])
            usage.add(u)
            timings["embed"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            lists.append(self.store.vector_search(vecs[0], self.candidates, allowed))
            timings["vector_search"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
        if mode in ("keyword", "hybrid", "hybrid+rerank"):
            lists.append(self.store.keyword_search(query, self.candidates, allowed))
            timings["keyword_search"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
        hits = lists[0] if len(lists) == 1 else rrf(lists)
        if mode == "hybrid+rerank":
            hits = self.reranker.rerank(query, hits[: self.candidates])
            timings["rerank"] = (time.perf_counter() - t) * 1000
        return Retrieval(hits[:k], usage, timings)
